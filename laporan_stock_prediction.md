# Proyek Machine Learning Terapan: Time Series Forecasting Harga Saham dengan LSTM

## 1. Domain Proyek

### Latar Belakang
Perkembangan pasar saham menjadi topik penting dalam dunia ekonomi dan keuangan. Prediksi harga saham merupakan tantangan yang kompleks karena banyaknya variabel yang memengaruhi pasar. Dengan meningkatnya ketersediaan data historis pasar saham dan kemajuan teknologi, pendekatan machine learning—khususnya deep learning—dapat menjadi alat yang andal untuk meramalkan pergerakan harga saham.

### Pentingnya Masalah
Investor, analis pasar, dan pelaku bisnis membutuhkan sistem prediksi yang akurat untuk membantu pengambilan keputusan investasi. Prediksi yang tepat dapat meminimalkan risiko dan memaksimalkan keuntungan. Oleh karena itu, pengembangan model prediktif berbasis machine learning untuk time series forecasting menjadi penting.

### Referensi
- Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques. Expert Systems with Applications.
- Brownlee, J. (2017). Deep Learning for Time Series Forecasting. Machine Learning Mastery.

## 2. Business Understanding

### Problem Statements
- Bagaimana memprediksi harga penutupan saham (`CloseUSD`) berdasarkan data historis?
- Seberapa akurat baseline model seperti moving average dibandingkan dengan model deep learning seperti LSTM?

### Goals
- Membangun model time series untuk meramalkan harga penutupan saham (CloseUSD).
- Membandingkan kinerja model baseline dan LSTM dalam hal akurasi.

### Solution Statement
- Menggunakan metode baseline Moving Average sebagai pembanding awal.
- Membangun model LSTM untuk time series forecasting.
- Mengukur kinerja kedua model dengan metrik MAE dan RMSE.

## 3. Data Understanding

### Deskripsi Data
- Sumber: [Kaggle - Stock Exchange Data](https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data)
- Jumlah sampel: 104.224 baris data
- Fitur:
  - `Index`: Indeks bursa saham
  - `Date`: Tanggal pencatatan
  - `Open`: Harga pembukaan
  - `High`: Harga tertinggi
  - `Low`: Harga terendah
  - `Close`: Harga penutupan
  - `Adj Close`: Harga penutupan yang disesuaikan
  - `Volume`: Volume perdagangan
  - `CloseUSD`: Harga penutupan dalam USD

### Kondisi Data
- Tidak terdapat nilai **missing** berdasarkan pengecekan `df.isnull().sum()`.
- Tidak ditemukan data duplikat berdasarkan `df.duplicated().sum()`.
- Outlier secara eksplisit tidak dihapus, namun karena penggunaan MinMaxScaler pada tahap preprocessing, efeknya diminimalisasi untuk pemodelan.

### EDA & Visualisasi
- Distribusi nilai CloseUSD menunjukkan fluktuasi dinamis.
- Korelasi antar fitur menunjukkan hubungan yang kuat antara `Open`, `Close`, dan `CloseUSD`.

## 4. Data Preparation

### Teknik Persiapan
- Filter data hanya untuk indeks 'NSEI' (India).
- Pilih kolom yang relevan untuk analisis: `CloseUSD` dan `Date`.
- Konversi kolom `Date` menjadi datetime dan urutkan berdasarkan tanggal (`df_nsei.sort_values('Date')`).
- Set kolom `Date` sebagai indeks (`df_nsei.set_index('Date')`).
- Normalisasi fitur `CloseUSD` menggunakan `MinMaxScaler`.
- Bagi data menjadi data latih dan data uji dengan proporsi 80:20.

### Alasan
- Urutan waktu penting dalam time series, maka perlu pengurutan dan pengindeksan yang benar.
- Filter indeks agar fokus pada satu pasar.
- Normalisasi diperlukan agar LSTM dapat belajar secara efektif dari data skala kecil.

## 5. Modeling

### Baseline Model: Moving Average
- Window size: 10
- Cara kerja: Moving Average memprediksi harga berikutnya berdasarkan rata-rata dari beberapa harga sebelumnya (dalam hal ini 10 hari terakhir).

### Model Deep Learning: LSTM
- Input: Sekuens 10 hari sebelumnya (`lookback=10`)
- Arsitektur:
  - 1 layer LSTM dengan 50 unit
  - 1 Dense layer untuk prediksi output
- Loss function: MSE
- Optimizer: Adam
- Epochs: 10
- Batch size: 32

### Cara Kerja LSTM
LSTM merupakan jenis Recurrent Neural Network (RNN) yang mampu mengingat informasi jangka panjang dalam data sekuensial. LSTM memiliki struktur khusus berupa **gates** (input, forget, output) dan **cell state** yang membantu mengatur informasi mana yang perlu disimpan atau dilupakan.

### Kelebihan & Kekurangan
- **Moving Average**: Mudah diimplementasikan, namun tidak bisa menangkap pola kompleks.
- **LSTM**: Mampu menangkap pola temporal jangka panjang, namun membutuhkan lebih banyak data dan waktu pelatihan.

## 6. Evaluation

### Metrik Evaluasi
- **MAE (Mean Absolute Error)**:
  \[
  MAE = rac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **RMSE (Root Mean Squared Error)**:
  \[
  RMSE = \sqrt{rac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]

### Hasil Evaluasi

| Model           | MAE    | RMSE   |
|----------------|--------|--------|
| Moving Average | 0.0081 | 0.0117 |
| Dynamic MA     | 0.0164 | 0.0238 |
| LSTM           | 0.0183 | 0.0253 |

Model Moving Average memberikan performa lebih baik dibandingkan LSTM pada dataset ini.

## 7. Kesimpulan

- **Model Moving Average** menunjukkan performa lebih baik dari LSTM.
- Hal ini bisa disebabkan oleh:
  - Konfigurasi LSTM yang masih sederhana.
  - Kurangnya fitur tambahan.
  - Data historis terbatas yang digunakan untuk pelatihan.

Meskipun LSTM dirancang untuk data sekuensial, hasil proyek ini menunjukkan pentingnya membandingkan model kompleks dengan baseline sederhana.

## 8. Saran Pengembangan

- Melakukan hyperparameter tuning lebih lanjut.
- Menggunakan data multivariat.
- Menambahkan indikator teknikal.

---
