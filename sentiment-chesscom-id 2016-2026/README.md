# Analisis Sentimen Ulasan Chess.com Indonesia

Project ini berisi pipeline analisis sentimen untuk ulasan aplikasi Chess.com di Google Play Store wilayah Indonesia. Alur kerja project mencakup scraping data ulasan, preprocessing teks bahasa Indonesia, analisis tren, topic modelling, pelatihan model klasifikasi sentimen, dan prediksi sentimen dari input baru.

Sentimen dibagi menjadi 3 kelas berdasarkan rating:

- `0` = Negatif, untuk rating 1-2
- `1` = Netral, untuk rating 3
- `2` = Positif, untuk rating 4-5

## Tujuan Project

- Mengumpulkan ulasan Chess.com dari Google Play Store untuk wilayah Indonesia.
- Membersihkan teks ulasan dengan pipeline NLP bahasa Indonesia.
- Menganalisis tren volume ulasan dan rating dari waktu ke waktu.
- Mengekstraksi topik utama dalam ulasan menggunakan LDA.
- Membandingkan beberapa model machine learning untuk klasifikasi sentimen.
- Menyediakan program prediksi sentimen interaktif berbasis model terbaik.

## Struktur Project

```text
sentiment-chesscom-id/
|-- data/
|   |-- raw/
|   |   `-- ulasan_chess_10_tahun_terakhir.csv
|   |-- processed/
|   |   `-- ulasan_chess_cleaned.csv
|   |-- kamus_slang.csv
|   `-- list_1.0.0.txt
|-- models/
|   |-- best_model.pkl
|   |-- tfidf_vectorizer.pkl
|   |-- rf_model.pkl
|   |-- rf_tfidf.pkl
|   |-- nb_model.pkl
|   |-- nb_tfidf.pkl
|   |-- svm_model.pkl
|   `-- svm_tfidf.pkl
|-- notebooks/
|   `-- eda_exploration.ipynb
|-- reports/
|   `-- figures/
|-- src/
|   |-- 1_preprocessing.py
|   |-- 2_trend_analysis.py
|   |-- 3_topic_modelling.py
|   |-- 4_sentiment_analysis.py
|   |-- 5_predict.py
|   |-- cari_slang_baru.py
|   `-- uji_optimal_topik.py
|-- scraper_playstore.py
|-- requirements.txt
`-- README.md
```

## Dataset

Data utama berasal dari ulasan aplikasi Chess.com di Google Play Store dengan package ID:

```text
com.chess
```

Kolom utama pada dataset mentah:

- `id`: ID ulasan
- `username`: nama pengguna
- `komentar`: isi ulasan
- `country`: kode negara
- `rate`: rating aplikasi dari 1 sampai 5
- `waktu`: waktu ulasan dibuat

Dataset hasil preprocessing disimpan pada:

```text
data/processed/ulasan_chess_cleaned.csv
```

Kolom tambahan penting:

- `komentar_bersih`: teks ulasan yang sudah dibersihkan dan dinormalisasi

## Metodologi

### 1. Scraping Data

File:

```bash
python scraper_playstore.py
```

Script ini mengambil ulasan aplikasi Chess.com dari Google Play Store dengan parameter:

- Bahasa: Indonesia (`id`)
- Negara: Indonesia (`id`)
- Urutan: ulasan terbaru
- Rentang data: mulai dari 1 Januari 2016

Output scraping disimpan sebagai CSV ulasan mentah.

### 2. Preprocessing Teks

File:

```bash
python src/1_preprocessing.py
```

Tahapan preprocessing:

- Case folding
- Menghapus angka, simbol, emoji, dan karakter non-huruf
- Menghapus baris baru dan spasi berlebih
- Normalisasi slang menggunakan `data/kamus_slang.csv`
- Stopword removal menggunakan Sastrawi dan stopword kustom

Output:

```text
data/processed/ulasan_chess_cleaned.csv
```

### 3. Analisis Tren

File:

```bash
python src/2_trend_analysis.py
```

Analisis yang dilakukan:

- Tren volume ulasan bulanan
- Rata-rata rating tahunan

Output grafik:

```text
reports/figures/1_tren_volume_bulanan.png
reports/figures/2_tren_rating_tahunan.png
```

### 4. Topic Modelling

File:

```bash
python src/3_topic_modelling.py
```

Topic modelling dilakukan menggunakan:

- `CountVectorizer`
- `LatentDirichletAllocation` atau LDA

Jumlah topik default pada script utama adalah 3 topik.

Untuk menguji jumlah topik optimal:

```bash
python src/uji_optimal_topik.py
```

Output evaluasi:

```text
reports/figures/3_evaluasi_perplexity.png
```

### 5. Analisis Sentimen

File:

```bash
python src/4_sentiment_analysis.py
```

Model yang dibandingkan:

- Random Forest
- Naive Bayes
- Support Vector Machine atau SVM

Fitur teks diekstraksi menggunakan TF-IDF dengan maksimal 5.000 fitur. Data dibagi menjadi data latih dan data uji dengan rasio 80:20 serta stratifikasi label.

Output model:

```text
models/rf_model.pkl
models/rf_tfidf.pkl
models/nb_model.pkl
models/nb_tfidf.pkl
models/svm_model.pkl
models/svm_tfidf.pkl
models/best_model.pkl
models/tfidf_vectorizer.pkl
```

### 6. Prediksi Sentimen Interaktif

File:

```bash
python src/5_predict.py
```

Program ini memuat:

- `models/best_model.pkl`
- `models/tfidf_vectorizer.pkl`

Lalu pengguna dapat mengetik ulasan baru dan sistem akan mengembalikan prediksi:

- Negatif
- Netral
- Positif

Untuk keluar dari program, ketik:

```text
keluar
```

atau:

```text
exit
```

## Instalasi

Gunakan Python 3.10 atau versi yang kompatibel, lalu buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependensi:

```bash
pip install pandas matplotlib seaborn scikit-learn Sastrawi google-play-scraper
```

Jika `requirements.txt` sudah dilengkapi, dependensi dapat di-install dengan:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Pipeline

Jalankan perintah dari root folder project.

```bash
python scraper_playstore.py
python src/1_preprocessing.py
python src/2_trend_analysis.py
python src/uji_optimal_topik.py
python src/3_topic_modelling.py
python src/4_sentiment_analysis.py
python src/5_predict.py
```

Jika data mentah dan model sudah tersedia, scraping dan training dapat dilewati sesuai kebutuhan.

## Output Project

Output utama project ini meliputi:

- Dataset bersih hasil preprocessing
- Grafik tren ulasan dan rating
- Grafik evaluasi jumlah topik LDA
- Hasil ekstraksi topik utama dari ulasan
- Model klasifikasi sentimen dalam format `.pkl`
- Program prediksi sentimen interaktif

Contoh file visualisasi yang tersedia:

```text
reports/figures/eda_sentimen_chesscom.png
reports/figures/eda_wordcloud_sentimen.png
reports/figures/eda_tren_makro_tahunan.png
reports/figures/eda_kata_negatif.png
reports/figures/eda_bigram_negatif.png
```

## Catatan

- Beberapa script masih menggunakan path absolut Windows pada bagian tertentu. Jika project dipindahkan ke komputer atau folder lain, sesuaikan path tersebut atau ubah menjadi relative path.
- Label sentimen dibuat dari rating pengguna, sehingga label mencerminkan asumsi bahwa rating rendah berarti negatif dan rating tinggi berarti positif.
- Kualitas prediksi sangat bergantung pada kelengkapan kamus slang, distribusi data, dan kualitas teks ulasan.

## Teknologi yang Digunakan

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Sastrawi
- Google Play Scraper

## Author

Muhammad Dimas Alfathir

Project portofolio data science untuk analisis sentimen ulasan aplikasi Chess.com regional Indonesia.
