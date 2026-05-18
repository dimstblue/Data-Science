# Analisis Sentimen Komentar YouTube #PESTABABI

Proyek analisis teks berbasis komentar YouTube terkait isu **#PESTABABI** menggunakan pendekatan Data Science end-to-end: dari data cleaning, EDA, word cloud, hingga model klasifikasi sentimen.

---

## Struktur Proyek

```
pestababi_project/
├── data/
│   ├── raw/                          # Data mentah asli (jangan dimodifikasi)
│   │   ├── PESTABABI_-_Data_Mentah.csv
│   │   └── PESTABABI_-_Daftar_Video.csv
│   └── processed/                    # Output dari setiap notebook
│       ├── pestababi_clean.csv
│       ├── pestababi_clean_preprocessed.csv
│       ├── pestababi_final_labeled.csv
│       ├── best_model.pkl
│       ├── model_comparison.csv
│       └── (berbagai file plot .png)
├── notebooks/
│   ├── 1_data_cleaning.ipynb         # Merge, clean, typing
│   ├── 2_eda_and_wordcloud.ipynb     # EDA, time series, word cloud
│   └── 3_modelling.ipynb             # Auto-labeling, training, evaluasi
├── README.md
└── requirements.txt
```

---

## Dataset

| File | Baris | Kolom | Deskripsi |
|---|---|---|---|
| `PESTABABI_-_Daftar_Video.csv` | 25 | 4 | Metadata 25 video YouTube |
| `PESTABABI_-_Data_Mentah.csv` | 1.555 | 3 | Komentar dari seluruh video |

### Skema Kolom Setelah Merge

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Video_ID` | str | ID unik video YouTube |
| `Komentar` | str | Teks komentar mentah |
| `Jumlah_Suka` | int | Jumlah likes pada komentar |
| `Judul_Video` | str | Judul video |
| `Channel` | str | Nama channel YouTube |
| `Tanggal_Rilis` | datetime | Tanggal rilis video |
| `Panjang_Komentar` | int | Panjang karakter komentar |
| `Jumlah_Kata` | int | Jumlah kata komentar |
| `Kategori_Likes` | str | none / low / medium / high |
| `Komentar_Bersih` | str | Teks setelah preprocessing |
| `Sentimen` | str | positif / negatif / netral |

---

## Cara Menjalankan

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan Jupyter
jupyter lab

# 4. Jalankan notebook secara berurutan
#    1_data_cleaning.ipynb → 2_eda_and_wordcloud.ipynb → 3_modelling.ipynb
```

> **Penting**: Jalankan notebook secara berurutan. Notebook 2 & 3 bergantung pada output notebook 1.

---

## Metodologi

### 1. Data Merging & Cleaning
- Left join `Data_Mentah` ← `Daftar_Video` via `Video_ID`
- Drop missing values pada kolom krusial & komentar kosong
- Drop duplikat berdasarkan `(Video_ID, Komentar)`
- Type casting: `Tanggal_Rilis` → `datetime`, `Jumlah_Suka` → `int`
- Feature engineering: `Panjang_Komentar`, `Jumlah_Kata`, `Kategori_Likes`

### 2. EDA
- Analisis video & channel terpopuler (komentar & likes)
- Time series jumlah komentar vs tanggal rilis video
- Distribusi interaksi (likes) dengan pie chart & histogram

### 3. Text Preprocessing
Pipeline (berurutan):
1. Lowercase
2. Hapus URL (regex)
3. Hapus emoji & karakter non-ASCII
4. Hapus tanda baca & angka
5. Hapus spasi berlebih
6. Stopword removal (Sastrawi + NLTK + custom slang)
7. *(Opsional)* Stemming Sastrawi

Word Cloud yang dibuat:
- WC Umum (semua komentar)
- WC Perbandingan: Likes Tinggi vs 0 Likes
- WC Perbandingan: Media Resmi vs Channel Independen

### 4. Modelling — Klasifikasi Sentimen
- **Auto-labeling**: Lexicon-based (InSet-style, Bahasa Indonesia)
- **Feature extraction**: TF-IDF (unigram + bigram, max 5000 fitur)
- **Algoritma**: Naive Bayes, Logistic Regression, Linear SVM, Random Forest
- **Evaluasi**: Accuracy, F1 Weighted, F1 Macro, Confusion Matrix
- **Pipeline**: Scikit-learn Pipeline untuk mencegah data leakage

---

## Hasil & Insight Utama

Berdasarkan hasil analisis dari **10.490 komentar unik** yang dikumpulkan dari 25 video YouTube terkait isu #PestaBabi:

- **Video/Channel paling banyak komentar**: Video *"PESTA BABI (Official Trailer)"* dari channel **Indonesia Baru** mendominasi dengan **5.274 komentar** (50% dari total data), diikuti channel Pari Kesit (1.730) dan Tempodotco (716).

- **Puncak diskusi publik**: Lonjakan terbesar terjadi pada **13 Maret 2026** dengan 5.274 komentar sekaligus — bertepatan dengan unggahan trailer resmi oleh channel Indonesia Baru. Gelombang diskusi kedua muncul pada **12–16 Mei 2026** seiring bertambahnya video reaksi dan ulasan dari channel lain.

- **Sentimen dominan**: Mayoritas komentar bersifat **Netral (78,3%)**, diikuti Positif (11,0%) dan Negatif (10,7%). Dominasi netral mengindikasikan bahwa publik cenderung memberikan komentar informatif atau deskriptif dibanding ekspresi emosional tegas terhadap isu ini.

- **Model terbaik**: **Linear SVM** mencapai performa tertinggi dengan Accuracy **97,01%** dan F1-Score Weighted **96,99%**, unggul tipis dari Logistic Regression (96,73%). Naive Bayes menjadi model dengan performa terendah (86,20%).

- **Kata paling dominan**: Kata **"papua"** (3.192 kemunculan) mendominasi secara signifikan, diikuti "rakyat" (1.241), "indonesia" (1.232), "negara" (1.057), dan "film" (1.043) — mencerminkan bahwa diskusi publik berpusat pada isu kedaulatan, identitas, dan konteks sinematik film tersebut.

---

## Referensi

- Ibrohim, M. O., & Budi, I. (2019). *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter.* ALW3.
- Sastrawi Python Library: https://github.com/har07/PySastrawi
- Scikit-learn Documentation: https://scikit-learn.org
- Auto Labelling: https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection.git

---

*Project oleh: Muhammad Dimas Alfathir — Mahasiswa Perpustakaan & Sains Informasi, UIN Malang*
