# Analisis Sentimen Komentar YouTube Program MBG

Project ini melakukan analisis sentimen komentar YouTube terkait program **MBG (Makan Bergizi Gratis)**. Alur kerja utama ada di notebook `notebooks/sentiment_analysis_mbg.ipynb`, mulai dari EDA, preprocessing teks, auto-labeling sentimen, training model machine learning, evaluasi, visualisasi, sampai penyimpanan model.

**Author:** Muhammad Dimas Alfathir  
**Program Studi:** Perpustakaan dan Sains Informasi, UIN Malang  
**Tahun:** 2026

---

## Ringkasan Project

Dataset komentar YouTube diproses menggunakan teknik NLP Bahasa Indonesia:

- Cleaning komentar dari URL, mention, hashtag, emoji, angka, dan tanda baca.
- Case folding, tokenizing, stopword removal, dan stemming menggunakan NLTK serta Sastrawi.
- Auto-labeling sentimen dengan model transformer Bahasa Indonesia.
- Klasifikasi sentimen menggunakan TF-IDF dan empat model machine learning klasik.
- Evaluasi model menggunakan Accuracy dan F1-Score weighted.
- Visualisasi distribusi data, distribusi sentimen, evaluasi model, dan WordCloud.

Label sentimen yang digunakan:

- `Negatif`
- `Netral`
- `Positif`

---

## Struktur Folder

```text
sentiment-mbg/
|
|-- data/
|   |-- raw/
|   |   `-- Dataset MBG - Sheet1.csv
|   |
|   `-- processed/
|       |-- dataset_preprocessed.csv
|       `-- dataset_labeled.csv
|
|-- models/
|   |-- best_model.pkl
|   |-- label_encoder.pkl
|   |-- logistic_regression.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest.pkl
|   |-- svm_linearsvc.pkl
|   |
|   `-- src/
|       `-- preprocessing.py
|
|-- notebooks/
|   `-- sentiment_analysis_mbg.ipynb
|
|-- outputs/
|   |-- figures/
|   |   |-- distribusi_sentimen.png
|   |   |-- eda_distribusi.png
|   |   |-- eda_panjang_komentar.png
|   |   |-- evaluasi_model.png
|   |   |-- wordcloud.png
|   |   |-- wordcloud_sentimen.png
|   |   |-- wordcloud_positif.png
|   |   |-- wordcloud_netral.png
|   |   `-- wordcloud_negatif.png
|   |
|   `-- reports/
|       `-- hasil_evaluasi_model.csv
|
|-- requirements.txt
`-- README.md
```

---

## Dataset

File dataset utama:

```text
data/raw/Dataset MBG - Sheet1.csv
```

Ukuran data:

| Dataset | Jumlah Baris | Jumlah Kolom | Keterangan |
|---|---:|---:|---|
| `data/raw/Dataset MBG - Sheet1.csv` | 22,929 | 8 | Dataset mentah komentar YouTube |
| `data/processed/dataset_preprocessed.csv` | 21,657 | 13 | Dataset setelah preprocessing |
| `data/processed/dataset_labeled.csv` | 21,657 | 15 | Dataset setelah diberi label sentimen |

Distribusi label pada `dataset_labeled.csv`:

| Sentimen | Jumlah |
|---|---:|
| Negatif | 14,270 |
| Positif | 3,825 |
| Netral | 3,562 |

---

## Pipeline Analisis

| Step | Tahap | Output Utama |
|---:|---|---|
| 0 | Install library | Dependensi Python siap digunakan |
| 1 | Load library dan data | Dataset mentah dimuat |
| 2 | Exploratory Data Analysis | Grafik EDA dan ringkasan data |
| 3 | Feature selection | Kolom komentar dipilih sebagai fitur utama |
| 4 | Data cleaning | Kolom `komentar_clean` |
| 5 | Case folding | Kolom `komentar_lower` |
| 6 | Tokenizing | Kolom `tokens` |
| 7 | Stopword removal | Kolom `tokens_clean` |
| 8 | Stemming | Kolom `tokens_stemmed` dan `teks_final` |
| 9 | Auto-labeling sentimen | Kolom `sentimen` dan `sentimen_score` |
| 10 | Encoding label | Label sentimen menjadi numerik |
| 11 | Pipeline model | TF-IDF + classifier |
| 12 | Training dan evaluasi | Accuracy, F1-Score, confusion matrix |
| 13 | Prediksi komentar baru | Fungsi prediksi sentimen |
| 14 | Simpan model | File `.pkl` di folder `models/` |

---

## Model yang Dibandingkan

Semua model menggunakan pipeline:

```text
TF-IDF Vectorizer + Classifier
```

Model yang dilatih:

- Logistic Regression
- Naive Bayes
- Random Forest
- SVM dengan `LinearSVC`

Hasil evaluasi akhir:

| Model | Accuracy | F1-Score Weighted |
|---|---:|---:|
| SVM (LinearSVC) | 0.7013 | 0.7063 |
| Random Forest | 0.7053 | 0.6920 |
| Logistic Regression | 0.6740 | 0.6911 |
| Naive Bayes | 0.6988 | 0.6144 |

Berdasarkan **F1-Score weighted**, model terbaik adalah:

```text
SVM (LinearSVC)
```

Model terbaik disimpan sebagai:

```text
models/best_model.pkl
```

---

## Visualisasi Output

Visualisasi yang dihasilkan notebook:

| File | Keterangan |
|---|---|
| `outputs/figures/eda_distribusi.png` | Visualisasi awal distribusi data |
| `outputs/figures/eda_panjang_komentar.png` | Distribusi panjang komentar |
| `outputs/figures/distribusi_sentimen.png` | Distribusi label sentimen |
| `outputs/figures/evaluasi_model.png` | Perbandingan metrik model dan confusion matrix |
| `outputs/figures/wordcloud.png` | WordCloud seluruh komentar setelah preprocessing |
| `outputs/figures/wordcloud_sentimen.png` | WordCloud gabungan berdasarkan label sentimen |
| `outputs/figures/wordcloud_positif.png` | WordCloud khusus sentimen positif |
| `outputs/figures/wordcloud_netral.png` | WordCloud khusus sentimen netral |
| `outputs/figures/wordcloud_negatif.png` | WordCloud khusus sentimen negatif |

---

## Cara Menjalankan Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan notebook

Buka notebook berikut melalui VS Code, Jupyter Notebook, atau JupyterLab:

```text
notebooks/sentiment_analysis_mbg.ipynb
```

Atau jalankan:

```bash
jupyter notebook notebooks/sentiment_analysis_mbg.ipynb
```

### 3. Jalankan cell berurutan

Jalankan notebook dari atas ke bawah agar variabel seperti `df_labeled`, `X_train`, `y_test`, `trained_models`, dan `BEST_MODEL_NAME` tersedia untuk cell berikutnya.

Catatan:

- Step auto-labeling dengan transformer dapat memakan waktu lebih lama.
- Jika hanya ingin memakai data yang sudah diproses, gunakan file `data/processed/dataset_labeled.csv`.
- Output visualisasi akan tersimpan di `outputs/figures/`.
- Output evaluasi model akan tersimpan di `outputs/reports/hasil_evaluasi_model.csv`.

---

## Dependensi Utama

Project ini menggunakan:

- `pandas`, `numpy` untuk manipulasi data.
- `matplotlib`, `seaborn`, `wordcloud` untuk visualisasi.
- `nltk`, `PySastrawi`, `regex` untuk preprocessing teks Bahasa Indonesia.
- `transformers`, `torch`, `sentencepiece` untuk auto-labeling berbasis transformer.
- `scikit-learn`, `imbalanced-learn`, `joblib` untuk modeling dan penyimpanan model.
- `notebook`, `jupyterlab`, `ipykernel`, `ipywidgets` untuk eksekusi notebook.

Detail lengkap tersedia di:

```text
requirements.txt
```

---

## File Penting

| File | Fungsi |
|---|---|
| `notebooks/sentiment_analysis_mbg.ipynb` | Notebook utama end-to-end |
| `data/processed/dataset_labeled.csv` | Dataset final dengan label sentimen |
| `models/best_model.pkl` | Model terbaik berdasarkan F1-Score weighted |
| `models/label_encoder.pkl` | Encoder label sentimen |
| `models/src/preprocessing.py` | Fungsi preprocessing modular |
| `outputs/reports/hasil_evaluasi_model.csv` | Ringkasan hasil evaluasi model |

---

## Author

**Muhammad Dimas Alfathir**  
Perpustakaan dan Sains Informasi  
UIN Malang
