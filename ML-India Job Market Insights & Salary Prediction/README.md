# India Job Market Insights & Salary Prediction

Proyek ini menganalisis pasar kerja India periode 2024-2026 dan membangun model Machine Learning untuk memprediksi gaji dalam satuan Lakhs Per Annum (LPA). Analisis dilakukan dari eksplorasi data, visualisasi tren, pemodelan regresi, evaluasi performa, hingga penyimpanan model siap pakai.

## Ringkasan Proyek

- Dataset: `india_job_market_2024_2026.csv`
- Jumlah data: 5.000 baris dan 17 kolom
- Target prediksi: `Salary_LPA`
- Model: Random Forest Regressor dalam pipeline `scikit-learn`
- Split data: 80% data latih dan 20% data uji
- Output utama: model prediksi gaji `models/salary_predictor_model.pkl`

## Tujuan Analisis

1. Melihat distribusi gaji pekerjaan di India.
2. Mengidentifikasi kota dengan jumlah lowongan terbanyak.
3. Membandingkan rentang gaji berdasarkan tingkat pengalaman.
4. Membangun model prediksi gaji berdasarkan informasi pekerjaan dan kandidat.
5. Menyimpan pipeline model agar dapat digunakan kembali untuk prediksi data baru.

## Dataset

Dataset memiliki beberapa fitur utama:

- `Job_Title`
- `Company_Type`
- `Industry`
- `City`
- `Location_Tier`
- `Experience_Level`
- `Job_Type`
- `Work_Mode`
- `Education_Required`
- `Salary_LPA`

Ringkasan gaji:

- Minimum: 0.80 LPA
- Median: 13.60 LPA
- Rata-rata: 19.83 LPA
- Maksimum: 115.40 LPA

Kategori dominan pada dataset:

- Kota terbanyak: Remote, Mumbai, Pune, Hyderabad, Bangalore
- Level pengalaman terbanyak: Mid, Junior, Fresher
- Posisi terbanyak: Software Engineer, Backend Developer, Full Stack Developer, Data Analyst, Java Developer
- Industri terbanyak: Information Technology, FinTech, E-Commerce, Banking & Finance, EdTech

## Hasil Visualisasi

Visualisasi yang dihasilkan tersimpan di folder `reports/figures/`:

- `salary_distribution.png`: distribusi gaji dalam LPA
- `job_distribution_by_city.png`: jumlah lowongan berdasarkan kota
- `salary_by_experience.png`: rentang gaji berdasarkan tingkat pengalaman

## Pemodelan Machine Learning

Model dibangun menggunakan pipeline berikut:

- `OneHotEncoder` untuk fitur kategorikal
- `ColumnTransformer` sebagai preprocessing
- `RandomForestRegressor` sebagai model regresi

Fitur yang digunakan untuk prediksi:

- `Job_Title`
- `Company_Type`
- `Industry`
- `City`
- `Location_Tier`
- `Experience_Level`
- `Job_Type`
- `Work_Mode`
- `Education_Required`

## Hasil Evaluasi Model

Evaluasi dilakukan pada 1.000 data uji.

| Metrik | Nilai |
| --- | ---: |
| MAE | 2.78 LPA |
| RMSE | 4.59 LPA |
| R2 Score | 93.42% |

Berdasarkan nilai R2, model mampu menjelaskan sekitar 93.42% variasi gaji pada data uji.

## Contoh Prediksi

Contoh input simulasi:

- Posisi: UI/UX Designer
- Company Type: Indian Unicorn
- Industri: Information Technology
- Kota: Mumbai
- Experience Level: Junior (1-3 yrs)
- Job Type: Full-Time
- Work Mode: On-Site
- Education Required: BCA

Hasil prediksi:

```text
Estimasi Gaji: 14.90 Lakhs Per Annum (LPA)
```

## Struktur Proyek

```text
.
|-- data/
|   `-- india_job_market_2024_2026.csv
|-- models/
|   `-- salary_predictor_model.pkl
|-- notebooks/
|   `-- job_market_analysis_prediction.ipynb
|-- reports/
|   `-- figures/
|       |-- job_distribution_by_city.png
|       |-- salary_by_experience.png
|       `-- salary_distribution.png
|-- README.md
`-- requirements.txt
```

## Cara Menjalankan

1. Buat virtual environment jika diperlukan.
2. Install dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan notebook:

```text
notebooks/job_market_analysis_prediction.ipynb
```

4. Model hasil training akan tersimpan di:

```text
models/salary_predictor_model.pkl
```

## Catatan

Notebook saat ini menggunakan path lokal absolut pada beberapa cell untuk membaca dataset dan memuat model. Jika proyek dijalankan di perangkat lain, sebaiknya ganti path tersebut menjadi path relatif seperti:

```python
pd.read_csv("../data/india_job_market_2024_2026.csv")
joblib.load("../models/salary_predictor_model.pkl")
```
