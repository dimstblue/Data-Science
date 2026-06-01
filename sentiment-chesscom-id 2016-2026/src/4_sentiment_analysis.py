import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("=================================================================")
print("Memulai Tahap 4: Komparasi & Ekspor Mandiri 3 Algoritma (3 Kelas)")
print("=================================================================")

# 1. Membuat folder 'models' jika belum ada
os.makedirs("models", exist_ok=True)

# 2. Memuat data steril hasil Tahap 1 menggunakan relative path
print("Membaca data bersih...")
df = pd.read_csv("D:\\MY PORTO\\DATA SCIENTIEST\\ANALISIS SENTIMEN\\sentiment-chesscom-id\\data\\processed\\ulasan_chess_cleaned.csv")
df['komentar_bersih'] = df['komentar_bersih'].fillna("")

print("\n=================================================================")
print("Melakukan pelabelan sentimen multi-class (3 Kelas)...")
print("=================================================================")

df_model = df.copy()

# Fungsi pembagi untuk 3 kelas sentimen
def petakan_sentimen(bintang):
    if bintang >= 4:
        return 2  # Positif
    elif bintang == 3:
        return 1  # Netral
    else:
        return 0  # Negatif

df_model['sentimen'] = df_model['rate'].apply(petakan_sentimen)

print("\nDistribusi 3 kelas sentimen asli:")
print(df_model['sentimen'].value_counts().sort_index())

print("\n=================================================================")
print("Ekstraksi fitur teks menggunakan TF-IDF & Splitting Data (80:20)")
print("=================================================================")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_model['komentar_bersih'])
y = df_model['sentimen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Konfigurasi 3 algoritma dengan penanganan Imbalance Data
daftar_model = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine (SVM)": LinearSVC(class_weight='balanced', random_state=42)
}

# Variabel untuk melacak performa terbaik
skor_terbaik = 0
model_terbaik_nama = ""
model_terbaik_objek = None

print("\n=================================================================")
print("--- Memulai Proses Pelatihan & Penyimpanan Mandiri ---")
print("=================================================================")

for nama, model in daftar_model.items():
    print(f"\n⏳ Melatih model {nama}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Hitung Akurasi Skor
    akurasi = accuracy_score(y_test, y_pred)
    print(f"✅ Model {nama} selesai. Akurasi: {akurasi:.4f}")
    
    # Menentukan nama file berdasarkan algoritmanya
    if "Random Forest" in nama:
        prefix = "rf"
    elif "Naive Bayes" in nama:
        prefix = "nb"
    else:
        prefix = "svm"
        
    # EKSPOR MANDIRI: Menyimpan setiap model dan pasangannya langsung di dalam loop
    opsi_model_path = f"models/{prefix}_model.pkl"
    opsi_tfidf_path = f"models/{prefix}_tfidf.pkl"
    
    with open(opsi_model_path, "wb") as f_m:
        pickle.dump(model, f_m)
    with open(opsi_tfidf_path, "wb") as f_t:
        pickle.dump(tfidf, f_t)
    print(f"   [TERTIMPAN] File mandiri disimpan: {opsi_model_path} & {opsi_tfidf_path}")
    
    # Validasi pelacakan model pemenang utama
    if akurasi > skor_terbaik:
        skor_terbaik = akurasi
        model_terbaik_nama = nama
        model_terbaik_objek = model

print("\n=================================================================")
print(f"🏆 RINGKASAN EVALUASI DAN PEMENANG UTAMA")
print("=================================================================")
print(f"Model Terbaik Global : {model_terbaik_nama}")
print(f"Skor Akurasi Tertinggi: {skor_terbaik:.4f}")
print("-----------------------------------------------------------------")

# Tampilkan rincian performa mendalam milik model pemenang utama
y_pred_terbaik = model_terbaik_objek.predict(X_test)
print("\nDetail Laporan Klasifikasi Model Pemenang:")
print(classification_report(y_test, y_pred_terbaik, target_names=['Negatif', 'Netral', 'Positif'], zero_division=0))

# EKSPOR PEMENANG GLOBAL: Menyimpan satu salinan ekstra untuk model terbaik keseluruhan
jalur_best_model = "models/best_model.pkl"
jalur_best_tfidf = "models/tfidf_vectorizer.pkl"

with open(jalur_best_model, "wb") as f_best_m:
    pickle.dump(model_terbaik_objek, f_best_m)
with open(jalur_best_tfidf, "wb") as f_best_t:
    pickle.dump(tfidf, f_best_t)

print(f"\n[SUKSES] Salinan Juara Umum disimpan di   : '{jalur_best_model}' & '{jalur_best_tfidf}'")
print("=================================================================")
print("Seluruh Model Berhasil Diekstrak! Pipeline Selesai dengan Sempurna.")
print("=================================================================")