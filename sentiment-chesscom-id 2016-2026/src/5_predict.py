import os
import re
import pickle
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("=================================================================")
print("Memulai Tahap 5: Aplikasi Prediksi Sentimen Otomatis (Interactive)")
print("=================================================================")

# -----------------------------------------------------------------
# 1. MEMUAT OBJEK MODEL & PIPELINE PENDUKUNG
# -----------------------------------------------------------------
print("Memuat model terbaik dan TF-IDF Vectorizer...")
try:
    with open("models/best_model.pkl", "rb") as f_model:
        model = pickle.load(f_model)
    with open("models/tfidf_vectorizer.pkl", "rb") as f_tfidf:
        tfidf = pickle.load(f_tfidf)
    print("[SUKSES] Model dan TF-IDF berhasil dimuat ke dalam sistem.")
except FileNotFoundError:
    print("[ERROR] File model tidak ditemukan! Pastikan sudah menjalankan 'src/4_sentiment_analysis.py' terlebih dahulu.")
    exit()

print("Memuat kamus slang dan menyusun daftar stopwords...")
# Memuat kamus slang untuk normalisasi teks input baru
df_kamus = pd.read_csv("data/kamus_slang.csv").dropna(subset=['formal'])
kamus_slang = dict(zip(df_kamus['slang'], df_kamus['formal']))

# Memuat stopword Sastrawi + Kustom
factory = StopWordRemoverFactory()
stopwords_standar = factory.get_stop_words()
stopwords_kustom = [
    'game', 'aplikasi', 'apk', 'catur', 'main', 'mainnya', 'bermain', 'chess', 'com',
    'saya', 'kamu', 'dia', 'mereka', 'kita', 'ini', 'itu', 'sini', 'situ', 'bgt', 'banget',
    'ya', 'gila', 'kok', 'sih', 'lah', 'kah', 'deh', 'oh', 'eh', 'ah', 'tapi', 'kalau', 'kalo',
    'nya', 'dan', 'yang', 'di', 'dari', 'untuk', 'ada', 'bisa', 'sudah', 'juga', 'mau', 'buat', 'sama'
]
semua_stopwords = set(stopwords_standar + stopwords_kustom)

# -----------------------------------------------------------------
# 2. EMBEDDING PIPELINE PREPROCESSING (Wajib sama dengan Tahap 1)
# -----------------------------------------------------------------
def bersihkan_teks_input(teks):
    teks = teks.lower().replace('\n', ' ')
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    
    # Normalisasi Slang
    kata_kata = teks.split()
    kata_normal = [kamus_slang.get(kata, kata) for kata in kata_kata]
    
    # Stopword Removal
    kata_bersih = [kata for kata in kata_normal if kata not in semua_stopwords]
    
    return " ".join(kata_bersih)

# Mapping angka prediksi ke label teks untuk manusia
label_sentimen = {
    0: "NEGATIF 🔴",
    1: "NETRAL 🟡",
    2: "POSITIF 🟢"
}

# -----------------------------------------------------------------
# 3. INTERACTIVE LOOP (Aplikasi Utama)
# -----------------------------------------------------------------
print("\n=================================================================")
print("PROGRAM PREDIKSI SENTIMEN ULASAN CHESS.COM READY!")
print("Ketik komentar kamu di bawah ini untuk menguji kecerdasan model.")
print("Ketik 'keluar' atau 'exit' untuk menyudahi program.")
print("=================================================================\n")

while True:
    input_pengguna = input("Masukkan teks ulasan: ")
    
    # Cek kondisi keluar program
    if input_pengguna.lower() in ['keluar', 'exit']:
        print("\nTerima kasih, pengguna! Program prediksi ditutup.")
        break
        
    if input_pengguna.strip() == "":
        print("Teks tidak boleh kosong. Silakan ketik kembali.\n")
        continue
        
    # Step A: Bersihkan teks baru menggunakan pipeline formal
    teks_steril = bersihkan_teks_input(input_pengguna)
    
    # Step B: Transformasikan teks menjadi matriks angka TF-IDF
    fitur_vektor = tfidf.transform([teks_steril])
    
    # Step C: Prediksi menggunakan model Machine Learning terbaik
    kode_prediksi = model.predict(fitur_vektor)[0]
    hasil_teks = label_sentimen[kode_prediksi]
    
    # Tampilkan Hasil ke Terminal
    print(f"-> Teks Setelah Preprocessing : '{teks_steril}'")
    print(f"-> Hasil Prediksi Sentimen    : {hasil_teks}")
    print("-" * 65 + "\n")