import os
import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("============== ===============")
print("LOAD DATA ")
print("============== ===============")

# Memuat dataset menggunakan relative path agar lebih fleksibel
print("membaca dataset dari folder raw...")
df = pd.read_csv("data/raw/ulasan_chess_10_tahun_terakhir.csv")

# menampilkan 5 baris pertama dataset
print("\n menampilkan 5 baris pertama dataset:")
print(df.head())

print("\n============== ===============")
print("PREPROCESSING & NLP PIPELINE")
print("============== ===============")

# --- FUNGSI 1: CLEANING & CASE FOLDING ---
def bersihkan_teks(teks):
    # Antisipasi jika ada baris yang kosong atau bernilai NaN
    if not isinstance(teks, str):
        return ""
    
    # 1. Case Folding: Menyeragamkan semua huruf menjadi huruf kecil
    teks = teks.lower()
    
    # 2. Menghapus karakter enter (\n) yang merusak baris teks
    teks = teks.replace('\n', ' ')
    
    # 3. Regex: Hanya menyisakan huruf a-z dan spasi, karakter lain dihapus
    teks = re.sub(r'[^a-z\s]', '', teks)
    
    # 4. Mengurangi spasi ganda yang tidak sengaja terbentuk menjadi spasi tunggal
    teks = re.sub(r'\s+', ' ', teks).strip()
    
    return teks


# --- LOAD KAMUS SLANG ---
print("\nMemuat kamus slang eksternal...")
df_kamus = pd.read_csv("data/kamus_slang.csv")
df_kamus = df_kamus.dropna(subset=['formal'])
kamus_slang = dict(zip(df_kamus['slang'], df_kamus['formal']))
print(f"[INFO] Berhasil memuat {len(kamus_slang)} kata gaul yang siap dinormalisasi.")


# --- FUNGSI 2: SLANG NORMALIZATION ---
def normalisasi_slang(teks):
    # Memecah kalimat menjadi daftar kata tunggal berdasarkan spasi
    kata_kata = teks.split()
    
    # Mengganti kata jika terdaftar di kamus, jika tidak ada tetap gunakan kata asli
    kata_normal = [kamus_slang.get(kata, kata) for kata in kata_kata]
    
    # Menyatukan kembali daftar kata menjadi satu string kalimat utuh
    return " ".join(kata_normal)


# --- SETUP STOPWORDS (SASTRAWI + KUSTOM CHESS) ---
print("\nInisialisasi Stopwords (Sastrawi + Kustom)...")
factory = StopWordRemoverFactory()
stopwords_standar = factory.get_stop_words()

# Daftar kata tugas/derau kustom yang sering muncul di review catur tapi tidak bermakna sentimen
stopwords_kustom = [
    'game', 'aplikasi', 'apk', 'catur', 'main', 'mainnya', 'bermain', 'chess', 'com',
    'saya', 'kamu', 'dia', 'mereka', 'kita', 'ini', 'itu', 'sini', 'situ', 'bgt', 'banget',
    'ya', 'gila', 'kok', 'sih', 'lah', 'kah', 'deh', 'oh', 'eh', 'ah', 'tapi', 'kalau', 'kalo',
    'nya', 'dan', 'yang', 'di', 'dari', 'untuk', 'ada', 'bisa', 'sudah', 'juga', 'mau', 'buat', 'sama'
]

# Satukan semua stopword ke dalam tipe data 'set' untuk pencarian super cepat
semua_stopwords = set(stopwords_standar + stopwords_kustom)
print(f"[INFO] Berhasil menyusun {len(semua_stopwords)} daftar kata buang (Stopwords).")


# --- FUNGSI 3: STOPWORD REMOVAL ---
def hapus_stopword(teks):
    kata_kata = teks.split()
    # Hanya ambil kata yang TIDAK terdaftar di dalam semua_stopwords
    kata_bersih = [kata for kata in kata_kata if kata not in semua_stopwords]
    return " ".join(kata_bersih)


print("============== ===============")
print("\nMenjalankan proses pipeline pembersihan teks (23.000+ data)...")
print("============== ===============")

# Langkah 1: Bersihkan simbol, angka, emoji, dan paksa huruf kecil
df['komentar_bersih'] = df['komentar'].apply(bersihkan_teks)

# Langkah 2: Ubah kata gaul menjadi formal (misal: "yg" -> "yang")
df['komentar_bersih'] = df['komentar_bersih'].apply(normalisasi_slang)

# Langkah 3: Hapus kata tugas formal ("yang") beserta kata derau kustom ("game", "catur")
df['komentar_bersih'] = df['komentar_bersih'].apply(hapus_stopword)

# Menampilkan sampel perbandingan data asli vs data bersih hasil pipeline baru
print("\n[INFO] Perbandingan hasil sebelum dan sesudah advanced preprocessing:")
print(df[['komentar', 'komentar_bersih']].head())

# Memastikan folder data/processed sudah terbentuk
os.makedirs("data/processed", exist_ok=True)

# Menyimpan output dataset bersih ke folder processed
nama_file_output = "data/processed/ulasan_chess_cleaned.csv"
df.to_csv(nama_file_output, index=False, encoding='utf-8')

print(f"\n[SUKSES] Tahap 1 selesai! Data bersih steril disimpan di: '{nama_file_output}'")