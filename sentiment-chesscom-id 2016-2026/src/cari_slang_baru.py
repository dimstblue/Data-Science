import pandas as pd
import re

print("=========================================")
print("Eksperimen: Ekstraksi Kandidat Kata Slang")
print("=========================================")

# 1. Memuat dataset ulasan Chess.com
print("Membaca ulasan Chess.com...")
df = pd.read_csv("data/raw/ulasan_chess_10_tahun_terakhir.csv")

# 2. Fungsi sederhana untuk memecah kalimat menjadi kumpulan kata tunggal
def ambil_kata(teks):
    if not isinstance(teks, str):
        return []
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', '', teks) # Hanya sisakan huruf dan spasi
    return teks.split()

# 3. Mengumpulkan semua kata unik dari 1.000 ulasan pertama (untuk uji coba cepat)
print("Mengumpulkan kata-kata dari ulasan netizen...")
kata_ulasan = set()
for komentar in df['komentar'].head(1000):
    kata_ulasan.update(ambil_kata(komentar))

# 4. Memuat kamus kata baku (list_1.0.0.txt) yang kamu unggah
print("Memuat daftar kata baku dari luar...")
with open("data/list_1.0.0.txt", "r", encoding="utf-8") as f:
    # Simpan ke format 'set' agar pencarian kata secepat kilat
    kata_baku = set(line.strip().lower() for line in f)

# 5. FILTER UTAMA: Mencari kata yang TIDAK ADA di daftar kata baku
print("Menyaring kata-kata non-baku...")
kandidat_slang = kata_ulasan - kata_baku

# 6. Menampilkan hasil temuan
print(f"\n[HASIL] Total kata unik dari ulasan: {len(kata_ulasan)}")
print(f"[HASIL] Jumlah kata gaul/typo yang terdeteksi: {len(kandidat_slang)}")
print("\nBerikut 30 contoh kata non-baku yang berhasil ditangkap saringan:")
print(list(kandidat_slang)[:30])

# 7. MENYIMPAN HASIL KE CSV UNTUK DIEDIT (Tambahkan di baris paling bawah skripmu)
print("\nEkspor kandidat slang ke file CSV...")

# Mengubah set menjadi DataFrame dengan kolom 'slang'
df_slang = pd.DataFrame(list(kandidat_slang), columns=['slang'])

# Menambahkan kolom 'formal' yang masih kosong agar bisa kamu isi manual nanti
df_slang['formal'] = ""

# Mengurutkan kata dari A-Z agar mudah dibaca
df_slang = df_slang.sort_values(by='slang')

# Simpan ke folder data/
df_slang.to_csv("data/kamus_slang_draft.csv", index=False, encoding='utf-8')
print("[SUKSES] File draft kamus disimpan di: 'data/kamus_slang_draft.csv'")