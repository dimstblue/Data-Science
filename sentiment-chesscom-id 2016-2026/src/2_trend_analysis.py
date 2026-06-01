import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("============== ===============")
print("TREND ANALYSIS")
print("============== ===============")

# memuat dataset yang sudah diproses dari folder processed
print("membaca dataset yang sudah diproses dari folder processed...")
df_processed = pd.read_csv(r"D:\MY PORTO\DATA SCIENTIEST\ANALISIS SENTIMEN\sentiment-chesscom-id\data\processed\ulasan_chess_cleaned.csv")

# menampilkan 5 baris pertama dataset yang sudah diproses
print("\n menampilkan 5 baris pertama dataset yang sudah diproses:")
print (df_processed.head())


print("============== ===============")
print("ekstraksi waktu")
print("============== ===============")

# 1. Mengubah tipe data kolom waktu menjadi datetime
df_processed['waktu'] = pd.to_datetime(df_processed['waktu'])

# 2. Ekstraksi Tahun dan Bulan-Tahun untuk basis pengelompokan
df_processed['tahun'] = df_processed['waktu'].dt.year
df_processed['bulan_tahun'] = df_processed['waktu'].dt.to_period('M')

print("\n[INFO] Struktur data waktu berhasil diekstrak.")
print(df_processed[['waktu', 'tahun', 'bulan_tahun', 'rate']].head())


print("============== ===============")
print("\nMembuat grafik tren volume ulasan bulanan...")
print("============== ===============")

# 1. Menghitung total ulasan per bulan
tren_bulanan = df_processed.groupby('bulan_tahun').size()

# 2. Mengatur ukuran dan tema grafik
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 3. Membuat grafik garis
tren_bulanan.plot(kind='line', marker='o', color='b', linewidth=2)

# 4. Mengatur label teks pada grafik
plt.title('Tren Volume Ulasan Chess.com Regional Indonesia (10 Tahun Terakhir)', fontsize=14, fontweight='bold')
plt.xlabel('Periode (Bulan-Tahun)', fontsize=12)
plt.ylabel('Jumlah Ulasan', fontsize=12)
plt.xticks(rotation=45) # Memiringkan teks tanggal agar tidak bertabrakan
plt.tight_layout()

# 5. Menyimpan grafik ke folder reports/figures/
jalur_gambar_1 = "reports/figures/1_tren_volume_bulanan.png"
plt.savefig(jalur_gambar_1, dpi=300)
plt.close() # Menutup grafik dari memori

print(f"[SUKSES] Grafik tren volume berhasil disimpan di: '{jalur_gambar_1}'")


print("============== ===============")
print("Membuat grafik rata-rata rating tahunan...")
print("============== ===============")

# 1. Menghitung rata-rata bintang (rate) per tahun
rata_rating_tahunan = df_processed.groupby('tahun')['rate'].mean().reset_index()

# 2. Mengatur ukuran grafik
plt.figure(figsize=(10, 5))

# 3. Membuat grafik batang menggunakan Seaborn
sns.barplot(data=rata_rating_tahunan, x='tahun', y='rate', hue='tahun', palette='Blues_d', legend=False)

# 4. Mengatur label teks pada grafik
plt.title('Rata-Rata Rating Chess.com per Tahun (Regional Indonesia)', fontsize=14, fontweight='bold')
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Rata-Rata Rating (Bintang 1-5)', fontsize=12)
plt.ylim(1, 5) # Mengunci batas bawah bintang 1 dan batas atas bintang 5
plt.tight_layout()

# 5. Menyimpan grafik ke folder reports/figures/
jalur_gambar_2 = "reports/figures/2_tren_rating_tahunan.png"
plt.savefig(jalur_gambar_2, dpi=300)
plt.close()

print(f"[SUKSES] Grafik tren rating berhasil disimpan di: '{jalur_gambar_2}'")
print("\n=========================================")
print("Tahap 2 Selesai! Semua grafik siap dianalisis.")
print("=========================================")
