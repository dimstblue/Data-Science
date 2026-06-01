import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("=========================================")
print("Memulai Tahap 3: Topic Modeling (LDA)")
print("=========================================")

# 1. Memuat dataset hasil preprocessing
print("Membaca data bersih...")
df_lda = pd.read_csv("D:\\MY PORTO\\DATA SCIENTIEST\\ANALISIS SENTIMEN\\sentiment-chesscom-id\\data\\processed\\ulasan_chess_cleaned.csv")

print("\nMenampilkan 5 baris pertama dataset:")
print(df_lda.head())

# 2. Antisipasi jika ada nilai kosong (NaN) setelah preprocessing
df_lda['komentar_bersih'] = df_lda['komentar_bersih'].fillna("")


print("Mengubah teks menjadi matriks angka (CountVectorizer)...")

# 1. Inisialisasi CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)

# 2. Transformasikan data komentar_bersih menjadi matriks frekuensi kata
tf_matrix = tf_vectorizer.fit_transform(df_lda['komentar_bersih'])

print(f"[INFO] Matriks teks berhasil dibuat dengan dimensi: {tf_matrix.shape}")

# Menentukan jumlah topik yang ingin diekstrak
jumlah_topik = 3

print(f"Melatih model LDA untuk mengekstrak {jumlah_topik} topik...")

# 1. Inisialisasi model LDA
lda_model = LatentDirichletAllocation(n_components=jumlah_topik, random_state=42)

# 2. Latih model menggunakan matriks teks kita
lda_model.fit(tf_matrix)

print("[SUKSES] Model LDA berhasil dilatih!")

print("\n--- Hasil Ekstraksi Topik Kunci ---")

# 1. Mengambil semua daftar kata dari vectorizer
nama_kata = tf_vectorizer.get_feature_names_out()

# 2. Melakukan looping untuk menampilkan kata kunci teratas di setiap topik
for indeks_topik, topik in enumerate(lda_model.components_):
    print(f"\n📢 TOPIK #{indeks_topik + 1}:")
    
    # Mengambil indeks 15 kata dengan bobot tertinggi di topik tersebut
    kata_teratas_indeks = topik.argsort()[:-15 - 1:-1]
    
    # Menggabungkan 15 kata tersebut menjadi satu baris teks
    kata_teratas = [nama_kata[i] for i in kata_teratas_indeks]
    print(" -> ", ", ".join(kata_teratas))

print("\n=========================================")
print("Tahap 3 Selesai! Silakan analisis topiknya.")
print("=========================================")