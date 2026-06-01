import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("Memulai pengujian jumlah topik optimal...")

# 1. Load Data
df = pd.read_csv("D:\\MY PORTO\\DATA SCIENTIEST\\ANALISIS SENTIMEN\\sentiment-chesscom-id\\data\\processed\\ulasan_chess_cleaned.csv")
df['komentar_bersih'] = df['komentar_bersih'].fillna("")

# 2. Vektorisasi
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
tf_matrix = tf_vectorizer.fit_transform(df['komentar_bersih'])

# 3. Looping untuk menguji beberapa nilai K (Jumlah Topik)
kandidat_k = [2, 3, 4, 5, 6, 7]
nilai_perplexity = []

for k in kandidat_k:
    print(f"Menguji model LDA dengan K = {k} topik...")
    lda = LatentDirichletAllocation(n_components=k, random_state=42, n_jobs=-1)
    lda.fit(tf_matrix)
    
    # Hitung skor perplexity untuk nilai K ini
    nilai_perplexity.append(lda.perplexity(tf_matrix))

# 4. Visualisasikan Hasilnya ke dalam Grafik
plt.figure(figsize=(8, 4))
sns.set_theme(style="whitegrid")
plt.plot(kandidat_k, nilai_perplexity, marker='o', color='r', linewidth=2)
plt.title('Evaluasi Jumlah Topik LDA Menggunakan Perplexity Score', fontsize=12, fontweight='bold')
plt.xlabel('Jumlah Topik (K)', fontsize=10)
plt.ylabel('Perplexity Score (Makin Rendah Makin Baik)', fontsize=10)
plt.tight_layout()

# Simpan grafik ke folder figures
jalur_grafik = "D:\\MY PORTO\\DATA SCIENTIEST\\ANALISIS SENTIMEN\\sentiment-chesscom-id\\reports\\figures\\3_evaluasi_perplexity.png"
plt.savefig(jalur_grafik, dpi=300)
plt.close()

print(f"\n[SUKSES] Pengujian selesai! Grafik evaluasi disimpan di: '{jalur_grafik}'")