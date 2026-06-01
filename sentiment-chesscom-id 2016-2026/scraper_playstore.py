from datetime import datetime
import pandas as pd
from google_play_scraper import Sort, reviews

print("Memulai scraping ulasan Chess.com untuk 10 tahun terakhir...")

# Batas waktu: 10 tahun ke belakang dari tahun 2026 (1 Januari 2016)
batas_waktu = datetime(2016, 1, 1)
target_country = 'id'

semua_ulasan = []
continuation_token = None
proses_berjalan = True

while proses_berjalan:
    # Tarik data per batch (misal per 200 ulasan)
    result, continuation_token = reviews(
        'com.chess',
        lang='id',
        country=target_country,
        sort=Sort.NEWEST,
        count=200,
        continuation_token=continuation_token
    )
    
    if not result:
        break  # Berhenti jika sudah tidak ada ulasan lagi dari Google Play
        
    for review in result:
        # Cek tanggal ulasan (kolom 'at' bawaan library mengembalikan objek datetime)
        waktu_ulasan = review['at']
        
        # Jika ulasan sudah lebih lama dari batas waktu (sebelum 2016), hentikan semua proses
        if waktu_ulasan < batas_waktu:
            proses_berjalan = False
            break
            
        # Jika masih masuk rentang 10 tahun terakhir, simpan data spesifik yang kamu butuhkan
        semua_ulasan.append({
            'id': review['reviewId'],
            'username': review['userName'],
            'komentar': review['content'],
            'country': target_country.upper(),
            'rate': review['score'],
            'waktu': waktu_ulasan.strftime('%Y-%m-%d %H:%M:%S') # Menyimpan format waktu asli komentar
        })
        
    print(f"Berhasil menarik {len(semua_ulasan)} ulasan...", end='\r')
    
    # Jika token habis, otomatis berhenti
    if not continuation_token:
        break

# Ubah ke DataFrame
df_final = pd.DataFrame(semua_ulasan)

# Simpan ke CSV
nama_file = 'ulasan_chess_10_tahun_terakhir.csv'
df_final.to_csv(nama_file, index=False, encoding='utf-8')

print(f"\n\nSelesai! Berhasil mengumpulkan {len(df_final)} ulasan dari 10 tahun terakhir.")
print(f"Data disimpan dalam file: '{nama_file}'")