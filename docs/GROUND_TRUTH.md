# GROUND TRUTH: Library Game AI Game Generator

> Dokumen ini di-generate otomatis oleh `core/ground_truth.py`. Jangan edit manual;
> jalankan `python scripts/update_ground_truth.py` atau tombol **Update Ground Truth** di app.

- Diperbarui: **2026-11-26T12:20:58+00:00**
- Minggu data terakhir: **2026-11-16**
- Model prediksi: **Gradient Boosting (HistGB)**
- Status Top-10: **BASELINE** (overlap 0/10 dengan versi sebelumnya, Kendall tau = None)
- Integritas: **LULUS**

## 1. Library aktif (Top-10)

Skor = (1 - w) x [0,65 x tren real + 0,35 x daya tarik BGG] + w x request in-app, w = 0.00 (naik otomatis seiring data request in-app bertambah).

| # | Game | Mekanik | Skor | Wikipedia views/minggu | Momentum | Google Trends ID | Daya tarik BGG | Request in-app 28 hari |
|---|---|---|---|---|---|---|---|---|
| 1 | Tebak Angka (`tebak_angka`) | estimate | 78.6 | 3 | +0.002 | 48 | 0.43 | 0 |
| 2 | Caption Meme (`voting_meme`) | open_vote | 74.8 | 12,484 | -0.052 | 89 | 0.42 | 0 |
| 3 | Survival Kuis (`survival_kuis`) | quiz | 70.7 | 14,998 | -0.029 | 1 | 0.33 | 0 |
| 4 | Harga Pas (`harga_pas`) | estimate | 67.1 | 5,414 | -0.199 | 40 | 0.43 | 0 |
| 5 | Cerita Berantai (`cerita_berantai`) | open_vote | 67.0 | 1,618 | -0.131 | 34 | 0.42 | 0 |
| 6 | Adu Receh (`adu_receh`) | open_vote | 57.9 | 1,596 | -0.075 | 31 | 0.42 | 0 |
| 7 | Fakta atau Hoaks (`fakta_hoaks`) | truefalse | 53.2 | 617 | -0.094 | 68 | 0.37 | 0 |
| 8 | Tebak Emoji (`emoji_tebak`) | quiz | 53.1 | 7,585 | -0.112 | 26 | 0.32 | 0 |
| 9 | Debat Kilat (`debat_kilat`) | open_vote | 43.3 | 921 | -0.146 | 59 | 0.42 | 0 |
| 10 | Tebak Kata Kunci (`tebak_kata_kunci`) | quiz | 45.9 | 91,291 | -0.047 | 14 | 0.32 | 0 |

## 2. Slot multiverse (eksperimen)

Aturan: game baru masuk slot ini (skor awal dari model daya tarik BGG). Dipromosikan ke Top-10 jika skornya >= game peringkat 10 dan sudah >= 5 request; dihapus jika setelah 7 hari request < 5.

- (kosong)

## 3. Validasi

**a. Perubahan vs snapshot sebelumnya:** masuk = ['tebak_angka', 'voting_meme', 'survival_kuis', 'harga_pas', 'cerita_berantai', 'adu_receh', 'fakta_hoaks', 'emoji_tebak', 'debat_kilat', 'tebak_kata_kunci'], keluar = -.

**b. Backtest model:** Top-10 momentum prediksi untuk minggu 2026-11-16 cocok 90% dengan aktual, arah naik/turun benar 86% (terlewat: ['voting_meme']; salah masuk: ['siapa_paling']).

**c. Cek integritas:**

- [x] Library berisi tepat 10 game
- [x] Tidak ada game ganda
- [x] Semua game Top-10 terdaftar di katalog/custom
- [x] Slot multiverse maksimal 1
- [x] Game custom aktif lolos aturan kunci

## 4. Aturan kunci game custom

- Jumlah aturan main: 4-7
- Peran wajib: host, peserta
- Pemain: 2-30
- Ronde: 3-15, 10-120 detik/ronde
- Total sesi maks 60 menit
- Mekanik harus salah satu: quiz, truefalse, majority, estimate, open_vote

## 5. Performa model tren (MAE_log (test 20 minggu terakhir))

| Model | MAE log | MAPE views | Akurasi arah | Precision@10 momentum |
|---|---|---|---|---|
| Baseline (naive: = minggu ini) | 0.113 | 0.1239 | 0.0024 | 0.85 |
| Baseline (rata-rata 4 minggu) | 0.1262 | 0.1348 | 0.5693 | 0.82 |
| Ridge Regression | 0.115 | 0.1253 | 0.5207 | 0.835 |
| Huber Regression (robust) | 0.1105 | 0.119 | 0.5766 | 0.845 |
| Random Forest | 0.1104 | 0.1193 | 0.5401 | 0.84 |
| Gradient Boosting (HistGB) | 0.1075 | 0.1159 | 0.5547 | 0.845 |
| MLP (Neural Net) | 0.1254 | 0.1335 | 0.4915 | 0.825 |

## Sumber data

- Wikipedia Pageviews API (Wikimedia), artikel per jenis game: `core/real_sources.py`
- Google Trends Indonesia (pytrends), 1 keyword per game
- BoardGameGeek via TidyTuesday 2022-01-25 (4.911 game party) untuk model daya tarik
- Log request in-app (mulai dari nol, disimpan di branch `data`)
- TikTok/X tidak dipakai: tidak ada API gratis yang stabil

## 6. Riwayat update

| Waktu | Verdict | Overlap | Backtest P@10 | Multiverse |
|---|---|---|---|---|
| 2026-11-26T12:20:58+00:00 | BASELINE | 0 | 0.9 | - |
