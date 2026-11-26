# GROUND TRUTH: Library Game AI Game Generator

> Dokumen ini di-generate otomatis oleh `core/ground_truth.py`. Jangan edit manual;
> jalankan `python scripts/update_ground_truth.py` atau tombol **Update Ground Truth** di app.

- Diperbarui: **2026-11-26T08:59:43+00:00**
- Minggu data terakhir: **2026-11-23**
- Model prediksi: **Ridge Regression**
- Status Top-10: **BASELINE** (overlap 0/10 dengan versi sebelumnya, Kendall tau = None)
- Integritas: **LULUS**

## 1. Library aktif (Top-10)

| # | Game | Mekanik | Prediksi request minggu depan | Request minggu ini | Google Trends | TikTok |
|---|---|---|---|---|---|---|
| 1 | Caption Meme (`voting_meme`) | open_vote | 110.7 | 114 | 100 | 717 |
| 2 | Survival Kuis (`survival_kuis`) | quiz | 49.9 | 57 | 100 | 377 |
| 3 | Kuis Kilat (`kuis_kilat`) | quiz | 42.6 | 50 | 35 | 90 |
| 4 | Satu Frekuensi (`satu_frekuensi`) | majority | 36.2 | 41 | 49 | 165 |
| 5 | Tebak Emoji (`emoji_tebak`) | quiz | 26.5 | 32 | 90 | 255 |
| 6 | Fakta atau Hoaks (`fakta_hoaks`) | truefalse | 26.0 | 30 | 37 | 99 |
| 7 | Adu Receh (`adu_receh`) | open_vote | 22.5 | 23 | 48 | 151 |
| 8 | Tebak Angka (`tebak_angka`) | estimate | 22.1 | 26 | 60 | 226 |
| 9 | Siapa Paling (`siapa_paling`) | majority | 21.9 | 24 | 60 | 185 |
| 10 | Tebak Kata Kunci (`tebak_kata_kunci`) | quiz | 19.7 | 21 | 44 | 78 |

## 2. Slot multiverse (eksperimen)

Aturan: game baru masuk slot ini. Dipromosikan ke Top-10 jika prediksinya >= game peringkat 10 dan sudah >= 5 request; dihapus jika setelah 7 hari request < 5.

- (kosong)

## 3. Validasi

**a. Perubahan vs snapshot sebelumnya:** masuk = ['voting_meme', 'survival_kuis', 'kuis_kilat', 'satu_frekuensi', 'emoji_tebak', 'fakta_hoaks', 'adu_receh', 'tebak_angka', 'siapa_paling', 'tebak_kata_kunci'], keluar = -.

**b. Backtest model:** Top-10 prediksi untuk minggu 2026-11-23 cocok 100% dengan Top-10 aktual (terlewat: -; salah masuk: -).

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

## 5. Performa model (test 26 minggu terakhir)

| Model | MAE | RMSE | R2 | Precision@10 |
|---|---|---|---|---|
| Baseline (naive: = minggu ini) | 6.913 | 9.242 | 0.8264 | 0.8615 |
| Ridge Regression | 6.105 | 8.074 | 0.8675 | 0.8962 |
| Poisson GLM | 11.275 | 30.592 | -0.9019 | 0.8846 |
| Random Forest | 6.406 | 9.074 | 0.8327 | 0.9038 |
| Gradient Boosting (HistGB) | 6.334 | 8.735 | 0.845 | 0.8846 |
| MLP (Neural Net) | 6.833 | 9.272 | 0.8253 | 0.8769 |

## 6. Riwayat update

| Waktu | Verdict | Overlap | Backtest P@10 | Multiverse |
|---|---|---|---|---|
| 2026-11-26T08:59:43+00:00 | BASELINE | 0 | 1.0 | - |
