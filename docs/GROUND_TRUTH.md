# GROUND TRUTH: Library Game AI Game Generator

> Dokumen ini di-generate otomatis oleh `core/ground_truth.py`. Jangan edit manual;
> jalankan `python scripts/update_ground_truth.py` atau tombol **Update Ground Truth** di app.

- Diperbarui: **2026-11-26T12:32:48+00:00**
- Minggu data terakhir: **2026-11-16**
- Model prediksi: **Gradient Boosting (HistGB)**
- Status Top-10: **STABIL** (overlap 10/10 dengan versi sebelumnya, Kendall tau = 0.467)
- Integritas: **LULUS**

## 1. Library aktif (Top-10)

Skor = (1 - w) x [0,65 x tren real + 0,35 x daya tarik BGG] + w x request in-app, w = 0.00 (naik otomatis seiring data request in-app bertambah).

| # | Game | Mekanik | Skor | Wikipedia views/minggu | Momentum | Google Trends ID | Daya tarik BGG | Request in-app 28 hari |
|---|---|---|---|---|---|---|---|---|
| 1 | Survival Kuis (`survival_kuis`) | quiz | 74.0 | 14,998 | -0.029 | 1 | 0.34 | 0 |
| 2 | Caption Meme (`voting_meme`) | open_vote | 73.9 | 12,484 | -0.049 | 88 | 0.39 | 0 |
| 3 | Harga Pas (`harga_pas`) | estimate | 70.0 | 5,414 | -0.186 | 42 | 0.41 | 0 |
| 4 | Cerita Berantai (`cerita_berantai`) | open_vote | 67.6 | 1,618 | -0.132 | 34 | 0.39 | 0 |
| 5 | Tebak Kata Kunci (`tebak_kata_kunci`) | quiz | 58.6 | 91,291 | -0.045 | - | 0.31 | 0 |
| 6 | Adu Receh (`adu_receh`) | open_vote | 57.0 | 1,596 | -0.075 | 31 | 0.39 | 0 |
| 7 | Fakta atau Hoaks (`fakta_hoaks`) | truefalse | 55.4 | 617 | -0.096 | 68 | 0.35 | 0 |
| 8 | Tebak Angka (`tebak_angka`) | estimate | 54.6 | 1,190 | -0.388 | 41 | 0.41 | 0 |
| 9 | Tebak Emoji (`emoji_tebak`) | quiz | 50.2 | 7,585 | -0.108 | 26 | 0.31 | 0 |
| 10 | Debat Kilat (`debat_kilat`) | open_vote | 45.9 | 921 | -0.142 | 59 | 0.39 | 0 |

## 2. Slot multiverse (eksperimen)

Aturan: game baru masuk slot ini (skor awal dari model daya tarik BGG). Dipromosikan ke Top-10 jika skornya >= game peringkat 10 dan sudah >= 5 request; dihapus jika setelah 7 hari request < 5.

- (kosong)

## 3. Validasi

**a. Perubahan vs snapshot sebelumnya:** masuk = -, keluar = -.

**b. Backtest model:** Top-10 momentum prediksi untuk minggu 2026-11-16 cocok 90% dengan aktual, arah naik/turun benar 67% (terlewat: ['voting_meme']; salah masuk: ['siapa_paling']).

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
| Baseline (naive: = minggu ini) | 0.1044 | 0.1044 | 0.0 | 0.84 |
| Baseline (rata-rata 4 minggu) | 0.1211 | 0.1219 | 0.5667 | 0.795 |
| Ridge Regression | 0.1052 | 0.1062 | 0.5286 | 0.845 |
| Huber Regression (robust) | 0.1014 | 0.1008 | 0.5571 | 0.855 |
| Random Forest | 0.1024 | 0.103 | 0.5262 | 0.845 |
| Gradient Boosting (HistGB) | 0.1 | 0.099 | 0.5619 | 0.85 |
| MLP (Neural Net) | 0.1114 | 0.1133 | 0.5262 | 0.84 |

## Sumber data

- Wikipedia Pageviews API (Wikimedia), artikel per jenis game: `core/real_sources.py`
- Google Trends Indonesia (pytrends), 1 keyword per game
- BoardGameGeek via TidyTuesday 2022-01-25 (4.911 game party) untuk model daya tarik
- Log request in-app (mulai dari nol, disimpan di branch `data`)
- TikTok/X tidak dipakai: tidak ada API gratis yang stabil

## 6. Riwayat update

| Waktu | Verdict | Overlap | Backtest P@10 | Multiverse |
|---|---|---|---|---|
| 2026-11-26T12:32:48+00:00 | STABIL | 10 | 0.9 | - |
| 2026-11-26T12:20:58+00:00 | BASELINE | 0 | 0.9 | - |
