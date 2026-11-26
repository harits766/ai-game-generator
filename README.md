# 🎲 AI Game Generator

Generator game party edukatif berbasis **tren real + dokumen + agent LangChain**. Ceritakan acaramu
(jumlah orang, durasi, jenis acara, online/offline, lokasi, topik) dan upload materi (PDF/DOCX/TXT/MD).
AI memilih game yang paling pas dari library yang lagi naik daun, membuat soalnya dari dokumen, lalu langsung
memberi **link HOST** dan **link PESERTA** (plus QR) untuk dimainkan di HP masing-masing.

> Capstone Project **Data Science & Generative AI** (Celerates, 26 Nov - 18 Des 2026).

## Masalah dan solusi

Acara kumpul sering berakhir dengan semua orang main HP sendiri-sendiri. Game party yang ada kaku:
pilihannya itu-itu saja (mis. cuma satu jenis game), harus gonta-ganti aplikasi, dan tidak menyesuaikan materi acara.
Aplikasi ini menjadikan HP alat main bareng, dengan game yang dibuat otomatis sesuai acara dan materi.

## Data (real, kecuali yang ditandai)

| Sumber | Jenis | Dipakai untuk |
|---|---|---|
| [Wikipedia Pageviews API](https://wikimedia.org/api/rest_v1/) | real, mingguan, 21 artikel | minat publik per jenis game (fitur + target model tren) |
| Google Trends Indonesia (pytrends) | real, mingguan, 15 keyword | momentum minat di Indonesia |
| [BoardGameGeek via TidyTuesday 2022-01-25](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25) | real, 4.911 game party | model daya tarik (cold start game baru) |
| Log request in-app | real, **mulai dari 0** | bobot ranking naik otomatis seiring data bertambah |
| TikTok / X | tidak dipakai | tidak ada API gratis yang stabil |

Pemetaan game -> artikel/keyword ada di `core/real_sources.py`. Data tren diperbarui otomatis **tiap Senin** oleh
GitHub Actions (`.github/workflows/refresh-real-data.yml`): tarik data, latih ulang model, regenerate ground truth, commit.

## Model ML (2 model, keduanya data real)

**1. Model tren** (`scripts/train_model.py`): memprediksi perubahan `log(1+views)` minggu depan per jenis game.
Split waktu, 20 minggu terakhir = test.

| Model | MAE log | Akurasi arah | Precision@10 momentum |
|---|---|---|---|
| Baseline (naive: = minggu ini) | 0.113 | 0.00 | 0.85 |
| Baseline (rata-rata 4 minggu) | 0.126 | 0.57 | 0.82 |
| Ridge Regression | 0.115 | 0.52 | 0.835 |
| Huber Regression (robust) | 0.111 | 0.58 | 0.845 |
| Random Forest | 0.110 | 0.54 | 0.84 |
| **Gradient Boosting (HistGB), dipakai** | **0.108** | 0.55 | 0.845 |
| MLP (Neural Net) | 0.125 | 0.49 | 0.825 |

Data web mingguan mirip random walk, jadi baseline naive kuat. Model terpilih mengalahkannya di MAE dan bisa menebak
arah naik/turun (baseline naive tidak bisa), yang dipakai untuk **momentum**.

**2. Model daya tarik / cold start** (`scripts/train_appeal_model.py`): klasifikasi game party populer (25% teratas
jumlah rating BGG) dari profil pemain, durasi, usia, kategori, mekanik. Stratified 5-fold CV.

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Baseline | 0.50 | 0.50 |
| Logistic Regression | 0.673 | 0.689 |
| **Random Forest, dipakai** | **0.699** | 0.720 |
| Gradient Boosting (HistGB) | 0.679 | 0.691 |
| MLP (Neural Net) | 0.663 | 0.681 |

**Skor ranking library:** `skor = (1 - w) x [0,65 x tren_real + 0,35 x daya_tarik] + w x request_in_app`,
dengan `w = n / (n + 50)` (maks 0,7) dan `n` = request in-app 28 hari terakhir. Saat app baru dipakai `w = 0`.

## GenAI: LangChain

| Komponen | Isi |
|---|---|
| **Agent Game Master** | `langchain.agents.create_agent` (ReAct di atas LangGraph), 7 tools: `get_trends`, `recommend_games`, `read_document`, `design_custom_game`, `validate_custom_game`, `register_custom_game`, `create_room` |
| **Chain Desainer** | ide bebas -> `GameSpec` (Pydantic, structured output) -> validasi aturan kunci |
| **Chain Konten** | potongan dokumen (retrieval TF-IDF, maks 4) -> `RoundsOut` (Pydantic) -> validasi skema -> cache |

**Pemakaian API key** (hanya di `core/llm.py`): Gemini (gratis) utama, OpenAI cadangan otomatis
(`ModelFallbackMiddleware`). Pengaman: maks 6 panggilan LLM per pesan (`ModelCallLimitMiddleware`), `create_room` dan
`register_custom_game` maks 1x per pesan, total tool call maks 8 (`ToolCallLimitMiddleware`), output maks 1.500 token,
temperature rendah, timeout 30 detik, dan semua angka/link wajib dari hasil tool. Model ML, skor, retrieval, validasi,
dan room **tidak** memakai API. Tanpa key, planner offline memanggil tools yang sama.

## Fitur aplikasi

| Halaman | Isi |
|---|---|
| 🤖 Game Master AI | chat -> rekomendasi -> link host + peserta + QR, jejak tool terlihat |
| 🛠️ Buat Room Manual | form parameter acara -> rekomendasi -> pilih game -> link host |
| 🧪 Game Custom | draft game pakai AI, wajib lolos aturan kunci, masuk slot multiverse |
| 📈 Tren & Library | Top-10 + rincian skor (tren / daya tarik / request), grafik sinyal real |
| ✅ Ground Truth | `docs/GROUND_TRUTH.md`: Top-10 STABIL/BERUBAH, backtest, integritas |
| 🧠 Data & Model | hasil 2 model, sumber data, alur sistem, status penyimpanan log |
| 🎮 Room | host + peserta, 5 mekanik (kuis, fakta/hoaks, mayoritas, tebak angka, jawab bebas + voting), varian survival & kubu |

## Aturan kunci & batas komputasi (`core/config.py`)

- Game custom: **4-7 aturan**, wajib peran **host** dan **peserta**, mekanik salah satu dari 5 mekanik engine.
- Pemain 2-30, ronde 3-15, 10-120 detik/ronde, total sesi maks 60 menit.
- Dokumen maks 5 MB / 40 halaman / 20.000 karakter; PDF scan (tanpa teks) ditolak.
- Konten di-cache TTL 30 menit (64 item); room idle > 2 jam dihapus; maks 20 room aktif.
- Slot multiverse: 1 game. Promosi ke Top-10 jika skor >= peringkat 10 dan >= 5 request; dihapus jika setelah 7 hari < 5 request.

## Jenis dokumen

| Jenis | Status | Pertimbangan |
|---|---|---|
| PDF (teks), DOCX, TXT/MD | ✅ | format materi paling umum, parsing ringan |
| Topik saja | ✅ | konten dari LLM (atau materi contoh saat offline) |
| URL artikel | ❌ | sering diblokir/berantakan; tempel teksnya saja |
| Scan/gambar (OCR), PPT, video | ❌ | berat untuk server gratis, rawan error saat demo |

## Menjalankan lokal

```bash
pip install -r requirements-dev.txt
streamlit run app.py                     # http://localhost:8501
pytest -q                                # 37 tes
python scripts/fetch_real_signals.py     # tarik ulang Wikipedia + Google Trends
python scripts/train_model.py            # latih model tren
python scripts/train_appeal_model.py     # latih model daya tarik (BGG)
python scripts/update_ground_truth.py    # regenerate ground truth
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/capstone_analysis.ipynb
```

## Deploy gratis (Streamlit Community Cloud)

1. https://share.streamlit.io -> **Create app** -> repo `harits766/ai-game-generator`, branch `main`, file `app.py`,
   Python 3.12+.
2. **Secrets** (lihat `.streamlit/secrets.toml.example`):
   - `GEMINI_API_KEY` (gratis, https://aistudio.google.com/apikey), opsional `OPENAI_API_KEY` sebagai cadangan
   - `GITHUB_TOKEN`: fine-grained token untuk repo ini saja, izin **Contents: Read and write**, supaya log request
     tersimpan permanen di branch `data`
   - `APP_URL` = URL app kamu
3. Deploy. App gratis bisa "tidur" kalau lama tidak dipakai; buka dulu sebelum acara.

## Struktur

```
app.py                  entry Streamlit (router studio / room)
core/                   config, catalog, real_sources, trend_model, appeal, library, storage,
                        documents, generator, llm, lc_agent, agent, rules, engine, ground_truth
ui/                     common, studio, room
scripts/                fetch_real_signals, fetch_bgg, train_model, train_appeal_model,
                        update_ground_truth, build_notebook
data/real/              data real (Wikipedia, Google Trends, BGG) + log pengambilan
models/                 model terlatih + laporan eksperimen
docs/                   GROUND_TRUTH.md + riwayat
notebooks/              analisis lengkap (EDA, 2 model, simulasi berlabel, demo LangChain)
.github/workflows/      refresh data mingguan
```
