# 🎲 AI Game Generator

Generator game party edukatif berbasis **tren + dokumen + Generative AI**. Ceritakan acaramu
(jumlah orang, durasi, jenis acara, online/offline, lokasi, topik) dan upload materi (PDF/DOCX/TXT/MD).
AI memilih game yang paling pas dari library yang lagi rame, membuat soalnya dari dokumen, lalu langsung
memberi **link HOST** dan **link PESERTA** (plus QR) untuk dimainkan di HP masing-masing.

> Capstone Project **Data Science & Generative AI** (Celerates, 26 Nov - 18 Des 2026).

## Masalah dan solusi

Acara kumpul sering berakhir dengan semua orang main HP sendiri-sendiri. Game party yang ada kaku:
pilihannya itu-itu saja (mis. cuma satu jenis game), harus gonta-ganti aplikasi, dan tidak menyesuaikan materi acara.
Aplikasi ini mengubah HP justru jadi alat main bareng, dengan game yang dibuat otomatis sesuai acara dan materi.

## Fitur

| Fitur | Isi |
|---|---|
| 🤖 **Game Master AI** | Chatbot agentic (Gemini, cadangan OpenAI, fallback offline) dengan 6 tools: `get_trends`, `recommend_games`, `read_document`, `validate_custom_game`, `register_custom_game`, `create_room`. |
| 🛠️ **Buat Room Manual** | Form parameter acara -> rekomendasi model -> pilih game -> link host. |
| 🧪 **Game Custom (Multiverse)** | Bikin game sendiri (atau draft pakai AI) selama lolos **aturan kunci**. Masuk 1 slot eksperimen; dipromosikan ke Top-10 kalau rame, dihapus kalau sepi. |
| 📈 **Tren & Library** | Top-10 hasil prediksi model dari Google Trends, TikTok, X, dan request in-app. |
| ✅ **Ground Truth** | Satu dokumen acuan (`docs/GROUND_TRUTH.md`) yang di-regenerate dan memvalidasi apakah Top-10 masih sama. |
| 🧠 **Data & Model** | Hasil eksperimen 5 model + baseline, feature importance, ringkasan data. |
| 🎮 **Room realtime** | Host (laptop/HP) + peserta (HP). 5 mekanik: kuis, fakta/hoaks, mayoritas, tebak angka, jawab bebas + voting. Varian survival (gugur) dan kubu vs kubu. |

## Arsitektur

```
Dokumen (PDF/DOCX/TXT/MD) / topik ─┐
Parameter acara ───────────────────┤
                                   ▼
             ┌──────────── Agent GenAI (core/agent.py) ─────────────┐
             │  LLM: Gemini → OpenAI → planner offline              │
             │  tools: tren · rekomendasi · baca dokumen ·          │
             │         validasi aturan · daftar custom · buat room  │
             └──────┬──────────────────┬─────────────────┬──────────┘
                    ▼                  ▼                 ▼
   Model prediksi (ML)        Generator konten       Room engine
   core/library.py            core/generator.py      core/engine.py
   Top-10 + multiverse        retrieval TF-IDF +     host/peserta, timer
   dari sinyal tren           LLM JSON + cache TTL   berbasis deadline
                    │                                    │
                    └──── setiap room = request baru ◄───┘
                                   ▼
                  Ground truth (core/ground_truth.py)
                  docs/GROUND_TRUTH.md: Top-10 STABIL / BERUBAH
```

## Aturan kunci & batas komputasi (`core/config.py`)

- Game custom: **4-7 aturan**, wajib peran **host** dan **peserta**, mekanik harus salah satu dari 5 mekanik engine.
- Pemain 2-30, ronde 3-15, 10-120 detik/ronde, total sesi maks 60 menit.
- Dokumen maks 5 MB / 40 halaman / 20.000 karakter; PDF hasil scan (tanpa teks) ditolak.
- Maks 2 panggilan LLM per room, maks 6 langkah agent per chat.
- Cache konten TTL 30 menit (64 item); room idle > 2 jam dihapus; maks 20 room aktif.

## Jenis dokumen yang didukung

| Jenis | Status | Pertimbangan |
|---|---|---|
| PDF (teks) | ✅ | Format materi paling umum. Parsing ringan (pypdf). |
| DOCX | ✅ | Umum untuk materi buatan sendiri. |
| TXT / MD | ✅ | Paling ringan, cocok untuk tempel teks. |
| Topik saja | ✅ | Tanpa dokumen, konten dari LLM (atau materi contoh saat offline). |
| URL artikel | ❌ | Sering diblokir/berantakan HTML-nya; tempel teksnya saja. |
| Scan/gambar (OCR), PPT, video | ❌ | Berat untuk server gratis dan rawan error saat demo. |

## Data

Data **sintetis** (diizinkan ketentuan capstone), dibuat `scripts/generate_synthetic.py`:
- `data/raw/requests_log.csv`: ~40 ribu permintaan room (2 tahun, 15 game) dengan parameter acara.
- `data/raw/signals_weekly.csv`: sinyal mingguan Google Trends, TikTok, X per keyword game.

Google Trends asli bisa diambil dengan `python scripts/fetch_trends.py` (pytrends). TikTok/X tidak punya API gratis
yang stabil sehingga tetap sintetis.

## Model

Target: jumlah request game minggu depan. Split waktu (26 minggu terakhir = test).

| Model | MAE | Precision@10 |
|---|---|---|
| Baseline (naive) | 6.91 | 0.86 |
| **Ridge Regression (dipakai)** | **6.11** | 0.90 |
| Random Forest | 6.41 | 0.90 |
| Gradient Boosting (HistGB) | 6.33 | 0.89 |
| MLP (Neural Net) | 6.83 | 0.88 |
| Poisson GLM | 11.28 | 0.89 |

Detail di `notebooks/capstone_analysis.ipynb` dan `models/model_report.json`.

## Menjalankan lokal

```bash
pip install -r requirements-dev.txt
streamlit run app.py                     # buka http://localhost:8501
pytest -q                                # 27 tes
python scripts/update_ground_truth.py    # regenerate ground truth
```

Regenerasi data & model dari nol:

```bash
python scripts/generate_synthetic.py
python scripts/train_model.py
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/capstone_analysis.ipynb
```

## Deploy gratis ke Streamlit Community Cloud

1. Buka https://share.streamlit.io, login GitHub, **Create app** -> repo `harits766/ai-game-generator`, branch `main`, file `app.py`.
   Di **Advanced settings** pilih Python 3.12 atau lebih baru.
2. **Secrets** (opsional, tanpa ini app jalan mode offline):
   ```toml
   GEMINI_API_KEY = "..."   # gratis di https://aistudio.google.com/apikey
   OPENAI_API_KEY = "..."   # cadangan
   APP_URL = "https://<nama-app>.streamlit.app"
   ```
3. Deploy. Link host/peserta otomatis memakai `APP_URL`.

**Catatan:** state room disimpan di memori 1 instance Streamlit (cukup untuk ~30 pemain per room). App gratis
akan "tidur" kalau tidak dipakai beberapa hari; buka dulu link app-nya sebelum acara.

## Struktur

```
app.py                  entry Streamlit (router studio / room)
core/                   logika: config, catalog, features, library, documents, generator,
                        llm, agent, rules, engine, ground_truth
ui/                     tampilan: common, studio, room
scripts/                generate_synthetic, train_model, update_ground_truth, fetch_trends, build_notebook
data/raw/               dataset; data/processed/ panel fitur
models/                 model terlatih + laporan eksperimen
docs/                   GROUND_TRUTH.md + riwayat
notebooks/              analisis lengkap (EDA, modeling, simulasi, demo GenAI)
samples/                materi contoh (pemilahan sampah, literasi digital)
tests/                  pytest
```
