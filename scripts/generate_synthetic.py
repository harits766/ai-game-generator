"""Generate dataset sintetis (diizinkan oleh ketentuan capstone: data sintetik buatan sendiri).

Output (data/raw/):
- signals_weekly.csv  : sinyal tren mingguan per keyword game
                        (google_trends 0-100, tiktok_mentions, x_mentions)
- requests_log.csv    : log permintaan game per event (1 baris = 1 permintaan room)

Data dibuat dengan proses generatif yang eksplisit supaya bisa dijelaskan di laporan:
  sinyal sosial (TikTok/X) MENDAHULUI Google Trends ~1 minggu,
  dan permintaan game dipengaruhi tren minggu sebelumnya, musim (Ramadan, 17-an,
  akhir tahun, masa sekolah), umur game (novelty), dan noise Poisson.
Kalau mau pakai Google Trends asli, jalankan scripts/fetch_trends.py; kolom
google_trends akan ditimpa nilai asli untuk keyword yang berhasil diambil.

Jalankan: python scripts/generate_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.catalog import GAMES  # noqa: E402
from core.config import DURATIONS, EVENT_TYPES, LOCATIONS, MODES, RAW_DIR  # noqa: E402

SEED = 42
N_WEEKS = 104
START = pd.Timestamp("2024-12-02")  # Senin

RAMADAN = [(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-30")),
           (pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19"))]

BASE_POP = {
    "kuis_kilat": 60, "fakta_hoaks": 35, "satu_frekuensi": 40, "tebak_angka": 18,
    "adu_receh": 28, "debat_kilat": 16, "survival_kuis": 30, "tim_vs_tim": 26,
    "emoji_tebak": 22, "siapa_paling": 24, "cerita_berantai": 10, "mitos_sains": 20,
    "harga_pas": 12, "tebak_kata_kunci": 30, "voting_meme": 34,
}

# afinitas jenis acara per mekanik (dipakai untuk sampling parameter request)
EVENT_AFFINITY = {
    "quiz":      [3, 3, 2, 2, 2, 3, 2],
    "truefalse": [2, 3, 2, 2, 2, 4, 3],
    "majority":  [2, 1, 3, 4, 3, 1, 1],
    "estimate":  [1, 2, 3, 2, 2, 2, 2],
    "open_vote": [1, 3, 3, 3, 4, 1, 1],
}
TOPICS = ["agama", "pendidikan", "kesehatan", "lingkungan", "teknologi",
          "keuangan", "sejarah", "budaya", "umum"]
DOC_TYPES = ["pdf", "docx", "txt", "md", "topic"]


def in_ramadan(d: pd.Timestamp) -> bool:
    return any(a - pd.Timedelta(days=6) <= d <= b for a, b in RAMADAN)


def season_factor(d: pd.Timestamp) -> float:
    f = 1.0
    if in_ramadan(d):
        f *= 1.35                      # banyak kajian/pesantren kilat
    if d.month == 8 and d.day <= 24:
        f *= 1.45                      # lomba 17-an
    if d.month == 12 and d.day >= 15 or d.month == 1 and d.day <= 3:
        f *= 0.75                      # libur akhir tahun
    if d.month in (6, 7) and not (d.month == 7 and d.day > 14):
        f *= 0.8                       # libur sekolah
    return f


def main() -> None:
    rng = np.random.default_rng(SEED)
    weeks = [START + pd.Timedelta(weeks=i) for i in range(N_WEEKS)]

    # ---------- sinyal tren ----------
    sig_rows = []
    latent = {}
    for g in GAMES:
        # random walk laten + spike viral acak
        walk = np.cumsum(rng.normal(0, 0.06, N_WEEKS))
        spikes = np.zeros(N_WEEKS)
        for _ in range(rng.integers(1, 4)):
            t0 = rng.integers(5, N_WEEKS - 3)
            amp = rng.uniform(0.4, 1.2)
            for k in range(6):
                if t0 + k < N_WEEKS:
                    spikes[t0 + k] += amp * np.exp(-k / 1.5)
        if g["launch_week"] > 0:   # game baru: tren naik setelah launch
            ramp = np.clip((np.arange(N_WEEKS) - g["launch_week"] + 4) / 10, 0, 1.2)
        else:
            ramp = np.ones(N_WEEKS)
        lat = np.exp(walk) * (1 + spikes) * ramp
        latent[g["id"]] = lat
        for i, w in enumerate(weeks):
            social_now = lat[i] * season_factor(w)
            search_now = (lat[i - 1] if i else lat[i]) * season_factor(w)  # search telat 1 minggu
            sig_rows.append({
                "week": w.date().isoformat(), "game_id": g["id"], "keyword": g["keywords"][0],
                "tiktok_mentions": int(rng.poisson(max(0.1, 120 * social_now))),
                "x_mentions": int(rng.poisson(max(0.1, 45 * social_now))),
                "google_trends": float(np.clip(search_now * 35 + rng.normal(0, 3), 0, 100)),
            })
    signals = pd.DataFrame(sig_rows)
    # normalisasi google_trends ke skala 0-100 per keyword (seperti Google Trends asli)
    signals["google_trends"] = signals.groupby("keyword")["google_trends"].transform(
        lambda s: (100 * s / s.max()).round(0))

    # ---------- log request ----------
    req_rows = []
    rid = 0
    for g in GAMES:
        lat = latent[g["id"]]
        aff = np.array(EVENT_AFFINITY[g["mechanic"]], dtype=float)
        aff /= aff.sum()
        for i, w in enumerate(weeks):
            if i < g["launch_week"]:
                continue
            age = i - g["launch_week"]
            novelty = 1.0 + (0.8 * np.exp(-age / 6) if g["launch_week"] else 0)
            fatigue = 1.0 - 0.15 * min(age, 104) / 104  # game lama sedikit membosankan
            prev = lat[i - 1] if i else lat[i]
            lam = BASE_POP[g["id"]] * (0.55 + 0.45 * prev) * season_factor(w) * novelty * fatigue
            n = rng.poisson(max(lam, 0.2))
            for _ in range(n):
                rid += 1
                ts = w + pd.Timedelta(minutes=int(rng.integers(0, 7 * 24 * 60)))
                ev = rng.choice(EVENT_TYPES, p=aff)
                mode = rng.choice(MODES, p=[0.6, 0.3, 0.1] if ev != "kantor" else [0.35, 0.5, 0.15])
                players = int(np.clip(rng.lognormal(2.4, 0.5), g["min_players"], g["max_players"]))
                dur = int(rng.choice(DURATIONS, p=[0.12, 0.25, 0.22, 0.25, 0.1, 0.06]))
                topic = "agama" if ev == "kajian" and rng.random() < 0.7 else rng.choice(TOPICS)
                loc = rng.choice(LOCATIONS, p=np.array([14, 18, 12, 8, 14, 6, 4, 6, 4, 6, 4, 4]) / 100)
                req_rows.append({
                    "request_id": f"R{rid:06d}", "timestamp": ts.isoformat(),
                    "game_id": g["id"], "players": players, "duration_min": dur,
                    "event_type": ev, "mode": mode, "location": loc, "topic": topic,
                    "doc_type": rng.choice(DOC_TYPES, p=[0.38, 0.17, 0.1, 0.05, 0.3]),
                    # rating sesi 1-5 (sesudah main); ~8% kosong untuk latihan cleaning
                    "rating": (np.nan if rng.random() < 0.08 else
                               int(np.clip(round(rng.normal(3.6 + 0.6 * (prev - 1), 0.9)), 1, 5))),
                })
    reqs = pd.DataFrame(req_rows).sort_values("timestamp").reset_index(drop=True)
    # sedikit data kotor: duplikat & pemain tidak valid, untuk tahap preprocessing
    dup = reqs.sample(frac=0.01, random_state=SEED)
    bad = reqs.sample(frac=0.005, random_state=SEED + 1).assign(players=0)
    reqs = pd.concat([reqs, dup, bad]).sort_values("timestamp").reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_csv(RAW_DIR / "signals_weekly.csv", index=False)
    reqs.to_csv(RAW_DIR / "requests_log.csv", index=False)
    print(f"signals: {len(signals):,} baris | requests: {len(reqs):,} baris")


if __name__ == "__main__":
    main()
