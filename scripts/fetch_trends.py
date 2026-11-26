"""Ambil Google Trends ASLI (opsional) dan timpa kolom google_trends di data/raw/signals_weekly.csv.

Butuh internet dan `pip install pytrends`. Google sering membatasi request (HTTP 429);
script ini mengambil per keyword dengan jeda dan melewati keyword yang gagal.
TikTok/X tidak punya API gratis yang stabil, jadi kolom itu tetap sintetis
(alternatif: isi manual dari TikTok Creative Center / hasil scraping yang diizinkan).

Jalankan: python scripts/fetch_trends.py --geo ID
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import RAW_DIR  # noqa: E402


def main(geo: str, sleep: float):
    from pytrends.request import TrendReq
    sig = pd.read_csv(RAW_DIR / "signals_weekly.csv", parse_dates=["week"])
    start, end = sig["week"].min().date(), sig["week"].max().date()
    py = TrendReq(hl="id-ID", tz=420)
    ok = 0
    for kw in sig["keyword"].unique():
        try:
            py.build_payload([kw], timeframe=f"{start} {end}", geo=geo)
            df = py.interest_over_time()
            if df.empty:
                continue
            s = df[kw].resample("W-MON").mean()
            m = sig["keyword"] == kw
            sig.loc[m, "google_trends"] = sig.loc[m, "week"].map(s).fillna(sig.loc[m, "google_trends"]).values
            sig.loc[m, "source_gtrends"] = "google_trends_asli"
            ok += 1
            print("OK  ", kw)
        except Exception as e:  # 429 / jaringan
            print("GAGAL", kw, type(e).__name__)
        time.sleep(sleep)
    sig["week"] = sig["week"].dt.date
    sig.to_csv(RAW_DIR / "signals_weekly.csv", index=False)
    print(f"{ok} keyword diperbarui. Jalankan ulang scripts/train_model.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="ID")
    ap.add_argument("--sleep", type=float, default=8)
    a = ap.parse_args()
    main(a.geo, a.sleep)
