"""Model prediksi tren REAL: minat publik minggu depan per jenis game.

Data: Wikipedia pageviews mingguan (data/real/wiki_weekly.csv) + Google Trends Indonesia
(data/real/gtrends_weekly.csv, kalau berhasil diambil).
Target: log(1 + views) minggu depan. Dari prediksi itu dihitung "momentum" =
prediksi minggu depan dibanding rata-rata 12 minggu terakhir (lagi naik atau turun).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROOT

REAL_DIR = ROOT / "data" / "real"
RAMADAN = [("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19"), ("2027-02-08", "2027-03-09")]

FEATURES = ["lv_t", "lv_lag1", "lv_lag2", "lv_lag3", "roll4", "roll12", "mom4", "mom12", "vol4",
            "gt_t", "gt_delta", "has_gt", "woy_sin", "woy_cos",
            "next_ramadan", "next_august", "next_yearend", "next_school_holiday"]
TARGET = "lv_next"


def load_signals() -> pd.DataFrame:
    wiki = pd.read_csv(REAL_DIR / "wiki_weekly.csv", parse_dates=["week"])
    gt_path = REAL_DIR / "gtrends_weekly.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path, parse_dates=["week"])[["week", "game_id", "google_trends"]]
        wiki = wiki.merge(gt, on=["week", "game_id"], how="left")
    else:
        wiki["google_trends"] = np.nan
    return wiki.sort_values(["game_id", "week"]).reset_index(drop=True)


def _flags(week: pd.Timestamp) -> dict:
    nxt = week + pd.Timedelta(weeks=1)
    ram = any(pd.Timestamp(a) - pd.Timedelta(days=6) <= nxt <= pd.Timestamp(b) for a, b in RAMADAN)
    return {"next_ramadan": int(ram),
            "next_august": int(nxt.month == 8 and nxt.day <= 24),
            "next_yearend": int((nxt.month == 12 and nxt.day >= 15) or (nxt.month == 1 and nxt.day <= 3)),
            "next_school_holiday": int(nxt.month == 6 or (nxt.month == 7 and nxt.day <= 14))}


def build_panel(sig: pd.DataFrame | None = None) -> pd.DataFrame:
    sig = load_signals() if sig is None else sig.copy()
    p = sig.copy()
    p["lv_t"] = np.log1p(p["wiki_views"])
    g = p.groupby("game_id")
    for k in (1, 2, 3):
        p[f"lv_lag{k}"] = g["lv_t"].shift(k)
    p["roll4"] = g["lv_t"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    p["roll12"] = g["lv_t"].transform(lambda s: s.rolling(12, min_periods=4).mean())
    p["mom4"] = p["lv_t"] - p["roll4"]
    p["mom12"] = p["lv_t"] - p["roll12"]
    p["vol4"] = g["lv_t"].transform(lambda s: s.rolling(4, min_periods=2).std())
    p["has_gt"] = p["google_trends"].notna().astype(int)
    p["gt_t"] = p["google_trends"].fillna(-1)
    p["gt_delta"] = g["google_trends"].diff().fillna(0)
    # momentum Google Trends Indonesia: minggu ini dibanding rata-rata 12 minggu (bukan fitur model)
    p["gt_mom"] = p["google_trends"] / g["google_trends"].transform(
        lambda s: s.rolling(12, min_periods=4).mean()).replace(0, np.nan)
    woy = p["week"].dt.isocalendar().week.astype(float)
    p["woy_sin"], p["woy_cos"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    flags = pd.DataFrame([_flags(w) for w in p["week"]], index=p.index)
    p = pd.concat([p, flags], axis=1)
    p[TARGET] = g["lv_t"].shift(-1)
    return p.dropna(subset=["lv_lag3", "roll12"]).reset_index(drop=True)
