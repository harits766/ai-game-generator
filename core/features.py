"""Preprocessing + feature engineering. Dipakai bersama oleh notebook, training, dan app.

Unit analisis: (game_id, minggu). Target: jumlah request minggu DEPAN.
"""
from __future__ import annotations

import pandas as pd

from .catalog import GAMES, MECHANICS
from .config import RAW_DIR, RUNTIME_REQUESTS

RAMADAN = [("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19"),
           ("2027-02-08", "2027-03-09")]

FEATURES = [
    "req_t", "req_lag1", "req_lag2", "req_lag3", "roll4_mean", "roll4_std", "growth",
    "gtrend_t", "gtrend_delta", "tiktok_t", "tiktok_delta", "x_t",
    "weeks_since_launch", "next_ramadan", "next_august", "next_yearend", "next_school_holiday",
] + [f"mech_{m}" for m in MECHANICS]
TARGET = "req_next"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    reqs = pd.read_csv(RAW_DIR / "requests_log.csv")
    sig = pd.read_csv(RAW_DIR / "signals_weekly.csv")
    return reqs, sig


def clean_requests(reqs: pd.DataFrame) -> pd.DataFrame:
    """Hapus duplikat, nilai tidak valid, dan tambahkan kolom minggu."""
    df = reqs.drop_duplicates("request_id").copy()
    df = df[df["players"].between(1, 100)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["week"] = df["timestamp"].dt.to_period("W-SUN").dt.start_time
    return df.reset_index(drop=True)


def load_runtime_requests() -> pd.DataFrame:
    if RUNTIME_REQUESTS.exists():
        try:
            return pd.read_csv(RUNTIME_REQUESTS)
        except Exception:  # file rusak -> abaikan
            pass
    return pd.DataFrame(columns=["request_id", "timestamp", "game_id"])


def _season_flags(week: pd.Timestamp) -> dict:
    nxt = week + pd.Timedelta(weeks=1)
    ram = any(pd.Timestamp(a) - pd.Timedelta(days=6) <= nxt <= pd.Timestamp(b) for a, b in RAMADAN)
    return {
        "next_ramadan": int(ram),
        "next_august": int(nxt.month == 8 and nxt.day <= 24),
        "next_yearend": int((nxt.month == 12 and nxt.day >= 15) or (nxt.month == 1 and nxt.day <= 3)),
        "next_school_holiday": int(nxt.month == 6 or (nxt.month == 7 and nxt.day <= 14)),
    }


def build_panel(reqs_clean: pd.DataFrame, sig: pd.DataFrame,
                extra_games: list[dict] | None = None) -> pd.DataFrame:
    """Panel mingguan lengkap (game x minggu) + fitur + target."""
    games = list(GAMES) + list(extra_games or [])
    mech = {g["id"]: g["mechanic"] for g in games}
    sig = sig.copy()
    sig["week"] = pd.to_datetime(sig["week"])
    weeks = sorted(set(sig["week"]) | set(reqs_clean["week"]))
    weeks = pd.date_range(min(weeks), max(weeks), freq="W-MON")

    counts = reqs_clean.groupby(["game_id", "week"]).size().rename("req_t")
    idx = pd.MultiIndex.from_product([[g["id"] for g in games], weeks], names=["game_id", "week"])
    panel = counts.reindex(idx, fill_value=0).reset_index()
    panel = panel.merge(sig[["game_id", "week", "google_trends", "tiktok_mentions", "x_mentions"]],
                        on=["game_id", "week"], how="left")
    panel = panel.sort_values(["game_id", "week"]).reset_index(drop=True)

    # game tanpa sinyal (mis. game custom) -> isi 0; sinyal minggu terakhir yang kosong -> ffill
    for c in ["google_trends", "tiktok_mentions", "x_mentions"]:
        panel[c] = panel.groupby("game_id")[c].ffill().fillna(0)

    g = panel.groupby("game_id")
    for k in (1, 2, 3):
        panel[f"req_lag{k}"] = g["req_t"].shift(k).fillna(0)
    panel["roll4_mean"] = g["req_t"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    panel["roll4_std"] = g["req_t"].transform(lambda s: s.rolling(4, min_periods=1).std()).fillna(0)
    panel["growth"] = (panel["req_t"] + 1) / (panel["roll4_mean"] + 1)
    panel["gtrend_t"] = panel["google_trends"]
    panel["gtrend_delta"] = g["google_trends"].diff().fillna(0)
    panel["tiktok_t"] = panel["tiktok_mentions"]
    panel["tiktok_delta"] = g["tiktok_mentions"].diff().fillna(0)
    panel["x_t"] = panel["x_mentions"]

    first = panel[panel["req_t"] > 0].groupby("game_id")["week"].min()
    panel["launch"] = panel["game_id"].map(first)
    panel["weeks_since_launch"] = ((panel["week"] - panel["launch"]).dt.days // 7).clip(lower=-1).fillna(-1)

    flags = pd.DataFrame([_season_flags(w) for w in panel["week"]])
    panel = pd.concat([panel, flags], axis=1)
    for m in MECHANICS:
        panel[f"mech_{m}"] = (panel["game_id"].map(mech) == m).astype(int)

    panel[TARGET] = g["req_t"].shift(-1)
    # baris sebelum launch tidak informatif untuk training
    panel["active"] = panel["weeks_since_launch"] >= 0
    return panel.drop(columns=["launch"])
