"""Model daya tarik (cold start) dari data REAL BoardGameGeek.

Pertanyaan: dengan profil seperti ini (jumlah pemain, durasi, usia, kategori, mekanik),
seberapa besar peluang sebuah game party jadi populer (masuk 25% teratas jumlah rating)?
Dipakai untuk game baru/custom yang belum punya riwayat request sama sekali.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, ROOT

APPEAL_PATH = MODELS_DIR / "appeal_model.joblib"
BGG_PATH = ROOT / "data" / "real" / "bgg_party_games.csv"
NUMERIC = ["minplayers", "maxplayers", "playingtime", "minage", "n_categories", "n_mechanics"]


def load_bgg() -> pd.DataFrame:
    df = pd.read_csv(BGG_PATH)
    df["categories"] = df["categories"].fillna("").str.split("|")
    df["mechanics"] = df["mechanics"].fillna("").str.split("|")
    return df


def vocab(df: pd.DataFrame, n_cat: int = 20, n_mech: int = 25) -> tuple[list[str], list[str]]:
    cats = df["categories"].explode().value_counts()
    mechs = df["mechanics"].explode().value_counts()
    return ([c for c in cats.index if c][:n_cat], [m for m in mechs.index if m][:n_mech])


def featurize(df: pd.DataFrame, cats: list[str], mechs: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["minplayers"] = df["minplayers"].clip(1, 10)
    X["maxplayers"] = df["maxplayers"].clip(1, 30)
    X["playingtime"] = df["playingtime"].clip(5, 240)
    X["minage"] = df["minage"].clip(0, 18)
    X["n_categories"] = df["categories"].map(lambda c: len([x for x in c if x]))
    X["n_mechanics"] = df["mechanics"].map(lambda c: len([x for x in c if x]))
    for c in cats:
        X[f"cat_{c}"] = df["categories"].map(lambda v, c=c: int(c in v))
    for m in mechs:
        X[f"mech_{m}"] = df["mechanics"].map(lambda v, m=m: int(m in v))
    return X


_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(APPEAL_PATH)
    return _bundle


def game_profile(game: dict) -> dict:
    """Ubah spesifikasi game (katalog/custom) jadi profil ala BGG."""
    from .real_sources import MECHANIC_TO_BGG
    prof = MECHANIC_TO_BGG.get(game.get("mechanic", "quiz"), {"categories": ["Party Game"], "mechanics": []})
    mechs = list(prof["mechanics"])
    if game.get("variant") == "survival":
        mechs.append("Player Elimination")
    if game.get("variant") == "teams":
        mechs.append("Team-Based Game")
    rounds = int(game.get("rounds") or 10)
    sec = int(game.get("seconds_per_round") or 20)
    return {"minplayers": int(game.get("min_players", 2)), "maxplayers": int(game.get("max_players", 20)),
            "playingtime": max(10, int(rounds * (sec + 15) / 60)), "minage": 10,
            "categories": list(prof["categories"]), "mechanics": mechs + ["Simultaneous Action Selection"]}


def appeal_score(game: dict) -> float:
    """Probabilitas (0-1) game dengan profil ini populer, menurut model BGG."""
    try:
        b = _load()
    except Exception:
        return 0.5
    row = pd.DataFrame([game_profile(game)])
    X = featurize(row, b["cats"], b["mechs"])[b["features"]]
    return float(np.clip(b["model"].predict_proba(X)[0, 1], 0, 1))
