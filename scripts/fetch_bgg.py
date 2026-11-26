"""Unduh dataset REAL BoardGameGeek (via TidyTuesday 2022-01-25) dan simpan subset game party.

Sumber: https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25
(data BoardGameGeek: detail game + rating pengguna, ~21 ribu game).
Output: data/real/bgg_party_games.csv (game dengan kategori party/trivia/deduksi/bluffing/kata/humor/edukasi)
"""
import ast
import io
import sys
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "real"
BASE = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/"
PARTY = {"Party Game", "Trivia", "Deduction", "Bluffing", "Word Game", "Humor", "Educational"}


def _list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def main(local_dir: str | None = None):
    def load(name):
        if local_dir:
            return pd.read_csv(Path(local_dir) / f"{name}.csv")
        r = requests.get(BASE + f"{name}.csv", timeout=120)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    det, rat = load("details"), load("ratings")
    det["categories"] = det["boardgamecategory"].map(_list)
    det["mechanics"] = det["boardgamemechanic"].map(_list)
    det = det[det["categories"].map(lambda c: bool(PARTY & set(c)))]
    df = det.merge(rat[["id", "users_rated", "average", "bayes_average", "rank"]], on="id", how="left")
    df = df.rename(columns={"primary": "name"})
    df["categories"] = df["categories"].map("|".join)
    df["mechanics"] = df["mechanics"].map("|".join)
    cols = ["id", "name", "yearpublished", "minplayers", "maxplayers", "playingtime", "minage",
            "categories", "mechanics", "owned", "wishing", "users_rated", "average", "bayes_average", "rank"]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT / "bgg_party_games.csv", index=False)
    print(f"{len(df):,} game party disimpan")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
