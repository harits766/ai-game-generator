"""Tarik sinyal tren REAL untuk tiap game, mingguan.

Sumber:
1. Wikipedia Pageviews API (Wikimedia, gratis, tanpa key): jumlah view artikel yang mewakili
   tiap jenis game (mis. "Kahoot!" untuk Kuis Kilat, "Family Feud" untuk Satu Frekuensi).
2. Google Trends (pytrends, geo=ID): indeks minat pencarian 0-100 per keyword. Google sering
   membatasi (HTTP 429); keyword yang gagal dibiarkan kosong dan ditandai.

Dijalankan otomatis oleh GitHub Actions (.github/workflows/refresh-real-data.yml) tiap Senin,
atau manual: python scripts/fetch_real_signals.py
Output: data/real/wiki_weekly.csv, data/real/gtrends_weekly.csv, data/real/fetch_log.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.real_sources import CONTEXT_WIKI, GAME_SOURCES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "real"
START = date(2024, 12, 2)
UA = "ai-game-generator-capstone/1.0 (educational project; github.com/harits766/ai-game-generator)"


def week_start(d: pd.Series) -> pd.Series:
    return d.dt.to_period("W-SUN").dt.start_time


def fetch_wiki(project: str, article: str, end: date) -> pd.Series:
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"{project}/all-access/user/{quote(article, safe='')}/daily/"
           f"{START:%Y%m%d}00/{end:%Y%m%d}00")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    items = r.json()["items"]
    s = pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items})
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    log = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wiki": {}, "gtrends": {}}

    # ---------------- Wikipedia ----------------
    rows = []
    targets = [(gid, src["wiki"]) for gid, src in GAME_SOURCES.items()] + \
              [(f"ctx:{name}", [ref]) for name, ref in CONTEXT_WIKI.items()]
    for gid, refs in targets:
        total = None
        for ref in refs:
            project, article = ref.split(":", 1)
            try:
                s = fetch_wiki(f"{project}.wikipedia", article, end)
                total = s if total is None else total.add(s, fill_value=0)
                log["wiki"][ref] = f"ok ({int(s.sum()):,} views)"
            except Exception as e:
                log["wiki"][ref] = f"gagal: {type(e).__name__} {str(e)[:80]}"
            time.sleep(0.2)
        if total is None:
            continue
        df = total.rename("views").to_frame()
        df["week"] = week_start(pd.Series(df.index, index=df.index))
        wk = df.groupby("week")["views"].agg(["sum", "count"]).reset_index()
        wk = wk[wk["count"] == 7]                       # hanya minggu lengkap
        for _, r in wk.iterrows():
            rows.append({"week": r["week"].date().isoformat(), "game_id": gid,
                         "wiki_views": int(r["sum"]), "articles": "|".join(refs)})
    pd.DataFrame(rows).to_csv(OUT / "wiki_weekly.csv", index=False)

    # ---------------- Google Trends ----------------
    rows = []
    try:
        from pytrends.request import TrendReq
        py = TrendReq(hl="id-ID", tz=420, timeout=(10, 30), retries=2, backoff_factor=1.5)
        for gid, src in GAME_SOURCES.items():
            kw = src["gtrends"]
            ok = False
            for attempt in range(3):
                try:
                    py.build_payload([kw], timeframe=f"{START} {end}", geo="ID")
                    df = py.interest_over_time()
                    if not df.empty:
                        s = df[kw].astype(float)
                        s.index = week_start(pd.Series(s.index, index=s.index))
                        s = s.groupby(level=0).mean()
                        for w, v in s.items():
                            rows.append({"week": w.date().isoformat(), "game_id": gid,
                                         "keyword": kw, "google_trends": v})
                    log["gtrends"][kw] = "ok" if not df.empty else "kosong"
                    ok = True
                    break
                except Exception as e:
                    log["gtrends"][kw] = f"gagal: {type(e).__name__}"
                    time.sleep(20 * (attempt + 1))
            time.sleep(8 if ok else 2)
    except ImportError:
        log["gtrends"]["_"] = "pytrends tidak terpasang"
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "gtrends_weekly.csv", index=False)
    (OUT / "fetch_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    ok_w = sum(v.startswith("ok") for v in log["wiki"].values())
    ok_g = sum(v == "ok" for v in log["gtrends"].values())
    print(f"wiki ok {ok_w}/{len(log['wiki'])} | gtrends ok {ok_g}/{len(GAME_SOURCES)}")


if __name__ == "__main__":
    main()
