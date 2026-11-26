"""Library game: Top-N berdasarkan prediksi model + 1 slot "multiverse" untuk game baru.

Alur:
1. compute_scores()   -> skor tiap game = tren real (model) + daya tarik (BGG) + request in-app
2. refresh_library()  -> ambil Top-10, nilai game multiverse (promosi / hapus), simpan state
3. recommend()        -> ranking game untuk parameter acara user (demand x kecocokan)
"""
from __future__ import annotations

import re

import csv
import json
import threading
import time
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from cachetools import TTLCache, cached

from .catalog import GAMES, get_game
from .config import (LIBRARY_STATE, MODEL_PATH, MULTIVERSE_GRACE_DAYS, MULTIVERSE_MIN_REQUESTS,
                     RUNTIME_DIR, RUNTIME_REQUESTS, TOP_N)

_lock = threading.RLock()
_pred_cache: TTLCache = TTLCache(maxsize=4, ttl=300)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _empty_state() -> dict:
    return {"top": [], "multiverse": None, "queue": [], "custom_games": {}, "archive": [],
            "updated_at": None, "events": []}


def load_state() -> dict:
    with _lock:
        if LIBRARY_STATE.exists():
            try:
                return {**_empty_state(), **json.loads(LIBRARY_STATE.read_text())}
            except Exception:
                pass
        return _empty_state()


def save_state(state: dict) -> None:
    with _lock:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        state["events"] = state.get("events", [])[-50:]
        LIBRARY_STATE.write_text(json.dumps(state, indent=2, default=str))


def _log(state: dict, msg: str) -> None:
    state.setdefault("events", []).append({"at": _now().isoformat(timespec="seconds"), "msg": msg})


# ---------------------------------------------------------------------------
# request logging (sinyal in-app)
# ---------------------------------------------------------------------------
def log_request(game_id: str, params: dict) -> None:
    """Setiap room yang dibuat dicatat -> jadi data untuk prediksi berikutnya."""
    with _lock:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        new = not RUNTIME_REQUESTS.exists()
        fields = ["request_id", "timestamp", "game_id", "players", "duration_min",
                  "event_type", "mode", "location", "topic", "doc_type", "rating"]
        with RUNTIME_REQUESTS.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({
                "request_id": "U" + uuid.uuid4().hex[:10],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "game_id": game_id,
                "players": params.get("players", 10),
                "duration_min": params.get("duration_min", 20),
                "event_type": params.get("event_type", ""),
                "mode": params.get("mode", ""),
                "location": params.get("location", ""),
                "topic": params.get("topic", ""),
                "doc_type": params.get("doc_type", "topic"),
                "rating": "",
            })
        _pred_cache.clear()
    from .storage import schedule_sync
    schedule_sync()


def load_runtime_requests() -> pd.DataFrame:
    from .storage import pull_once
    pull_once()
    if RUNTIME_REQUESTS.exists():
        try:
            return pd.read_csv(RUNTIME_REQUESTS)
        except Exception:
            pass
    return pd.DataFrame(columns=["request_id", "timestamp", "game_id"])


def runtime_counts(days: int | None = None) -> dict:
    rt = load_runtime_requests()
    if not len(rt):
        return {}
    if days:
        ts = pd.to_datetime(rt["timestamp"], errors="coerce")
        rt = rt[ts >= pd.Timestamp.now() - pd.Timedelta(days=days)]
    return rt["game_id"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# skor ranking: tren real + daya tarik (BGG) + request in-app
# ---------------------------------------------------------------------------
REQ_HALF_WEIGHT = 50     # saat ada 50 request in-app (28 hari), bobotnya 50% dari maks
REQ_MAX_WEIGHT = 0.7     # request in-app maksimal menentukan 70% skor


@cached(TTLCache(maxsize=1, ttl=3600))
def _model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # versi scikit-learn beda / file hilang -> latih ulang (~10 detik) lalu muat
        from scripts.train_model import main as train
        train()
        return joblib.load(MODEL_PATH)


@cached(TTLCache(maxsize=1, ttl=3600))
def _trend_now() -> pd.DataFrame:
    """Prediksi minat minggu depan untuk setiap jenis game di katalog (data real)."""
    from .trend_model import FEATURES as TF
    from .trend_model import build_panel as trend_panel
    panel = trend_panel()
    last = panel.groupby("game_id")["week"].transform("max")
    cur = panel[panel["week"] == last].copy()
    b = _model()
    delta = b["model"].predict(cur[TF]) if b.get("target") == "delta" else b["model"].predict(cur[TF]) - cur["lv_t"]
    cur["pred_lv"] = cur["lv_t"] + delta
    cur["views_t"] = np.expm1(cur["lv_t"]).round()
    cur["pred_views"] = np.expm1(cur["pred_lv"]).round()
    cur["momentum"] = cur["pred_lv"] - cur["roll12"]          # >0 = lagi naik dibanding 3 bulan terakhir
    # artikel dengan view sangat sedikit -> momentum berisik; disusutkan ke 0 (shrinkage)
    cur["momentum"] = cur["momentum"] * cur["views_t"] / (cur["views_t"] + 200)
    return cur[["game_id", "week", "views_t", "pred_views", "momentum", "google_trends", "gt_mom"]]


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) if len(s) > 1 else pd.Series(1.0, index=s.index)


def compute_scores() -> pd.DataFrame:
    """Skor 0-1 tiap game (katalog + custom). Kolom 'pred' = skor x 100 (untuk tampilan)."""
    key = (RUNTIME_REQUESTS.stat().st_mtime if RUNTIME_REQUESTS.exists() else 0,
           LIBRARY_STATE.stat().st_mtime if LIBRARY_STATE.exists() else 0)
    if key in _pred_cache:
        return _pred_cache[key]
    from .appeal import appeal_score
    state = load_state()
    games = list(GAMES) + [g for g in state["custom_games"].values() if g.get("status") != "evicted"]
    df = pd.DataFrame({"game_id": [g["id"] for g in games], "name": [g["name"] for g in games],
                       "custom": [bool(g.get("custom")) for g in games]})
    df["appeal"] = [appeal_score(g) for g in games]
    tr = _trend_now()
    df = df.merge(tr, on="game_id", how="left")
    cat = ~df["custom"]
    df["trend_part"] = 0.0
    gt = df.loc[cat, "gt_mom"].fillna(1.0)
    df.loc[cat, "trend_part"] = (0.40 * _pct(df.loc[cat, "momentum"]) + 0.35 * _pct(gt)
                                 + 0.25 * _pct(df.loc[cat, "views_t"]))
    df["appeal_part"] = _pct(df["appeal"])
    df["base"] = np.where(df["custom"], df["appeal_part"], 0.65 * df["trend_part"] + 0.35 * df["appeal_part"])

    counts = runtime_counts(days=28)
    n = sum(counts.values())
    w_req = min(REQ_MAX_WEIGHT, n / (n + REQ_HALF_WEIGHT)) if n else 0.0
    df["req_28d"] = df["game_id"].map(counts).fillna(0).astype(int)
    df["req_part"] = df["req_28d"] / max(df["req_28d"].max(), 1)
    df["w_req"] = round(w_req, 3)
    df["score"] = (1 - w_req) * df["base"] + w_req * df["req_part"]
    df["pred"] = (100 * df["score"]).round(1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    _pred_cache[key] = df
    return df


predict_demand = compute_scores   # nama lama, tetap didukung


# ---------------------------------------------------------------------------
# library refresh + multiverse
# ---------------------------------------------------------------------------
def refresh_library() -> dict:
    with _lock:
        state = load_state()
        pred = compute_scores().set_index("game_id")["score"]
        counts = runtime_counts()
        pool_ids = [g["id"] for g in GAMES]
        promoted = [gid for gid in state["custom_games"]
                    if state["custom_games"][gid].get("status") == "promoted"]
        ranked = sorted(pool_ids + promoted, key=lambda gid: -float(pred.get(gid, 0)))
        old_top = list(state.get("top", []))
        top = ranked[:TOP_N]

        mv = state.get("multiverse")
        if mv:
            gid = mv["id"]
            age_days = (_now() - datetime.fromisoformat(mv["created_at"])).total_seconds() / 86400
            n_req = int(counts.get(gid, 0))
            p = float(pred.get(gid, 0))
            cutoff = float(pred.get(top[-1], 0)) if top else 0
            mv.update({"requests": n_req, "pred": round(p, 3), "age_days": round(age_days, 2)})
            if p >= cutoff and n_req >= MULTIVERSE_MIN_REQUESTS:
                state["custom_games"][gid]["status"] = "promoted"
                dropped = top[-1]
                top = top[:-1] + [gid]
                _log(state, f"PROMOSI: '{mv['name']}' masuk Top-{TOP_N}, menggantikan '{dropped}'.")
                state["archive"].append({"id": dropped, "reason": "tergeser game multiverse",
                                         "at": _now().isoformat(timespec="seconds")})
                state["multiverse"] = None
            elif age_days >= MULTIVERSE_GRACE_DAYS and n_req < MULTIVERSE_MIN_REQUESTS:
                state["custom_games"][gid]["status"] = "evicted"
                state["archive"].append({"id": gid, "reason": f"sepi peminat ({n_req} request)",
                                         "at": _now().isoformat(timespec="seconds")})
                _log(state, f"HAPUS: '{mv['name']}' dikeluarkan dari multiverse (sepi peminat).")
                state["multiverse"] = None
        if state["multiverse"] is None and state["queue"]:
            nxt = state["queue"].pop(0)
            _occupy(state, nxt)

        for gid in set(old_top) - set(top):
            if gid in pool_ids:
                _log(state, f"Top-{TOP_N}: '{gid}' keluar.")
        for gid in set(top) - set(old_top):
            if old_top:
                _log(state, f"Top-{TOP_N}: '{gid}' masuk.")
        state["top"] = top
        state["updated_at"] = _now().isoformat(timespec="seconds")
        save_state(state)
        return state


def _occupy(state: dict, gid: str) -> None:
    g = state["custom_games"][gid]
    g["status"] = "multiverse"
    state["multiverse"] = {"id": gid, "name": g["name"], "created_at": _now().isoformat(),
                           "requests": 0, "pred": 0.0, "age_days": 0.0}
    _log(state, f"MULTIVERSE: '{g['name']}' masuk slot eksperimen.")


def add_custom_game(spec: dict) -> tuple[str, str]:
    """Daftarkan game custom yang SUDAH lolos validasi aturan kunci.

    Return (game_id, status) dengan status 'multiverse' atau 'antri'.
    """
    with _lock:
        state = load_state()
        gid = "c_" + uuid.uuid4().hex[:8]
        spec = {**spec, "id": gid, "custom": True, "status": "baru",
                "created_at": _now().isoformat(timespec="seconds")}
        spec.setdefault("keywords", [])
        spec.setdefault("min_players", 2)
        spec.setdefault("modes", ["offline", "online", "hybrid"])
        spec.setdefault("launch_week", 0)
        state["custom_games"][gid] = spec
        if state.get("multiverse") is None:
            _occupy(state, gid)
            status = "multiverse"
        else:
            state["queue"] = (state.get("queue", []) + [gid])[-3:]
            spec["status"] = "antri"
            status = "antri"
            _log(state, f"ANTRI: '{spec['name']}' menunggu slot multiverse.")
        save_state(state)
        _pred_cache.clear()
        return gid, status


def active_games(state: dict | None = None) -> list[dict]:
    state = state or load_state()
    if not state.get("top"):
        state = refresh_library()
    out = [get_game(gid, state["custom_games"]) for gid in state["top"]]
    if state.get("multiverse"):
        out.append(state["custom_games"][state["multiverse"]["id"]])
    return [g for g in out if g]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")


def all_games() -> list[dict]:
    from .catalog import GAMES
    return list(GAMES) + list(load_state()["custom_games"].values())


def find_game(game_id: str) -> dict | None:
    """Cari game dari id ATAU nama tampilan (LLM sering mengirim 'Caption Meme' bukan 'voting_meme')."""
    state = load_state()
    g = get_game(game_id, state["custom_games"])
    if g:
        return g
    key = _norm(game_id)
    if not key:
        return None
    games = all_games()
    for g in games:
        if key in (_norm(g["id"]), _norm(g["name"])):
            return g
    for g in games:
        if key in _norm(g["name"]) or _norm(g["name"]) in key:
            return g
    return None


def games_mentioned(text: str) -> list[dict]:
    """Semua game yang namanya disebut di teks (urut posisi kemunculan)."""
    t = _norm(text)
    hits = [(t.find(_norm(g["name"])), g) for g in all_games() if _norm(g["name"]) and _norm(g["name"]) in t]
    return [g for _, g in sorted(hits, key=lambda x: x[0])]


def game_mentioned(text: str) -> dict | None:
    """Game pertama (posisi paling awal) yang namanya disebut di teks."""
    t = _norm(text)
    hits = [(t.find(_norm(g["name"])), g) for g in all_games() if _norm(g["name"]) and _norm(g["name"]) in t]
    return min(hits, key=lambda x: x[0])[1] if hits else None


# ---------------------------------------------------------------------------
# rekomendasi berdasarkan parameter acara
# ---------------------------------------------------------------------------
# Prior pakar: kecocokan mekanik x jenis acara (lift, 1 = netral). Dipakai selama data in-app
# masih sedikit, lalu otomatis tergeser oleh data nyata (rata-rata berbobot jumlah request).
EVENT_PRIOR = {
    "quiz":      {"kajian": 1.3, "kelas/sekolah": 1.3, "kantor": 0.9, "keluarga": 0.9, "komunitas": 1.0,
                  "sosialisasi desa": 1.3, "seminar": 1.2},
    "truefalse": {"kajian": 1.1, "kelas/sekolah": 1.2, "kantor": 1.0, "keluarga": 1.0, "komunitas": 1.0,
                  "sosialisasi desa": 1.4, "seminar": 1.2},
    "majority":  {"kajian": 0.9, "kelas/sekolah": 0.8, "kantor": 1.2, "keluarga": 1.4, "komunitas": 1.2,
                  "sosialisasi desa": 0.8, "seminar": 0.8},
    "estimate":  {"kajian": 0.9, "kelas/sekolah": 1.0, "kantor": 1.3, "keluarga": 1.1, "komunitas": 1.0,
                  "sosialisasi desa": 1.1, "seminar": 1.0},
    "open_vote": {"kajian": 0.7, "kelas/sekolah": 1.1, "kantor": 1.3, "keluarga": 1.2, "komunitas": 1.4,
                  "sosialisasi desa": 0.7, "seminar": 0.7},
}
PRIOR_STRENGTH = 20


def _affinity(game: dict, col: str, value: str) -> float:
    prior = EVENT_PRIOR.get(game["mechanic"], {}).get(value, 1.0) if col == "event_type" else 1.0
    rt = load_runtime_requests()
    if not len(rt) or col not in rt:
        return prior
    base = (rt[col] == value).mean()
    sub = rt[rt["game_id"] == game["id"]]
    n = len(sub)
    if n == 0 or base == 0:
        return prior
    lift = (sub[col] == value).mean() / base
    return (prior * PRIOR_STRENGTH + lift * n) / (PRIOR_STRENGTH + n)


def recommend(params: dict, k: int = 3) -> list[dict]:
    state = load_state()
    games = active_games(state)
    pred = compute_scores().set_index("game_id")["score"]
    players = int(params.get("players") or 10)
    duration = int(params.get("duration_min") or 20)
    mode = params.get("mode") or "offline"
    res = []
    for g in games:
        reasons = []
        if not (g.get("min_players", 2) <= players <= g.get("max_players", 30)):
            continue
        if mode not in g.get("modes", ["offline", "online", "hybrid"]):
            continue
        demand = float(pred.get(g["id"], 0))
        fit = 1.0
        for col, key in [("event_type", "event_type"), ("location", "location"), ("topic", "topic"),
                         ("mode", "mode")]:
            v = params.get(key)
            if v:
                lift = _affinity(g, col, v)
                fit *= float(np.clip(lift, 0.6, 1.6)) ** (0.5 if col != "event_type" else 1.0)
                if col == "event_type" and lift > 1.1:
                    reasons.append(f"cocok untuk acara {v}")
        if g["mechanic"] == "open_vote" and duration < 15:
            fit *= 0.7
        if g["mechanic"] == "open_vote" and players > 20:
            fit *= 0.6
        if g.get("custom"):
            reasons.append("game baru di slot multiverse")
        if demand > 0.75:
            reasons.append("lagi rame minggu ini")
        score = 0.45 * demand + 0.55 * min(fit, 1.6) / 1.6
        res.append({"game_id": g["id"], "name": g["name"], "mechanic": g["mechanic"],
                    "score": round(score, 3), "trend_score": round(100 * demand, 1),
                    "reasons": reasons or ["cocok dengan parameter acara"]})
    return sorted(res, key=lambda r: -r["score"])[:k]


def rounds_for(duration_min: int, mechanic: str) -> tuple[int, int]:
    """Hitung (jumlah ronde, detik per ronde) dari durasi acara, di dalam batas aturan kunci."""
    sec = {"quiz": 20, "truefalse": 15, "majority": 20, "estimate": 25, "open_vote": 60}[mechanic]
    per_round = sec + (15 if mechanic != "open_vote" else 50)   # + waktu reveal/voting
    n = int(np.clip(duration_min * 60 * 0.7 // per_round, 3, 15))
    return n, sec


def seconds_until(ts: float) -> float:
    return max(0.0, ts - time.time())
