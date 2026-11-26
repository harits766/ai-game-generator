"""Library game: Top-N berdasarkan prediksi model + 1 slot "multiverse" untuk game baru.

Alur:
1. predict_demand()   -> prediksi request minggu depan untuk semua game (pool + custom)
2. refresh_library()  -> ambil Top-10, nilai game multiverse (promosi / hapus), simpan state
3. recommend()        -> ranking game untuk parameter acara user (demand x kecocokan)
"""
from __future__ import annotations

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
from .features import build_panel, clean_requests, load_raw, load_runtime_requests

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


def runtime_counts() -> dict:
    rt = load_runtime_requests()
    return rt["game_id"].value_counts().to_dict() if len(rt) else {}


# ---------------------------------------------------------------------------
# prediksi
# ---------------------------------------------------------------------------
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
def _raw():
    reqs, sig = load_raw()
    return clean_requests(reqs), sig


def predict_demand() -> pd.DataFrame:
    """Prediksi request minggu depan untuk setiap game (pool + custom)."""
    key = (RUNTIME_REQUESTS.stat().st_mtime if RUNTIME_REQUESTS.exists() else 0,
           LIBRARY_STATE.stat().st_mtime if LIBRARY_STATE.exists() else 0)
    if key in _pred_cache:
        return _pred_cache[key]
    reqs, sig = _raw()
    rt = load_runtime_requests()
    if len(rt):
        rt = clean_requests(rt.assign(players=pd.to_numeric(rt["players"], errors="coerce").fillna(10)))
        # Data historis berhenti di minggu terakhir dataset. Request in-app dihitung
        # sebagai bagian "minggu berjalan" itu, supaya panel tidak bolong saat app dipakai
        # setelah tanggal dataset (jam simulasi).
        rt["week"] = reqs["week"].max()
        reqs = pd.concat([reqs, rt[reqs.columns.intersection(rt.columns)]], ignore_index=True)
    customs = list(load_state()["custom_games"].values())
    panel = build_panel(reqs, sig, extra_games=customs)
    last_week = panel["week"].max()
    cur = panel[panel["week"] == last_week].copy()
    bundle = _model()
    cur["pred"] = np.clip(bundle["model"].predict(cur[bundle["features"]]), 0, None)
    # game yang belum launch (belum ada request sama sekali) tidak masuk ranking
    cur.loc[cur["weeks_since_launch"] < 0, "pred"] = 0.0
    out = cur[["game_id", "week", "req_t", "roll4_mean", "gtrend_t", "tiktok_t", "x_t", "pred"]]
    out = out.sort_values("pred", ascending=False).reset_index(drop=True)
    _pred_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# library refresh + multiverse
# ---------------------------------------------------------------------------
def refresh_library() -> dict:
    with _lock:
        state = load_state()
        pred = predict_demand().set_index("game_id")["pred"]
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
            mv.update({"requests": n_req, "pred": round(p, 2), "age_days": round(age_days, 2)})
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


def find_game(game_id: str) -> dict | None:
    return get_game(game_id, load_state()["custom_games"])


# ---------------------------------------------------------------------------
# rekomendasi berdasarkan parameter acara
# ---------------------------------------------------------------------------
@cached(TTLCache(maxsize=1, ttl=3600))
def _affinity() -> dict:
    """Lift P(nilai | game) / P(nilai) untuk event_type, mode, location, topic dari data log."""
    reqs, _ = _raw()
    out = {}
    for col in ["event_type", "mode", "location", "topic"]:
        base = reqs[col].value_counts(normalize=True)
        cond = reqs.groupby("game_id")[col].value_counts(normalize=True)
        out[col] = (cond / cond.index.get_level_values(1).map(base).values).to_dict()
    return out


def recommend(params: dict, k: int = 3) -> list[dict]:
    state = load_state()
    games = active_games(state)
    pred = predict_demand().set_index("game_id")["pred"]
    pmax = max(float(pred.max()), 1.0)
    aff = _affinity()
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
        # skala log agar 1 game viral tidak mendominasi semua rekomendasi
        demand = float(np.log1p(pred.get(g["id"], 0)) / np.log1p(pmax))
        fit = 1.0
        for col, key in [("event_type", "event_type"), ("location", "location"), ("topic", "topic"),
                         ("mode", "mode")]:
            v = params.get(key)
            if v:
                lift = aff[col].get((g["id"], v), 1.0)
                fit *= float(np.clip(lift, 0.6, 1.6)) ** (0.5 if col != "event_type" else 1.0)
                if col == "event_type" and lift > 1.1:
                    reasons.append(f"sering dipakai di acara {v}")
        if g["mechanic"] == "open_vote" and duration < 15:
            fit *= 0.7
        if g["mechanic"] == "open_vote" and players > 20:
            fit *= 0.6
        if g.get("custom"):
            reasons.append("game baru di slot multiverse")
        if demand > 0.85:
            reasons.append("lagi rame minggu ini")
        score = 0.45 * demand + 0.55 * min(fit, 1.6) / 1.6
        res.append({"game_id": g["id"], "name": g["name"], "mechanic": g["mechanic"],
                    "score": round(score, 3), "pred_requests": round(float(pred.get(g["id"], 0)), 1),
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
