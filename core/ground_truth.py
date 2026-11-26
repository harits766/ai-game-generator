"""Satu dokumen GROUND TRUTH: sumber kebenaran isi library game.

Setiap update:
1. refresh library (prediksi model terbaru + request in-app)
2. bandingkan Top-10 baru dengan snapshot sebelumnya -> overlap, korelasi ranking, masuk/keluar
3. backtest: Top-10 prediksi model vs Top-10 aktual minggu terakhir yang sudah lengkap
4. cek integritas (jumlah game, slot multiverse, aturan kunci)
5. tulis ulang docs/GROUND_TRUTH.md + tambah riwayat di docs/ground_truth_history.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import library
from .catalog import get_game
from .config import (GROUND_TRUTH_HISTORY, GROUND_TRUTH_MD, KEY_RULES, MODEL_REPORT,
                     MULTIVERSE_GRACE_DAYS, MULTIVERSE_MIN_REQUESTS, MULTIVERSE_SLOTS, TOP_N,
                     TOP10_STABLE_OVERLAP)
from .features import FEATURES, TARGET, build_panel


def load_history() -> list[dict]:
    if GROUND_TRUTH_HISTORY.exists():
        try:
            return json.loads(GROUND_TRUTH_HISTORY.read_text())
        except Exception:
            pass
    return []


def backtest_last_week() -> dict:
    """Top-10 prediksi untuk minggu terakhir yang sudah punya data aktual."""
    reqs, sig = library._raw()
    panel = build_panel(reqs, sig)
    data = panel[panel["active"] & panel[TARGET].notna()]
    wk = data["week"].max()
    cur = data[data["week"] == wk].copy()
    bundle = library._model()
    cur["pred"] = bundle["model"].predict(cur[FEATURES])
    pred_top = set(cur.nlargest(TOP_N, "pred")["game_id"])
    act_top = set(cur.nlargest(TOP_N, TARGET)["game_id"])
    return {"week_predicted": str((wk + pd.Timedelta(weeks=1)).date()),
            "precision_at_10": len(pred_top & act_top) / TOP_N,
            "missed": sorted(act_top - pred_top), "false_in": sorted(pred_top - act_top)}


def compare(prev: list[str], new: list[str]) -> dict:
    inter = [g for g in new if g in prev]
    tau = None
    if len(inter) >= 3:
        tau, _ = kendalltau([prev.index(g) for g in inter], [new.index(g) for g in inter])
        tau = None if np.isnan(tau) else round(float(tau), 3)
    overlap = len(inter)
    if not prev:
        verdict = "BASELINE"
    elif overlap >= TOP10_STABLE_OVERLAP:
        verdict = "STABIL"
    else:
        verdict = "BERUBAH"
    return {"overlap": overlap, "kendall_tau": tau, "masuk": [g for g in new if g not in prev],
            "keluar": [g for g in prev if g not in new], "verdict": verdict}


def integrity(state: dict) -> list[dict]:
    checks = []
    top = state.get("top", [])
    checks.append({"cek": f"Library berisi tepat {TOP_N} game", "ok": len(top) == TOP_N})
    checks.append({"cek": "Tidak ada game ganda", "ok": len(set(top)) == len(top)})
    checks.append({"cek": "Semua game Top-10 terdaftar di katalog/custom",
                   "ok": all(get_game(g, state.get("custom_games")) for g in top)})
    checks.append({"cek": f"Slot multiverse maksimal {MULTIVERSE_SLOTS}",
                   "ok": state.get("multiverse") is None or isinstance(state["multiverse"], dict)})
    from .rules import validate
    customs_ok = all(validate(g)[0] for g in state.get("custom_games", {}).values()
                     if g.get("status") in ("multiverse", "promoted"))
    checks.append({"cek": "Game custom aktif lolos aturan kunci", "ok": customs_ok})
    return checks


def update_ground_truth() -> dict:
    hist = load_history()
    prev = hist[-1]["top"] if hist else []
    state = library.refresh_library()
    pred = library.predict_demand().set_index("game_id")
    cmp = compare(prev, state["top"])
    bt = backtest_last_week()
    checks = integrity(state)
    report = json.loads(MODEL_REPORT.read_text()) if MODEL_REPORT.exists() else {}
    snap = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_week": str(pred["week"].max().date()) if len(pred) else None,
        "top": state["top"],
        "pred": {g: round(float(pred.loc[g, "pred"]), 2) for g in state["top"] if g in pred.index},
        "multiverse": state.get("multiverse"),
        "compare": cmp, "backtest": bt,
        "integrity_ok": all(c["ok"] for c in checks),
        "model": report.get("best_model"),
    }
    hist.append(snap)
    GROUND_TRUTH_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_HISTORY.write_text(json.dumps(hist[-100:], indent=2, default=str))
    GROUND_TRUTH_MD.write_text(render_md(state, pred, snap, checks, report, hist))
    return snap


def render_md(state, pred, snap, checks, report, hist) -> str:
    cg = state.get("custom_games", {})
    L = ["# GROUND TRUTH: Library Game AI Game Generator", "",
         "> Dokumen ini di-generate otomatis oleh `core/ground_truth.py`. Jangan edit manual;",
         "> jalankan `python scripts/update_ground_truth.py` atau tombol **Update Ground Truth** di app.", "",
         f"- Diperbarui: **{snap['at']}**",
         f"- Minggu data terakhir: **{snap['data_week']}**",
         f"- Model prediksi: **{snap['model']}**",
         f"- Status Top-{TOP_N}: **{snap['compare']['verdict']}** "
         f"(overlap {snap['compare']['overlap']}/{TOP_N} dengan versi sebelumnya, "
         f"Kendall tau = {snap['compare']['kendall_tau']})",
         f"- Integritas: **{'LULUS' if snap['integrity_ok'] else 'GAGAL'}**", "",
         f"## 1. Library aktif (Top-{TOP_N})", "",
         "| # | Game | Mekanik | Prediksi request minggu depan | Request minggu ini | Google Trends | TikTok |",
         "|---|---|---|---|---|---|---|"]
    for i, gid in enumerate(state["top"], 1):
        g = get_game(gid, cg) or {"name": gid, "mechanic": "?"}
        r = pred.loc[gid] if gid in pred.index else None
        L.append(f"| {i} | {g['name']} (`{gid}`) | {g['mechanic']} | "
                 f"{r['pred']:.1f} | {int(r['req_t'])} | {r['gtrend_t']:.0f} | {int(r['tiktok_t'])} |"
                 if r is not None else f"| {i} | {g['name']} | {g['mechanic']} | - | - | - | - |")
    mv = state.get("multiverse")
    L += ["", "## 2. Slot multiverse (eksperimen)", "",
          f"Aturan: game baru masuk slot ini. Dipromosikan ke Top-{TOP_N} jika prediksinya >= game "
          f"peringkat {TOP_N} dan sudah >= {MULTIVERSE_MIN_REQUESTS} request; dihapus jika setelah "
          f"{MULTIVERSE_GRACE_DAYS} hari request < {MULTIVERSE_MIN_REQUESTS}.", ""]
    if mv:
        L.append(f"- **{mv['name']}** (`{mv['id']}`): {mv.get('requests', 0)} request, "
                 f"prediksi {mv.get('pred', 0)}, umur {mv.get('age_days', 0)} hari")
    else:
        L.append("- (kosong)")
    if state.get("queue"):
        L.append(f"- Antrian: {', '.join(cg[q]['name'] for q in state['queue'] if q in cg)}")
    L += ["", "## 3. Validasi", "",
          f"**a. Perubahan vs snapshot sebelumnya:** masuk = {snap['compare']['masuk'] or '-'}, "
          f"keluar = {snap['compare']['keluar'] or '-'}.", "",
          f"**b. Backtest model:** Top-{TOP_N} prediksi untuk minggu {snap['backtest']['week_predicted']} "
          f"cocok {snap['backtest']['precision_at_10']*100:.0f}% dengan Top-{TOP_N} aktual "
          f"(terlewat: {snap['backtest']['missed'] or '-'}; salah masuk: {snap['backtest']['false_in'] or '-'}).", "",
          "**c. Cek integritas:**", ""]
    L += [f"- [{'x' if c['ok'] else ' '}] {c['cek']}" for c in checks]
    L += ["", "## 4. Aturan kunci game custom", ""]
    L += [f"- Jumlah aturan main: {KEY_RULES['min_rules']}-{KEY_RULES['max_rules']}",
          f"- Peran wajib: {', '.join(KEY_RULES['required_roles'])}",
          f"- Pemain: {KEY_RULES['min_players']}-{KEY_RULES['max_players']}",
          f"- Ronde: {KEY_RULES['min_rounds']}-{KEY_RULES['max_rounds']}, "
          f"{KEY_RULES['min_seconds_per_round']}-{KEY_RULES['max_seconds_per_round']} detik/ronde",
          f"- Total sesi maks {KEY_RULES['max_session_minutes']} menit",
          "- Mekanik harus salah satu: quiz, truefalse, majority, estimate, open_vote"]
    if report.get("results"):
        L += ["", "## 5. Performa model (test 26 minggu terakhir)", "",
              "| Model | MAE | RMSE | R2 | Precision@10 |", "|---|---|---|---|---|"]
        for r in report["results"]:
            L.append(f"| {r['model']} | {r['MAE']} | {r['RMSE']} | {r['R2']} | {r['Precision@10']} |")
    L += ["", "## 6. Riwayat update", "", "| Waktu | Verdict | Overlap | Backtest P@10 | Multiverse |",
          "|---|---|---|---|---|"]
    for h in hist[-10:][::-1]:
        L.append(f"| {h['at']} | {h['compare']['verdict']} | {h['compare']['overlap']} | "
                 f"{h['backtest']['precision_at_10']:.1f} | {(h.get('multiverse') or {}).get('name', '-')} |")
    if state.get("archive"):
        L += ["", "## 7. Arsip (game yang keluar/dihapus)", ""]
        for a in state["archive"][-10:]:
            name = (get_game(a["id"], cg) or {"name": a["id"]})["name"]
            L.append(f"- {a['at']}: {name} ({a['reason']})")
    return "\n".join(L) + "\n"
