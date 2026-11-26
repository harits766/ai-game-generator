"""Halaman studio (untuk pembuat acara): chat agent, buat room manual, game custom,
tren & library, ground truth, data & model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import library
from core.agent import AgentContext, chat, create_room_service
from core.catalog import GAMES, MECHANICS
from core.config import (DURATIONS, EVENT_TYPES, GROUND_TRUTH_MD, KEY_RULES, LOCATIONS, MODEL_REPORT,
                         MODES, TOP_N)
from core.ground_truth import load_history, update_ground_truth
from core.agent import design_game
from core.rules import TEMPLATE, validate

from .common import base_url, qr_png

PALETTE = ["#6D28D9", "#DB2777", "#2563EB", "#D97706", "#059669", "#0891B2", "#9333EA", "#E11D48"]


def _ctx() -> AgentContext:
    return AgentContext(base_url=base_url(), doc=st.session_state.get("doc"),
                        use_llm=st.session_state.get("use_llm", True),
                        ai_log=st.session_state.setdefault("ai_log", []))


def hero(title: str, sub: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def room_card(info: dict, key: str):
    if "error" in info:
        st.error(info["error"])
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(qr_png(info["player_url"]), caption="QR peserta", width=170)
    with c2:
        st.markdown(f"**Room {info['room_code']}** · {info['game']} · {info['rounds']} ronde · "
                    f"{info['seconds_per_round']} dtk/ronde")
        st.caption(f"Konten: {info.get('content_source')} · {info.get('provider')}")
        st.link_button("🎛️ Buka sebagai HOST", info["host_url"], type="primary", use_container_width=True)
        st.code(info["player_url"], language=None)
        st.caption("Bagikan link di atas ke peserta (atau suruh scan QR).")


# ---------------------------------------------------------------------------
def page_chat():
    hero("🤖 Game Master AI",
         "Ceritain acaramu (jumlah orang, durasi, jenis acara, online/offline, lokasi, topik). "
         "AI pilih game yang pas dari tren + dokumenmu, lalu langsung kasih link room.")
    st.session_state.setdefault("chat", [])
    examples = ["Kajian 15 orang di Bogor, offline 30 menit, topik literasi digital",
                "Game apa yang lagi rame minggu ini?",
                "Gathering kantor online 25 orang, 20 menit, bikin yang receh",
                "Sosialisasi desa 20 warga, 30 menit, materi dari dokumen yang aku upload"]
    if not st.session_state["chat"]:
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state["pending"] = ex
                st.rerun()
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                _trace(m["trace"], m.get("engine", ""))
    prompt = st.chat_input("Contoh: arisan keluarga 12 orang, 20 menit, topik sejarah kemerdekaan")
    prompt = prompt or st.session_state.pop("pending", None)
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Mikir + manggil tools..."):
                hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat"]]
                answer, trace, engine = chat(_ctx(), hist)
            st.markdown(answer)
            _trace(trace, engine)
        st.session_state["chat"].append({"role": "assistant", "content": answer, "trace": trace,
                                         "engine": engine})
    if st.session_state["chat"] and st.button("Mulai chat baru"):
        st.session_state["chat"] = []
        st.rerun()


def _trace(trace: list[dict], engine: str):
    rooms = [t["result"] for t in trace if t["tool"] == "create_room" and "room_code" in t["result"]]
    for i, r in enumerate(rooms):
        room_card(r, key=f"rc_{r['room_code']}_{i}")
    with st.expander(f"🔧 Jejak agent · {engine} · {len(trace)} tool call"):
        for t in trace:
            st.markdown(f"**{t['tool']}**({json.dumps(t['args'], ensure_ascii=False)[:200]})")
            st.json(t["result"], expanded=False)


# ---------------------------------------------------------------------------
def page_manual():
    hero("🛠️ Buat Room Manual", "Isi parameter acara, lihat rekomendasi model, pilih game, dapat link host.")
    with st.form("params"):
        c1, c2, c3 = st.columns(3)
        players = c1.number_input("Jumlah pemain", 2, KEY_RULES["max_players"], 12)
        duration = c2.selectbox("Durasi main (menit)", DURATIONS, index=3)
        event = c3.selectbox("Jenis acara", EVENT_TYPES)
        c4, c5, c6 = st.columns(3)
        mode = c4.selectbox("Mode", MODES)
        loc = c5.selectbox("Lokasi", ["(tidak diisi)"] + LOCATIONS)
        topic = c6.text_input("Topik", placeholder="mis. pemilahan sampah")
        go = st.form_submit_button("Cari game yang cocok", type="primary", use_container_width=True)
    if go:
        st.session_state["manual_params"] = {
            "players": players, "duration_min": duration, "event_type": event, "mode": mode,
            "location": "" if loc.startswith("(") else loc, "topic": topic}
    params = st.session_state.get("manual_params")
    if not params:
        return
    recs = library.recommend(params, k=5)
    if not recs:
        st.warning("Tidak ada game yang cocok. Ubah jumlah pemain/mode.")
        return
    st.markdown("#### Rekomendasi")
    for r in recs:
        st.markdown(f'<div class="card"><h4>{r["name"]} <span class="pill">{r["mechanic"]}</span>'
                    f'<span class="pill">skor {r["score"]:.2f}</span></h4>'
                    f'<span class="muted">Skor tren {r["trend_score"]} · '
                    f'{"; ".join(r["reasons"])}</span></div>', unsafe_allow_html=True)
    choice = st.radio("Pilih game", [r["game_id"] for r in recs],
                      format_func=lambda g: next(r["name"] for r in recs if r["game_id"] == g), horizontal=True)
    if not st.session_state.get("doc") and not params.get("topic"):
        st.caption("Tip: upload dokumen di sidebar atau isi topik supaya kontennya relevan.")
    if st.button("🚀 Buat room", type="primary", use_container_width=True):
        with st.spinner("Membuat konten game..."):
            info = create_room_service(_ctx(), choice, params)
        st.session_state["manual_room"] = info
    if st.session_state.get("manual_room"):
        room_card(st.session_state["manual_room"], "manual")


# ---------------------------------------------------------------------------
def page_custom():
    hero("🧪 Game Custom (Multiverse)",
         f"Bikin game sendiri selama lolos aturan kunci. Game baru masuk slot multiverse; kalau rame "
         f"dipromosikan ke Top-{TOP_N}, kalau sepi dihapus.")
    with st.expander("📏 Aturan kunci", expanded=False):
        st.markdown(
            f"- {KEY_RULES['min_rules']}-{KEY_RULES['max_rules']} aturan main, maks {KEY_RULES['max_rule_chars']} "
            f"karakter/aturan\n- Wajib ada peran **host** dan **peserta**\n"
            f"- Mekanik: {', '.join(MECHANICS)}\n- Pemain {KEY_RULES['min_players']}-{KEY_RULES['max_players']}, "
            f"ronde {KEY_RULES['min_rounds']}-{KEY_RULES['max_rounds']}, "
            f"{KEY_RULES['min_seconds_per_round']}-{KEY_RULES['max_seconds_per_round']} dtk/ronde, "
            f"total maks {KEY_RULES['max_session_minutes']} menit")
    idea = st.text_input("Ide game (opsional, biar AI bikinin draft)",
                         placeholder="mis. tebak-tebakan ayat/hadis tentang sabar, versi cepat")
    if st.button("✨ Draft pakai AI") and idea:
        with st.spinner("Chain desainer bekerja..."):
            spec, prov = design_game(idea, st.session_state.get("use_llm", True))
        st.session_state["draft"] = spec
        st.toast(f"Draft dari {prov}")
    d = {**TEMPLATE, **st.session_state.get("draft", {})}
    with st.form("custom"):
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("Nama game", d.get("name", ""))
        mech = c2.selectbox("Mekanik", list(MECHANICS), index=list(MECHANICS).index(d.get("mechanic", "quiz"))
                            if d.get("mechanic") in MECHANICS else 0, format_func=lambda m: f"{m}: {MECHANICS[m]}")
        desc = st.text_input("Deskripsi", d.get("desc", ""))
        roles = d.get("roles") or {}
        c3, c4 = st.columns(2)
        host = c3.text_area("Peran HOST", roles.get("host", ""), height=80)
        peserta = c4.text_area("Peran PESERTA", roles.get("peserta", roles.get("participant", "")), height=80)
        rules = st.text_area("Aturan main (1 baris = 1 aturan)", "\n".join(d.get("rules", [])), height=140)
        c5, c6, c7 = st.columns(3)
        maxp = c5.number_input("Maks pemain", 1, 100, int(d.get("max_players", 20)))
        rounds = c6.number_input("Jumlah ronde", 1, 50, int(d.get("rounds", 8)))
        sec = c7.number_input("Detik per ronde", 5, 300, int(d.get("seconds_per_round", 20)))
        submitted = st.form_submit_button("Validasi & daftarkan ke multiverse", type="primary",
                                          use_container_width=True)
    if submitted:
        spec = {"name": name, "mechanic": mech, "desc": desc, "roles": {"host": host, "peserta": peserta},
                "rules": rules.splitlines(), "max_players": maxp, "rounds": rounds, "seconds_per_round": sec}
        ok, errs, clean = validate(spec)
        if not ok:
            st.error("Belum lolos aturan kunci:\n\n" + "\n".join(f"- {e}" for e in errs))
        else:
            gid, status = library.add_custom_game(clean)
            st.session_state["custom_gid"] = gid
            st.success(f"Lolos! **{clean['name']}** "
                       + ("masuk slot multiverse." if status == "multiverse"
                          else "masuk antrian (slot multiverse sedang terisi), tapi tetap bisa dimainkan."))
    gid = st.session_state.get("custom_gid")
    if gid:
        if st.button("🚀 Buat room untuk game ini", use_container_width=True):
            with st.spinner("Membuat konten..."):
                st.session_state["custom_room"] = create_room_service(
                    _ctx(), gid, {"players": 10, "duration_min": 20, "topic": idea})
        if st.session_state.get("custom_room"):
            room_card(st.session_state["custom_room"], "custom")


# ---------------------------------------------------------------------------
def page_trends():
    hero("📈 Tren & Library", f"Top-{TOP_N} game aktif. Skor = tren real (Wikipedia + Google Trends Indonesia, "
                              "diprediksi model) + daya tarik (model BoardGameGeek) + request in-app.")
    state = library.load_state()
    if not state.get("top"):
        state = library.refresh_library()
    if st.button("🔄 Refresh library sekarang"):
        state = library.refresh_library()
        st.toast("Library diperbarui")
    sc = library.compute_scores()
    report = json.loads(MODEL_REPORT.read_text()) if MODEL_REPORT.exists() else {}
    w = float(sc["w_req"].iloc[0]) if len(sc) else 0.0
    c = st.columns(4)
    c[0].metric("Game aktif", f"{len(state['top'])} + {1 if state.get('multiverse') else 0}")
    c[1].metric("Minggu data tren", str(pd.to_datetime(sc["week"]).max().date()))
    c[2].metric("Request in-app (28 hari)", int(sc["req_28d"].sum()))
    c[3].metric("Bobot request in-app", f"{w:.0%}")
    st.caption(f"Model tren: {report.get('best_model', '-')}. Bobot request in-app mulai dari 0% dan naik "
               "otomatis seiring data pemakaian nyata bertambah (maks 70%).")

    names = {g["id"]: g["name"] for g in GAMES} | {k: v["name"] for k, v in state["custom_games"].items()}
    top = sc[sc["game_id"].isin(state["top"])].copy()
    top["name"] = top["game_id"].map(names)
    top["Tren real"] = (1 - w) * 0.65 * top["trend_part"].fillna(0) * 100
    top["Daya tarik BGG"] = (1 - w) * np.where(top["custom"], 1.0, 0.35) * top["appeal_part"] * 100
    top["Request in-app"] = w * top["req_part"] * 100
    long = top.melt(id_vars="name", value_vars=["Tren real", "Daya tarik BGG", "Request in-app"],
                    var_name="komponen", value_name="poin")
    order = top.sort_values("score")["name"].tolist()
    fig = px.bar(long, x="poin", y="name", color="komponen", orientation="h", height=380,
                 category_orders={"name": order[::-1]},
                 color_discrete_sequence=["#6D28D9", "#DB2777", "#059669"],
                 labels={"poin": "Skor (0-100)", "name": ""})
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    mv = state.get("multiverse")
    st.markdown("#### 🌌 Slot multiverse")
    if mv:
        st.markdown(f'<div class="card"><h4>{mv["name"]}</h4><span class="muted">{mv.get("requests", 0)} request · '
                    f'skor {mv.get("pred", 0)} · umur {mv.get("age_days", 0)} hari</span></div>',
                    unsafe_allow_html=True)
    else:
        st.caption("Kosong. Bikin game di halaman Game Custom.")
    if state.get("events"):
        with st.expander("Log perubahan library"):
            for e in state["events"][::-1][:20]:
                st.markdown(f"- `{e['at']}` {e['msg']}")

    st.markdown("#### Sinyal tren real per game")
    from core.trend_model import load_signals
    sig = load_signals()
    sig = sig[sig["game_id"].isin(names)]
    pick = st.multiselect("Game", list(names), default=state["top"][:3], format_func=lambda g: names.get(g, g))
    metric = st.radio("Sinyal", ["wiki_views", "google_trends"], horizontal=True,
                      format_func={"wiki_views": "Wikipedia views/minggu",
                                   "google_trends": "Google Trends Indonesia (0-100)"}.get)
    s = sig[sig["game_id"].isin(pick)].assign(game=lambda d: d["game_id"].map(names))
    fig2 = px.line(s, x="week", y=metric, color="game", color_discrete_sequence=PALETTE, height=320,
                   log_y=metric == "wiki_views")
    fig2.update_layout(margin=dict(l=0, r=10, t=10, b=0), legend_title=None, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Sumber: Wikimedia Pageviews API, Google Trends (pytrends), diperbarui otomatis tiap Senin "
               "oleh GitHub Actions.")


# ---------------------------------------------------------------------------
def page_ground_truth():
    hero("✅ Ground Truth", f"Satu dokumen acuan isi library. Tiap update dicek: apakah Top-{TOP_N} masih sama?")
    if st.button("🔁 Update Ground Truth", type="primary"):
        with st.spinner("Refresh library, backtest, cek integritas..."):
            snap = update_ground_truth()
        c = snap["compare"]
        (st.success if snap["integrity_ok"] else st.error)(
            f"Verdict: **{c['verdict']}** · overlap {c['overlap']}/{TOP_N} · "
            f"backtest P@10 {snap['backtest']['precision_at_10']:.0%}")
    hist = load_history()
    if len(hist) > 1:
        h = pd.DataFrame([{"waktu": x["at"], "overlap": x["compare"]["overlap"],
                           "backtest_p10": x["backtest"]["precision_at_10"] * TOP_N} for x in hist])
        fig = px.line(h, x="waktu", y=["overlap", "backtest_p10"], markers=True, height=260,
                      color_discrete_sequence=PALETTE, labels={"value": f"dari {TOP_N}", "variable": ""})
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    if GROUND_TRUTH_MD.exists():
        md = GROUND_TRUTH_MD.read_text()
        st.download_button("⬇️ Unduh GROUND_TRUTH.md", md.encode(), "GROUND_TRUTH.md")
        st.markdown(md)


# ---------------------------------------------------------------------------
def page_model():
    hero("🧠 Data & Model", "Dua model ML dari data real + komponen GenAI.")
    from core.config import MODELS_DIR
    rep = json.loads(MODEL_REPORT.read_text()) if MODEL_REPORT.exists() else {}
    ap_path = MODELS_DIR / "appeal_report.json"
    ap = json.loads(ap_path.read_text()) if ap_path.exists() else {}
    t1, t2, t3 = st.tabs(["📈 Model tren", "🎲 Model daya tarik (BGG)", "🔌 Sumber data & alur"])
    with t1:
        if rep:
            st.markdown(f"**Tugas:** {rep['task']}  \n**Model terpilih:** {rep['best_model']} "
                        f"({rep['selection_metric']}) · {rep['n_series']} seri · data {rep['data_weeks'][0]} s/d "
                        f"{rep['data_weeks'][1]} · cakupan Google Trends {rep['google_trends_coverage']:.0%}")
            st.dataframe(pd.DataFrame(rep["results"]), hide_index=True, use_container_width=True)
            st.caption("Data web mingguan mirip random walk, jadi baseline naive sudah kuat. Model dipilih kalau "
                       "mengalahkan baseline pada MAE dan menebak arah naik/turun lebih baik dari 50%.")
            _importance(rep["feature_importance"])
    with t2:
        if ap:
            st.markdown(f"**Tugas:** klasifikasi game party populer (25% teratas jumlah rating) dari profilnya. "
                        f"Data real: {ap['n_games']:,} game BoardGameGeek (2000-2021). **Model terpilih:** "
                        f"{ap['best_model']} ({ap['selection_metric']}).")
            st.dataframe(pd.DataFrame(ap["results"]), hide_index=True, use_container_width=True)
            st.caption("Dipakai untuk cold start: game baru/custom yang belum punya riwayat dapat skor awal dari sini.")
            _importance(ap["feature_importance"])
    with t3:
        st.markdown("""
| Sumber | Jenis | Dipakai untuk |
|---|---|---|
| Wikipedia Pageviews API | real, mingguan, gratis | minat publik per jenis game (fitur + target model tren) |
| Google Trends Indonesia (pytrends) | real, mingguan | momentum minat di Indonesia |
| BoardGameGeek (TidyTuesday 2022) | real, 4.911 game party | model daya tarik (cold start game baru) |
| Log request in-app | real, mulai dari 0 | bobot ranking naik seiring data bertambah, afinitas acara |
| TikTok / X | tidak dipakai | tidak ada API gratis yang stabil |

**Alur sistem**
1. Input: dokumen (PDF/DOCX/TXT/MD) atau topik + parameter acara.
2. Model tren + model daya tarik + request in-app -> skor -> Top-10 + 1 slot multiverse.
3. Agent LangChain (Gemini -> OpenAI -> offline) memilih game lewat 7 tools; chain konten membuat soal dari
   potongan dokumen (retrieval TF-IDF), output dipaksa ke skema Pydantic.
4. Konten di-cache (TTL 30 menit); room idle > 2 jam dihapus.
5. Room dibuat: link HOST + link PESERTA. Tiap room = 1 request in-app, disimpan permanen di branch `data` GitHub.
6. GitHub Actions tiap Senin: tarik data tren, latih ulang, regenerate ground truth (cek Top-10 masih sama).
""")
        from core.storage import status as storage_status
        stt = storage_status()
        st.caption(f"Penyimpanan log: {stt['backend']} · sinkron terakhir: {stt['last_sync'] or '-'}"
                   + (f" · error: {stt['last_error']}" if stt.get("last_error") else ""))


def _importance(fi: dict):
    s = pd.Series(fi).head(10).sort_values()
    fig = px.bar(s, orientation="h", labels={"value": "importance", "index": ""},
                 color_discrete_sequence=["#DB2777"], height=320)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
