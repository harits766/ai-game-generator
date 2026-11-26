"""Halaman studio (untuk pembuat acara): chat agent, buat room manual, game custom,
tren & library, ground truth, data & model."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from core import library
from core.agent import AgentContext, chat, create_room_service
from core.catalog import GAMES, MECHANICS
from core.config import (DURATIONS, EVENT_TYPES, GROUND_TRUTH_MD, KEY_RULES, LOCATIONS, MODEL_REPORT,
                         MODES, TOP_N)
from core.ground_truth import load_history, update_ground_truth
from core.llm import generate_json
from core.rules import TEMPLATE, validate

from .common import base_url, qr_png

PALETTE = ["#6D28D9", "#DB2777", "#2563EB", "#D97706", "#059669", "#0891B2", "#9333EA", "#E11D48"]


def _ctx() -> AgentContext:
    return AgentContext(base_url=base_url(), doc=st.session_state.get("doc"),
                        use_llm=st.session_state.get("use_llm", True))


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
                    f'<span class="muted">Prediksi {r["pred_requests"]} request minggu depan · '
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
        prompt = (f"Buat spesifikasi game party dari ide: {idea}. Format JSON persis seperti contoh: "
                  f"{json.dumps(TEMPLATE, ensure_ascii=False)}. mechanic salah satu {list(MECHANICS)}. "
                  f"rules {KEY_RULES['min_rules']}-{KEY_RULES['max_rules']} kalimat pendek. "
                  f"roles wajib punya host dan peserta.")
        data, prov = generate_json(prompt, "Kamu desainer game party. Jawab JSON saja.") \
            if st.session_state.get("use_llm", True) else (None, "offline")
        if data:
            st.session_state["draft"] = data
            st.toast(f"Draft dari {prov}")
        else:
            st.session_state["draft"] = {**TEMPLATE, "name": idea[:40].title(),
                                         "desc": idea[:200],
                                         "rules": ["Host membacakan soal dari materi.",
                                                   "Peserta menjawab lewat HP sebelum waktu habis.",
                                                   "Jawaban benar dan cepat dapat poin lebih besar.",
                                                   "Skor tertinggi di akhir ronde jadi pemenang."]}
            st.toast("LLM tidak tersedia, pakai template.")
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
    hero("📈 Tren & Library", f"Top-{TOP_N} game aktif dipilih oleh model prediksi dari sinyal Google Trends, "
                              "TikTok, X, dan request di aplikasi.")
    state = library.load_state()
    if not state.get("top"):
        state = library.refresh_library()
    if st.button("🔄 Refresh library sekarang"):
        state = library.refresh_library()
        st.toast("Library diperbarui")
    pred = library.predict_demand()
    report = json.loads(MODEL_REPORT.read_text()) if MODEL_REPORT.exists() else {}
    c = st.columns(4)
    c[0].metric("Game aktif", f"{len(state['top'])} + {1 if state.get('multiverse') else 0}")
    c[1].metric("Model", report.get("best_model", "-"))
    c[2].metric("Minggu data", str(pred["week"].max().date()))
    c[3].metric("Request in-app", sum(library.runtime_counts().values()))

    names = {g["id"]: g["name"] for g in GAMES} | {k: v["name"] for k, v in state["custom_games"].items()}
    top = pred[pred["game_id"].isin(state["top"])].assign(name=lambda d: d["game_id"].map(names))
    fig = px.bar(top.sort_values("pred"), x="pred", y="name", orientation="h",
                 labels={"pred": "Prediksi request minggu depan", "name": ""},
                 color_discrete_sequence=["#6D28D9"], height=360)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    mv = state.get("multiverse")
    st.markdown("#### 🌌 Slot multiverse")
    if mv:
        st.markdown(f'<div class="card"><h4>{mv["name"]}</h4><span class="muted">{mv.get("requests", 0)} request · '
                    f'prediksi {mv.get("pred", 0)} · umur {mv.get("age_days", 0)} hari</span></div>',
                    unsafe_allow_html=True)
    else:
        st.caption("Kosong. Bikin game di halaman Game Custom.")
    if state.get("events"):
        with st.expander("Log perubahan library"):
            for e in state["events"][::-1][:20]:
                st.markdown(f"- `{e['at']}` {e['msg']}")

    st.markdown("#### Sinyal tren per game")
    from core.features import load_raw
    _, sig = load_raw()
    pick = st.multiselect("Game", list(names), default=state["top"][:3], format_func=lambda g: names.get(g, g))
    metric = st.radio("Sinyal", ["google_trends", "tiktok_mentions", "x_mentions"], horizontal=True)
    s = sig[sig["game_id"].isin(pick)].assign(game=lambda d: d["game_id"].map(names))
    fig2 = px.line(s, x="week", y=metric, color="game", color_discrete_sequence=PALETTE, height=320)
    fig2.update_layout(margin=dict(l=0, r=10, t=10, b=0), legend_title=None, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)


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
    hero("🧠 Data & Model", "Ringkasan data, eksperimen model, dan cara kerja sistem.")
    report = json.loads(MODEL_REPORT.read_text()) if MODEL_REPORT.exists() else {}
    if report:
        st.markdown(f"**Model terpilih:** {report['best_model']} (metrik seleksi: {report['selection_metric']})")
        res = pd.DataFrame(report["results"])
        st.dataframe(res, hide_index=True, use_container_width=True)
        fi = pd.Series(report["feature_importance"]).head(10).sort_values()
        fig = px.bar(fi, orientation="h", labels={"value": "importance", "index": ""},
                     color_discrete_sequence=["#DB2777"], height=320)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    from core.features import clean_requests, load_raw
    reqs, _ = load_raw()
    df = clean_requests(reqs)
    st.markdown(f"**Dataset request:** {len(reqs):,} baris mentah -> {len(df):,} setelah cleaning · "
                f"{df['game_id'].nunique()} game · {df['week'].nunique()} minggu")
    c1, c2 = st.columns(2)
    for col, box in [("event_type", c1), ("doc_type", c2)]:
        vc = df[col].value_counts().reset_index()
        f = px.bar(vc, x="count", y=col, orientation="h", color_discrete_sequence=["#6D28D9"], height=260)
        f.update_layout(margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", yaxis_title=None)
        box.plotly_chart(f, use_container_width=True)
    wk = df.groupby("week").size().reset_index(name="request")
    f = px.area(wk, x="week", y="request", color_discrete_sequence=["#6D28D9"], height=240)
    f.update_layout(margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f, use_container_width=True)
    st.markdown("""
**Alur sistem**
1. Input: dokumen (PDF/DOCX/TXT/MD) atau topik + parameter acara.
2. Model prediksi (ML) memberi skor permintaan tiap game dari sinyal tren -> Top-10 + slot multiverse.
3. Agent GenAI memilih game lewat tools, lalu generator membuat konten dari potongan dokumen (retrieval TF-IDF).
4. Konten di-cache (TTL 30 menit) per dokumen+game+parameter; room idle > 2 jam dihapus.
5. Room dibuat: link HOST + link PESERTA. Setiap room tercatat sebagai request baru -> data untuk prediksi berikutnya.
6. Ground truth di-regenerate dan memvalidasi apakah Top-10 masih sama.
""")
