"""Tampilan room: HOST (laptop/HP host) dan PESERTA (HP masing-masing).

Sinkronisasi memakai st.fragment(run_every=1s): tiap HP membaca state room dari
memori server tiap detik; hanya fragment ini yang dirender ulang, bukan seluruh halaman.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from core.agent import room_links
from core.config import RUNTIME_DIR
from core.engine import STORE, Room

from .common import OPTION_ICONS, base_url, qr_png

POLL = 1.0


def render_room():
    qp = st.query_params
    code = (qp.get("room") or "").upper()
    room = STORE.get(code)
    if not room:
        st.markdown('<div class="hero"><h1>Room tidak ditemukan</h1>'
                    '<p>Kode salah, atau room sudah ditutup (room idle lebih dari 2 jam otomatis dihapus).</p></div>',
                    unsafe_allow_html=True)
        st.link_button("Ke halaman utama", base_url() + "/")
        return
    if qp.get("role") == "host":
        if qp.get("key") != room.host_key:
            st.error("Link host tidak valid. Pakai link host yang didapat saat membuat room.")
            return
        host_view(room)
    else:
        player_view(room)


# ---------------------------------------------------------------------------
# HOST
# ---------------------------------------------------------------------------
def host_view(room: Room):
    _, player_url = room_links(base_url(), room.code, room.host_key)
    st.markdown(f'<div class="hero"><h1>🎮 {room.game["name"]}</h1>'
                f'<p>Kamu HOST. {len(room.rounds)} ronde · {room.seconds} detik/ronde · '
                f'materi: {room.meta.get("doc", "-")} · konten: {room.meta.get("provider", "-")}</p></div>',
                unsafe_allow_html=True)

    @st.fragment(run_every=POLL)
    def live():
        room.tick()
        if room.phase == "lobby":
            host_lobby(room, player_url)
        elif room.phase in ("question", "vote"):
            host_question(room)
        elif room.phase == "reveal":
            host_reveal(room)
        else:
            host_final(room)
    live()


def host_lobby(room: Room, player_url: str):
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.image(qr_png(player_url), caption="Scan untuk join", width=240)
    with c2:
        st.markdown('<div class="muted">Kode room</div>'
                    f'<div class="code">{room.code}</div>', unsafe_allow_html=True)
        st.code(player_url, language=None)
        st.markdown("**Aturan main:**")
        for r in room.game.get("rules") or _default_rules(room):
            st.markdown(f"- {r}")
    st.markdown(f"### 👥 Pemain ({len(room.players)}/{room.game.get('max_players', 30)})")
    if room.players:
        cols = st.columns(4)
        for i, p in enumerate(room.players.values()):
            team = f" · Kubu {p.team}" if p.team else ""
            cols[i % 4].markdown(f'<div class="card">{p.name}{team}</div>', unsafe_allow_html=True)
    else:
        st.info("Belum ada yang join. Bagikan link/QR di atas.")
    if st.button("▶️ Mulai game", type="primary", disabled=not room.players, use_container_width=True):
        with STORE.lock:
            room.start()
        st.rerun(scope="fragment")


def _default_rules(room: Room) -> list[str]:
    return {
        "quiz": ["Pilih jawaban yang benar sebelum waktu habis.", "Makin cepat, makin besar poinnya."],
        "truefalse": ["Tentukan pernyataan itu FAKTA atau HOAKS.", "Makin cepat, makin besar poinnya."],
        "majority": ["Tidak ada jawaban benar.", "Poin sebanding dengan jumlah orang yang memilih sama denganmu."],
        "estimate": ["Tebak angkanya.", "Paling dekat dapat bonus 300 poin."],
        "open_vote": ["Tulis jawaban sekreatif mungkin.", "Lalu vote jawaban terbaik (bukan punya sendiri).",
                      "Tiap vote = 300 poin."],
    }[room.mechanic] + (["Salah 2x = gugur."] if room.variant == "survival" else []) + \
        (["Skor kubu = rata-rata skor anggotanya."] if room.variant == "teams" else [])


def _timer(room: Room, total: float):
    left = room.time_left()
    st.progress(min(1.0, left / max(total, 1)), text=f"⏱️ {int(left)} detik")


def host_question(room: Room):
    r = room.current
    st.markdown(f'<span class="pill">Ronde {room.idx + 1}/{len(room.rounds)}</span>'
                f'<span class="pill">{room.phase.upper()}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-q">{r["q"]}</div>', unsafe_allow_html=True)
    if room.mechanic in ("quiz", "majority"):
        for i, o in enumerate(r["options"]):
            if i % 2 == 0:
                cols = st.columns(2)
            cols[i % 2].markdown(f'<div class="card" style="border-left:8px solid '
                                 f'{["#E11D48", "#2563EB", "#D97706", "#059669"][i]}">'
                                 f'{OPTION_ICONS[i]} {o}</div>', unsafe_allow_html=True)
    elif room.mechanic == "truefalse":
        st.markdown("**FAKTA** atau **HOAKS**?")
    if room.phase == "question":
        _timer(room, room.seconds)
        n = len(room.answers.get(room.idx, {}))
        st.caption(f"Sudah menjawab: {n}/{len(room.eligible())}")
    else:
        from core.engine import VOTE_SEC
        _timer(room, VOTE_SEC)
        st.caption(f"Voting: {len(room.votes.get(room.idx, {}))}/{len(room.players)}")
        for _, ans in room.answers_for_vote():
            st.markdown(f'<div class="card">💬 {ans}</div>', unsafe_allow_html=True)
    if st.button("⏭️ Akhiri ronde sekarang", use_container_width=True):
        with STORE.lock:
            room.next()
        st.rerun(scope="fragment")


def host_reveal(room: Room):
    r, res = room.current, room.results.get(room.idx, {})
    st.markdown(f'<span class="pill">Hasil ronde {room.idx + 1}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-q">{r["q"]}</div>', unsafe_allow_html=True)
    ans = room.answers.get(room.idx, {})
    if room.mechanic == "quiz":
        st.success(f"Jawaban benar: **{r['options'][r['answer']]}** · {res.get('n_correct', 0)} orang benar")
        _bar([r["options"][a["value"]] for a in ans.values()], r["options"], r["options"][r["answer"]])
    elif room.mechanic == "truefalse":
        st.success(f"Jawaban: **{'FAKTA' if r['answer'] else 'HOAKS'}**. {r.get('explain', '')}")
        _bar(["FAKTA" if a["value"] else "HOAKS" for a in ans.values()], ["FAKTA", "HOAKS"],
             "FAKTA" if r["answer"] else "HOAKS")
    elif room.mechanic == "majority":
        st.info("Pilihan mayoritas dapat poin paling banyak.")
        _bar([r["options"][a["value"]] for a in ans.values()], r["options"])
    elif room.mechanic == "estimate":
        st.success(f"Jawaban: **{res.get('truth'):g} {r.get('unit', '')}**")
        diffs = res.get("diffs", {})
        if diffs:
            best = min(diffs, key=diffs.get)
            st.markdown(f"Paling dekat: **{room.players[best].name}** "
                        f"(tebak {ans[best]['value']})" if best in room.players else "")
    elif room.mechanic == "open_vote":
        tally = res.get("tally", {})
        for pid, a in sorted(ans.items(), key=lambda kv: -tally.get(kv[0], 0)):
            name = room.players[pid].name if pid in room.players else "?"
            st.markdown(f'<div class="card">💬 {a["value"]}<br><span class="muted">{name} · '
                        f'{tally.get(pid, 0)} vote</span></div>', unsafe_allow_html=True)
    _leaderboard(room, 5)
    _timer(room, 12)
    if st.button("⏭️ Lanjut", type="primary", use_container_width=True):
        with STORE.lock:
            room.next()
        st.rerun(scope="fragment")


def host_final(room: Room):
    st.balloons() if not st.session_state.get(f"balloons_{room.code}") else None
    st.session_state[f"balloons_{room.code}"] = True
    st.markdown("## 🏆 Hasil akhir")
    if room.variant == "teams":
        ts = room.team_scores()
        win = max(ts, key=ts.get)
        st.success(f"Kubu **{win}** menang! (A: {ts['A']} · B: {ts['B']})")
    _leaderboard(room, 50)
    df = pd.DataFrame([{"nama": p.name, "skor": p.score, "kubu": p.team or "", "gugur": not p.alive}
                       for p in room.leaderboard()])
    st.download_button("⬇️ Unduh skor (CSV)", df.to_csv(index=False).encode(), f"skor_{room.code}.csv",
                       use_container_width=True)
    if st.button("🗑️ Tutup room", use_container_width=True):
        STORE.delete(room.code)
        st.rerun()


def _bar(values: list[str], order: list[str], correct: str | None = None):
    import plotly.graph_objects as go
    counts = pd.Series(values, dtype="object").value_counts().reindex(order, fill_value=0)
    colors = ["#059669" if (correct is not None and o == correct) else "#A78BFA" for o in order]
    fig = go.Figure(go.Bar(x=counts.values, y=[f"{OPTION_ICONS[i]} {o}" for i, o in enumerate(order)],
                           orientation="h", marker_color=colors, text=counts.values, textposition="outside"))
    fig.update_layout(height=60 + 40 * len(order), margin=dict(l=0, r=30, t=0, b=0),
                      yaxis=dict(autorange="reversed"), xaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _leaderboard(room: Room, k: int):
    rows = "".join(f'<div class="lb-row"><span>{i + 1}. {p.name}'
                   f'{" · " + p.team if p.team else ""}{" 💀" if not p.alive else ""}</span>'
                   f'<span class="score">{p.score}</span></div>'
                   for i, p in enumerate(room.leaderboard()[:k]))
    st.markdown(f'<div class="card"><h4>Papan skor</h4>{rows}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PESERTA
# ---------------------------------------------------------------------------
def player_view(room: Room):
    pid = st.query_params.get("pid") or st.session_state.get(f"pid_{room.code}")
    if not pid or pid not in room.players:
        st.markdown(f'<div class="hero"><h1>{room.game["name"]}</h1>'
                    f'<p>{room.game.get("desc", "")}</p></div>', unsafe_allow_html=True)
        if room.phase not in ("lobby",) and room.variant == "survival":
            st.warning("Game survival sudah mulai, kamu tidak bisa ikut di tengah.")
            return
        with st.form("join"):
            name = st.text_input("Nama panggilan", max_chars=20)
            ok = st.form_submit_button("Masuk room", type="primary", use_container_width=True)
        if ok:
            try:
                with STORE.lock:
                    p = room.join(name)
            except ValueError as e:
                st.error(str(e))
                return
            st.session_state[f"pid_{room.code}"] = p.pid
            st.query_params["pid"] = p.pid
            st.rerun()
        return

    me = room.players[pid]

    @st.fragment(run_every=POLL)
    def live():
        room.tick()
        me.last_seen = time.time()
        team = f" · Kubu {me.team}" if me.team else ""
        st.markdown(f'<div class="card"><b>{me.name}</b>{team} · <span class="score">{me.score}</span> poin'
                    f'{" · 💀 gugur" if not me.alive else ""}</div>', unsafe_allow_html=True)
        if room.phase == "lobby":
            st.info(f"Kamu sudah masuk. Tunggu host memulai game... ({len(room.players)} pemain)")
        elif room.phase == "question":
            player_question(room, me)
        elif room.phase == "vote":
            player_vote(room, me)
        elif room.phase == "reveal":
            player_reveal(room, me)
        else:
            player_final(room, me)
    live()


def player_question(room: Room, me):
    r = room.current
    st.markdown(f'<span class="pill">Ronde {room.idx + 1}/{len(room.rounds)}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-q">{r["q"]}</div>', unsafe_allow_html=True)
    _timer(room, room.seconds)
    if not me.alive:
        st.caption("Kamu sudah gugur. Tonton dan dukung temanmu!")
        return
    mine = room.answers.get(room.idx, {}).get(me.pid)
    if mine is not None:
        st.success("Jawaban terkirim ✅ Tunggu yang lain...")
        return
    k = f"{room.code}_{room.idx}"
    if room.mechanic in ("quiz", "majority"):
        for i, o in enumerate(r["options"]):
            if i % 2 == 0:
                cols = st.columns(2)
            if cols[i % 2].button(f"{OPTION_ICONS[i]}  {o}", key=f"opt_{k}_{i}_c{i}", use_container_width=True):
                _send(room, me.pid, i)
    elif room.mechanic == "truefalse":
        c1, c2 = st.columns(2)
        if c1.button("✅ FAKTA", key=f"opt_{k}_t_c3", use_container_width=True):
            _send(room, me.pid, True)
        if c2.button("❌ HOAKS", key=f"opt_{k}_f_c0", use_container_width=True):
            _send(room, me.pid, False)
    elif room.mechanic == "estimate":
        with st.form(f"est_{k}"):
            v = st.number_input(f"Tebakanmu {r.get('unit', '')}", value=0.0, step=1.0)
            if st.form_submit_button("Kirim", type="primary", use_container_width=True):
                _send(room, me.pid, v)
    elif room.mechanic == "open_vote":
        with st.form(f"open_{k}"):
            v = st.text_area("Jawabanmu", max_chars=200)
            if st.form_submit_button("Kirim", type="primary", use_container_width=True):
                _send(room, me.pid, v)


def _send(room: Room, pid: str, value):
    with STORE.lock:
        msg = room.submit(pid, value)
    if msg != "ok":
        st.toast(msg)
    st.rerun(scope="fragment")


def player_vote(room: Room, me):
    from core.engine import VOTE_SEC
    st.markdown("### 🗳️ Vote jawaban terbaik")
    _timer(room, VOTE_SEC)
    if me.pid in room.votes.get(room.idx, {}):
        st.success("Vote terkirim ✅")
        return
    for i, (pid, ans) in enumerate(room.answers_for_vote(exclude=me.pid)):
        if st.button(f"💬 {ans}", key=f"vote_{room.code}_{room.idx}_{pid}", use_container_width=True):
            with STORE.lock:
                msg = room.vote(me.pid, pid)
            if msg != "ok":
                st.toast(msg)
            st.rerun(scope="fragment")


def player_reveal(room: Room, me):
    r = room.current
    gain = room.results.get(room.idx, {}).get("gains", {}).get(me.pid, 0)
    if room.mechanic == "quiz":
        st.markdown(f"Jawaban benar: **{r['options'][r['answer']]}**")
    elif room.mechanic == "truefalse":
        st.markdown(f"Jawaban: **{'FAKTA' if r['answer'] else 'HOAKS'}**. {r.get('explain', '')}")
    elif room.mechanic == "estimate":
        st.markdown(f"Jawaban: **{float(r['answer']):g} {r.get('unit', '')}**")
    (st.success if gain > 0 else st.warning)(f"+{gain} poin")
    rank = [p.pid for p in room.leaderboard()].index(me.pid) + 1
    st.metric("Peringkatmu", f"#{rank} dari {len(room.players)}")


def player_final(room: Room, me):
    rank = [p.pid for p in room.leaderboard()].index(me.pid) + 1
    st.markdown(f"## {'🏆' if rank == 1 else '🎉'} Peringkat #{rank}")
    st.metric("Skor akhir", me.score)
    _leaderboard(room, 10)
    key = f"rated_{room.code}"
    if not st.session_state.get(key):
        st.markdown("**Seru gak gamenya?** (rating ini dipakai buat nentuin game yang dipertahankan)")
        v = st.feedback("stars", key=f"stars_{room.code}")
        if v is not None:
            _save_rating(room, v + 1)
            st.session_state[key] = True
            st.success("Makasih!")


def _save_rating(room: Room, stars: int):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    f = RUNTIME_DIR / "ratings.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "room", "game_id", "stars"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), room.code, room.game["id"], stars])
