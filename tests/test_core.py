"""Tes inti: aturan kunci, engine room, generator offline, library multiverse, agent, ground truth.

Jalankan: pytest -q
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import engine, library
from core.agent import AgentContext, chat, extract_params
from core.catalog import GAME_BY_ID
from core.documents import from_text, parse_bytes
from core.generator import generate_rounds, offline_rounds
from core.ground_truth import compare
from core.rules import TEMPLATE, validate

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = parse_bytes("materi_pemilahan_sampah.md", (ROOT / "samples/materi_pemilahan_sampah.md").read_bytes())


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(library, "LIBRARY_STATE", tmp_path / "state.json")
    monkeypatch.setattr(library, "RUNTIME_REQUESTS", tmp_path / "req.csv")
    import core.features as F
    monkeypatch.setattr(F, "RUNTIME_REQUESTS", tmp_path / "req.csv")
    library._pred_cache.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    yield


# ------------------------------------------------------------------ rules
def test_template_is_valid():
    ok, errs, _ = validate(TEMPLATE)
    assert ok, errs


@pytest.mark.parametrize("patch,msg", [
    ({"rules": ["a", "b", "c"]}, "Jumlah aturan"),
    ({"rules": list("abcdefgh")}, "Jumlah aturan"),
    ({"roles": {"host": "x"}}, "peserta"),
    ({"mechanic": "catur"}, "Mekanik"),
    ({"max_players": 99}, "Maks pemain"),
    ({"rounds": 15, "seconds_per_round": 120, "mechanic": "open_vote"}, "melebihi batas"),
])
def test_rules_reject(patch, msg):
    ok, errs, _ = validate({**TEMPLATE, **patch})
    assert not ok and any(msg in e for e in errs)


def test_participant_alias():
    ok, _, clean = validate({**TEMPLATE, "roles": {"host": "a", "participant": "b"}})
    assert ok and clean["roles"]["peserta"] == "b"


# ------------------------------------------------------------------ engine
def _room(game_id, rounds, n_players=3):
    r = engine.RoomStore().create(GAME_BY_ID[game_id], rounds, 20, {}, {})
    ps = [r.join(f"P{i}") for i in range(n_players)]
    r.start()
    return r, ps


def test_quiz_scoring_and_autoreveal():
    r, ps = _room("kuis_kilat", [{"q": "1+1?", "options": ["1", "2", "3", "4"], "answer": 1}] * 3)
    assert r.submit(ps[0].pid, 1) == "ok"
    assert r.submit(ps[0].pid, 1) != "ok"            # tidak bisa jawab 2x
    r.submit(ps[1].pid, 0)
    r.submit(ps[2].pid, 1)
    assert r.phase == "reveal"                         # semua jawab -> reveal otomatis
    assert ps[0].score >= 500 and ps[1].score == 0


def test_deadline_tick_moves_phase():
    r, ps = _room("kuis_kilat", [{"q": "x", "options": list("abcd"), "answer": 0}] * 3)
    r.deadline = time.time() - 1
    r.tick()
    assert r.phase == "reveal"
    r.deadline = time.time() - 1
    r.tick()
    assert r.phase == "question" and r.idx == 1


def test_survival_eliminates_after_two_wrong():
    r, ps = _room("survival_kuis", [{"q": "x", "options": list("abcd"), "answer": 0}] * 5)
    for _ in range(2):
        r.submit(ps[0].pid, 1)
        r.submit(ps[1].pid, 0)
        r.submit(ps[2].pid, 0)
        r.next()
    assert not ps[0].alive and ps[1].alive


def test_teams_balanced():
    r, ps = _room("tim_vs_tim", [{"q": "x", "options": list("abcd"), "answer": 0}] * 3, n_players=4)
    assert sorted(p.team for p in ps) == ["A", "A", "B", "B"]


def test_majority_and_estimate():
    r, ps = _room("satu_frekuensi", [{"q": "?", "options": list("abcd")}] * 3)
    r.submit(ps[0].pid, 0)
    r.submit(ps[1].pid, 0)
    r.submit(ps[2].pid, 2)
    assert ps[0].score > ps[2].score
    r2, qs = _room("tebak_angka", [{"q": "?", "answer": 100, "unit": ""}] * 3)
    r2.submit(qs[0].pid, 99)
    r2.submit(qs[1].pid, 50)
    r2.submit(qs[2].pid, 500)
    assert qs[0].score > qs[1].score > qs[2].score


def test_open_vote_flow():
    r, ps = _room("adu_receh", [{"q": "jelaskan"}] * 3)
    for i, p in enumerate(ps):
        r.submit(p.pid, f"jawaban {i}")
    assert r.phase == "vote"
    assert r.vote(ps[0].pid, ps[0].pid) != "ok"        # tidak boleh vote diri sendiri
    r.vote(ps[0].pid, ps[1].pid)
    r.vote(ps[1].pid, ps[2].pid)
    r.vote(ps[2].pid, ps[1].pid)
    assert r.phase == "reveal" and ps[1].score > ps[0].score


def test_room_full_and_duplicate_names():
    r = engine.RoomStore().create({**GAME_BY_ID["kuis_kilat"], "max_players": 2}, [{}], 20, {}, {})
    a, b = r.join("Ani"), r.join("ani")
    assert a.name != b.name
    with pytest.raises(ValueError):
        r.join("C")


# --------------------------------------------------------------- generator
@pytest.mark.parametrize("gid", ["kuis_kilat", "fakta_hoaks", "satu_frekuensi", "tebak_angka", "adu_receh"])
def test_offline_rounds_schema(gid):
    g = GAME_BY_ID[gid]
    rounds = offline_rounds(SAMPLE, g, 6, seed="t")
    assert len(rounds) == 6
    for r in rounds:
        assert r["q"]
        if g["mechanic"] == "quiz":
            assert len(r["options"]) == 4 and r["options"][r["answer"]]


def test_generation_cache_hit():
    g = GAME_BY_ID["kuis_kilat"]
    _, m1 = generate_rounds(SAMPLE, g, 4, "sampah", use_llm=False)
    _, m2 = generate_rounds(SAMPLE, g, 4, "sampah", use_llm=False)
    assert m1["source"] == "generated" and m2["source"] == "cache"


def test_estimate_numbers_extracted():
    doc = from_text("Botol plastik terurai dalam 450 tahun. Sekitar 40 persen sampah adalah sisa makanan.")
    rounds = offline_rounds(doc, GAME_BY_ID["tebak_angka"], 2, seed="n")
    assert {r["answer"] for r in rounds} == {450.0, 40.0}


# ----------------------------------------------------------------- library
def test_top10_and_recommend():
    s = library.refresh_library()
    assert len(s["top"]) == 10 and len(set(s["top"])) == 10
    recs = library.recommend({"players": 12, "duration_min": 20, "event_type": "kajian", "mode": "offline"})
    assert recs and all(r["game_id"] in s["top"] for r in recs)


def test_multiverse_promote_and_evict():
    spec = validate(TEMPLATE)[2]
    gid, status = library.add_custom_game(spec)
    assert status == "multiverse"
    gid2, status2 = library.add_custom_game({**spec, "name": "Game Kedua"})
    assert status2 == "antri"
    # sepi peminat + sudah lewat grace period -> dihapus, antrian naik
    st = library.load_state()
    st["multiverse"]["created_at"] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    library.save_state(st)
    st = library.refresh_library()
    assert st["custom_games"][gid]["status"] == "evicted"
    assert st["multiverse"]["id"] == gid2
    # banyak request -> dipromosikan ke Top-10
    for _ in range(60):
        library.log_request(gid2, {"players": 10})
    st = library.refresh_library()
    assert gid2 in st["top"] and len(st["top"]) == 10 and st["multiverse"] is None


# ------------------------------------------------------------------- agent
def test_extract_params():
    p = extract_params("Kajian 15 orang di Bogor, online 30 menit, topik zakat")
    assert p == {"players": 15, "duration_min": 30, "mode": "online", "event_type": "kajian",
                 "location": "Jawa Barat", "topic": "zakat"}


def test_offline_agent_creates_room():
    ctx = AgentContext(base_url="https://x.app", doc=SAMPLE, use_llm=False)
    answer, trace, eng = chat(ctx, [{"role": "user", "content": "sosialisasi desa 20 warga 30 menit"}])
    tools = [t["tool"] for t in trace]
    assert "recommend_games" in tools and "create_room" in tools
    assert "https://x.app/?room=" in answer and "role=host" in answer
    assert eng.startswith("offline")


# ------------------------------------------------------------ ground truth
def test_compare_verdicts():
    a = [f"g{i}" for i in range(10)]
    assert compare([], a)["verdict"] == "BASELINE"
    assert compare(a, a)["verdict"] == "STABIL"
    b = a[:7] + ["x", "y", "z"]
    c = compare(a, b)
    assert c["verdict"] == "BERUBAH" and c["overlap"] == 7 and c["masuk"] == ["x", "y", "z"]
