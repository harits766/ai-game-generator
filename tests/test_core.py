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
    import core.storage as S
    monkeypatch.setattr(S, "RUNTIME_REQUESTS", tmp_path / "req.csv")
    monkeypatch.setenv("GITHUB_TOKEN", "")
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
    for _ in range(120):
        library.log_request(gid2, {"players": 10})
    st = library.refresh_library()
    assert gid2 in st["top"] and len(st["top"]) == 10 and st["multiverse"] is None


def test_cold_start_weight_grows_with_requests():
    s0 = library.compute_scores()
    assert s0["w_req"].iloc[0] == 0                          # log in-app mulai dari nol
    for _ in range(50):
        library.log_request("siapa_paling", {"players": 8})
    s1 = library.compute_scores()
    assert 0.45 < s1["w_req"].iloc[0] <= 0.7
    rank0 = s0.index[s0["game_id"] == "siapa_paling"][0]
    rank1 = s1.index[s1["game_id"] == "siapa_paling"][0]
    assert rank1 < rank0                                     # request nyata menaikkan ranking


def test_appeal_model_scores_profiles():
    from core.appeal import appeal_score
    s = appeal_score(GAME_BY_ID["kuis_kilat"])
    assert 0 <= s <= 1


def test_storage_merge_dedupes():
    from core.storage import _merge
    remote = "request_id,game_id\nA,x\nB,y\n"
    local = "request_id,game_id\nB,y\nC,z\n"
    assert _merge(remote, local) == "request_id,game_id\nA,x\nB,y\nC,z\n"


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
def test_langchain_agent_with_fake_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from core.agent import SYSTEM, build_tools
    from core.lc_agent import run_agent

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    msgs = iter([
        AIMessage("", tool_calls=[{"name": "recommend_games", "args": {"players": 12}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "create_room", "args": {"game_id": "kuis_kilat"}, "id": "2"},
                                  {"name": "create_room", "args": {"game_id": "kuis_kilat"}, "id": "3"}]),
        AIMessage("Room siap"),
    ])
    ctx = AgentContext(base_url="https://x.app", doc=SAMPLE, use_llm=False)
    answer, trace = run_agent([Fake(messages=msgs)], build_tools(ctx), SYSTEM,
                              [{"role": "user", "content": "kajian 12 orang"}])
    assert answer == "Room siap"
    assert [t["tool"] for t in trace] == ["recommend_games", "create_room"]   # create_room dibatasi 1x
    assert len(ctx.rooms) == 1


def test_design_game_offline_passes_rules():
    from core.agent import design_game
    spec, prov = design_game("tebak ayat tentang sabar", use_llm=False)
    assert validate(spec)[0] and prov == "bank ide offline"


def test_compare_verdicts():
    a = [f"g{i}" for i in range(10)]
    assert compare([], a)["verdict"] == "BASELINE"
    assert compare(a, a)["verdict"] == "STABIL"
    b = a[:7] + ["x", "y", "z"]
    c = compare(a, b)
    assert c["verdict"] == "BERUBAH" and c["overlap"] == 7 and c["masuk"] == ["x", "y", "z"]


# ------------------------------------------------------- regresi bug live
def _fake_agent_run(monkeypatch, scripted):
    """Jalankan chat() lewat jalur LangChain dengan model palsu (meniru perilaku Gemini)."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    import core.agent as A

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    monkeypatch.setattr(A, "chat_models", lambda temperature=0.2: [Fake(messages=iter(scripted))])
    monkeypatch.setattr(A, "status_label", lambda: "fake")
    return A


def test_agent_that_only_recommends_still_gets_room(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "recommend_games",
                                   "args": {"players": 10, "mode": "offline"}, "id": "1"}]),
        AIMessage('Aku rekomendasikan "Caption Meme". Mau langsung main "Caption Meme"?'),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    answer, trace, eng = A.chat(ctx, [{"role": "user", "content": "Buat sebuah Game offline utk 10 orang"}])
    assert eng.startswith("LangChain")
    assert "role=host" in answer and "https://x.app/?room=" in answer
    assert "Mau langsung" not in answer
    assert any(t["tool"] == "create_room" and "room_code" in t["result"] for t in trace)


def test_agent_display_name_resolves(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "create_room", "args": {"game_id": "Caption Meme"}, "id": "1"}]),
        AIMessage("Room siap"),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    answer, trace, _ = A.chat(ctx, [{"role": "user", "content": "caption meme"}])
    assert trace[0]["result"]["game"] == "Caption Meme"
    assert "role=host" in answer


def test_agent_denies_named_game_but_room_still_made(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage('Maaf, game "Caption Meme" belum terdaftar di library.'),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    hist = [{"role": "user", "content": "Buat game offline 10 orang"},
            {"role": "assistant", "content": 'Aku rekomendasikan "Caption Meme". Mau langsung main?'},
            {"role": "user", "content": "caption meme"}]
    answer, trace, _ = A.chat(ctx, hist)
    assert "belum terdaftar" not in answer and "role=host" in answer
    assert ctx.rooms and ctx.rooms[0]["game"] == "Caption Meme"


def test_trend_question_does_not_force_room(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "get_trends", "args": {}, "id": "1"}]),
        AIMessage("Yang lagi rame: Kuis Kilat"),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    answer, trace, _ = A.chat(ctx, [{"role": "user", "content": "game apa yang lagi rame minggu ini?"}])
    assert not ctx.rooms


def test_find_game_by_name():
    assert library.find_game("Caption Meme")["id"] == "voting_meme"
    assert library.find_game("caption-meme")["id"] == "voting_meme"
    assert library.find_game("voting_meme")["id"] == "voting_meme"
    assert library.find_game("catur") is None


def test_tes_does_not_create_room_even_if_llm_tries(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "recommend_games", "args": {}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "create_room", "args": {"game_id": "voting_meme"}, "id": "2"}]),
        AIMessage(""),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    answer, trace, _ = A.chat(ctx, [{"role": "user", "content": "tes"}])
    assert not ctx.rooms and "role=host" not in answer
    assert "jumlah orang" in answer


@pytest.mark.parametrize("msg", ["tes", "halo", "p", "assalamualaikum", "test 123"])
def test_vague_messages_offline_ask_for_details(msg):
    ctx = AgentContext(base_url="https://x.app", use_llm=False)
    answer, trace, _ = chat(ctx, [{"role": "user", "content": msg}])
    assert not ctx.rooms and "jumlah orang" in answer


@pytest.mark.parametrize("msg", ["Buat sebuah Game offline utk 10 orang", "kajian 15 orang", "bikinin kuis dong",
                                 "caption meme", "gathering kantor online"])
def test_real_requests_pass_gate(msg):
    from core.agent import is_game_request
    assert is_game_request(msg)


def test_confirmation_after_recommendation_creates_room():
    from core.agent import is_game_request
    prev = 'Aku rekomendasikan "Kuis Kilat". Mau langsung main?'
    assert is_game_request("gas", prev) and not is_game_request("gas", "")
    ctx = AgentContext(base_url="https://x.app", use_llm=False)
    answer, _, _ = chat(ctx, [{"role": "user", "content": "kajian 10 orang"},
                              {"role": "assistant", "content": prev},
                              {"role": "user", "content": "gas"}])
    assert ctx.rooms and "role=host" in answer


# ------------------------------------------------- "yang lain" + ide baru + kuota
def test_offline_yang_lain_gives_different_game():
    ctx = AgentContext(base_url="https://x.app", use_llm=False)
    h = [{"role": "user", "content": "arisan keluarga 8 orang 20 menit"}]
    a1, _, _ = chat(ctx, h)
    first = ctx.rooms[-1]["game"]
    seen = {first}
    h += [{"role": "assistant", "content": a1}]
    for _ in range(3):
        h += [{"role": "user", "content": "yang lain dong"}]
        ctx = AgentContext(base_url="https://x.app", use_llm=False)
        a, _, _ = chat(ctx, h)
        g = ctx.rooms[-1]["game"]
        assert g not in seen, (g, seen)
        assert ctx.rooms[-1]["host_url"] in a
        seen.add(g)
        h += [{"role": "assistant", "content": a}]


def test_offline_new_idea_creates_custom_room():
    ctx = AgentContext(base_url="https://x.app", use_llm=False)
    a, trace, _ = chat(ctx, [{"role": "user", "content": "kajian 12 orang, kasih ide game baru yang beda"}])
    tools = [t["tool"] for t in trace]
    assert "design_custom_game" in tools and "register_custom_game" in tools
    assert ctx.rooms and "role=host" in a and "Aturan" in a
    st = library.load_state()
    assert any(g["name"] == ctx.rooms[-1]["game"] for g in st["custom_games"].values())


def test_new_ideas_do_not_repeat():
    from core.agent import design_game
    names = set()
    for _ in range(4):
        spec, _ = design_game("", use_llm=False)
        assert validate(spec)[0]
        library.add_custom_game(validate(spec)[2])
        names.add(spec["name"])
    assert len(names) == 4


def test_llm_alt_request_excludes_shown(monkeypatch):
    from langchain_core.messages import AIMessage
    A = _fake_agent_run(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "recommend_games", "args": {"players": 8}, "id": "1"}]),
        AIMessage("Coba yang ini"),
    ])
    ctx = A.AgentContext(base_url="https://x.app", use_llm=True)
    hist = [{"role": "user", "content": "keluarga 8 orang"},
            {"role": "assistant", "content": "Room siap (Kuis Kilat). Alternatif: Fakta atau Hoaks"},
            {"role": "user", "content": "yang lain"}]
    answer, trace, _ = A.chat(ctx, hist)
    recs = trace[0]["result"]["recommendations"]
    assert recs and all(r["game_id"] not in ("kuis_kilat", "fakta_hoaks") for r in recs)
    assert ctx.rooms and ctx.rooms[-1]["game"] not in ("Kuis Kilat", "Fakta atau Hoaks")


def test_rate_limit_cooldown_switches_model(monkeypatch):
    from core import llm
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(llm, "_COOLDOWN", {})
    assert [n for _, n in llm.model_names()] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    llm.mark_rate_limited("Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429")
    assert [n for _, n in llm.model_names()] == ["gemini-2.5-flash-lite"]
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    llm.mark_rate_limited("429", "gemini-2.5-flash-lite")
    assert [n for _, n in llm.model_names()] == ["gpt-4o-mini"]
