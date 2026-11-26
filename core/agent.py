"""Agent GenAI: chatbot yang memakai tools untuk merekomendasikan & membuat room game.

Arsitektur agentic: LLM (Gemini/OpenAI) memutuskan tool mana yang dipanggil,
hasil tool dikembalikan ke LLM, sampai LLM memberi jawaban akhir (maks
MAX_AGENT_STEPS langkah). Tanpa API key, planner rule-based memanggil tools yang
sama secara berurutan, jadi aplikasi tetap jalan saat demo.

Komponen GenAI (semuanya LangChain):
  1. Agent "game_master" (langchain create_agent, ReAct di atas LangGraph) dengan 7 tools:
     get_trends, recommend_games, read_document, design_custom_game,
     validate_custom_game, register_custom_game, create_room
  2. Chain "desainer" (design_custom_game): LLM -> GameSpec (Pydantic) -> validasi aturan kunci
  3. Chain "konten" (core/generator.py): potongan dokumen -> soal terstruktur
"""
from __future__ import annotations

import re

import pandas as pd
from dataclasses import dataclass, field
from urllib.parse import urlencode

from . import library
from .catalog import MECHANICS
from .config import EVENT_TYPES, LOCATIONS, MAX_AGENT_STEPS, MODES
from .documents import Document, from_text, retrieve
from .engine import STORE
from .generator import generate_rounds
from .llm import chat_models, status_label, structured
from .rules import TEMPLATE, validate
from pydantic import BaseModel, Field


@dataclass
class AgentContext:
    base_url: str = ""
    doc: Document | None = None
    use_llm: bool = True
    rooms: list[dict] = field(default_factory=list)
    # diisi chat(): pesan user terakhir + jawaban AI sebelumnya, untuk gerbang create_room
    user_text: str | None = None
    prev_answer: str = ""


# ---------------------------------------------------------------------------
# service (dipakai agent DAN form UI)
# ---------------------------------------------------------------------------
def room_links(base_url: str, code: str, host_key: str) -> tuple[str, str]:
    base = base_url.rstrip("/") + "/"
    return (base + "?" + urlencode({"room": code, "role": "host", "key": host_key}),
            base + "?" + urlencode({"room": code}))


def normalize_params(p: dict) -> dict:
    out = dict(p or {})
    out["players"] = int(min(max(int(out.get("players") or 10), 2), 30))
    out["duration_min"] = int(min(max(int(out.get("duration_min") or 20), 5), 60))
    out["mode"] = out.get("mode") if out.get("mode") in MODES else "offline"
    if out.get("event_type") not in EVENT_TYPES:
        out["event_type"] = _closest(out.get("event_type"), EVENT_TYPES) or "komunitas"
    if out.get("location") and out["location"] not in LOCATIONS:
        out["location"] = _closest(out["location"], LOCATIONS) or out["location"]
    out["topic"] = str(out.get("topic") or "").strip()[:80]
    return out


def _closest(v, options):
    if not v:
        return None
    v = str(v).lower()
    for o in options:
        if v in o.lower() or o.lower().split("/")[0] in v:
            return o
    return None


def create_room_service(ctx: AgentContext, game_id: str, params: dict) -> dict:
    game = library.find_game(game_id)
    if not game:
        ids = ", ".join(f"{g['id']} ({g['name']})" for g in library.active_games())
        return {"error": f"Game '{game_id}' tidak ditemukan. Pakai salah satu game_id: {ids}"}
    params = normalize_params(params)
    doc = ctx.doc
    if doc is None:
        topic = params["topic"] or "pengetahuan umum"
        if ctx.use_llm and library_has_llm():
            doc = from_text(f"Topik: {topic}. Buat pertanyaan seputar {topic} yang umum diketahui "
                            f"masyarakat Indonesia.", name=topic)
        else:
            # tanpa dokumen & tanpa LLM: pakai materi contoh yang paling mirip topiknya
            doc = sample_doc_for(topic)
    if game.get("custom"):
        n, sec = int(game["rounds"]), int(game["seconds_per_round"])
    else:
        n, sec = library.rounds_for(params["duration_min"], game["mechanic"])
    rounds, meta = generate_rounds(doc, game, n, params["topic"], use_llm=ctx.use_llm)
    if not rounds:
        return {"error": "Konten tidak bisa dibuat dari dokumen ini. Coba dokumen yang lebih panjang."}
    room = STORE.create(game, rounds, sec, params, {**meta, "doc": doc.name})
    params["doc_type"] = doc.doc_type
    library.log_request(game["id"], params)
    host, player = room_links(ctx.base_url, room.code, room.host_key)
    info = {"room_code": room.code, "game": game["name"], "rounds": len(rounds),
            "seconds_per_round": sec, "host_url": host, "player_url": player,
            "content_source": meta.get("source"), "provider": meta.get("provider")}
    ctx.rooms.append(info)
    return info


def library_has_llm() -> bool:
    from .llm import provider_names
    return bool(provider_names())


# ---------------------------------------------------------------------------
# chain desainer game custom
# ---------------------------------------------------------------------------
class Roles(BaseModel):
    host: str = Field(description="apa yang dilakukan host")
    peserta: str = Field(description="apa yang dilakukan peserta")


class GameSpec(BaseModel):
    name: str = Field(description="nama game, maks 40 karakter")
    desc: str = Field(description="deskripsi 1 kalimat")
    mechanic: str = Field(description=f"salah satu: {', '.join(MECHANICS)}")
    roles: Roles
    rules: list[str] = Field(description="4-7 aturan main, kalimat pendek")
    max_players: int = Field(description="2-30")
    rounds: int = Field(description="3-15")
    seconds_per_round: int = Field(description="10-120")


def design_game(idea: str, use_llm: bool = True) -> tuple[dict, str]:
    """Ide bebas -> spesifikasi game. LLM (structured output) atau template offline."""
    if use_llm and library_has_llm():
        spec, prov = structured(
            GameSpec, "Kamu desainer game party edukatif untuk acara kumpul di Indonesia. Patuhi batas angka.",
            f"Rancang game dari ide berikut: {idea}\nMekanik yang tersedia: "
            + "; ".join(f"{k} = {v}" for k, v in MECHANICS.items()), temperature=0.7)
        if spec is not None:
            return spec.model_dump(), prov
    return ({**TEMPLATE, "name": idea[:40].strip().title() or "Game Baru", "desc": idea[:200],
             "rules": ["Host membacakan soal dari materi.", "Peserta menjawab lewat HP sebelum waktu habis.",
                       "Jawaban benar dan cepat dapat poin lebih besar.",
                       "Skor tertinggi di akhir ronde jadi pemenang."]}, "template offline")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def build_tools(ctx: AgentContext) -> dict:
    def get_trends(top_k: int = 10) -> dict:
        state = library.load_state()
        if not state.get("top"):
            state = library.refresh_library()
        sc = library.compute_scores().set_index("game_id")

        def num(r, c, nd=1):
            v = r.get(c) if r is not None else None
            return None if v is None or pd.isna(v) else round(float(v), nd)
        rows = []
        for g in library.active_games(state)[: int(top_k) + 1]:
            r = sc.loc[g["id"]] if g["id"] in sc.index else None
            rows.append({"game_id": g["id"], "name": g["name"], "mechanic": g["mechanic"],
                         "skor": num(r, "pred"), "wikipedia_views_minggu_ini": num(r, "views_t", 0),
                         "momentum": num(r, "momentum", 3), "google_trends_id": num(r, "google_trends", 0),
                         "daya_tarik_bgg": num(r, "appeal", 3), "request_in_app_28h": num(r, "req_28d", 0),
                         "multiverse": bool(g.get("custom"))})
        w = float(sc["w_req"].iloc[0]) if len(sc) else 0
        return {"library": rows, "multiverse": state.get("multiverse"), "bobot_request_in_app": w,
                "updated_at": state.get("updated_at")}

    def recommend_games(players: int = 10, duration_min: int = 20, event_type: str = "",
                        mode: str = "offline", location: str = "", topic: str = "") -> dict:
        p = normalize_params(locals())
        return {"params": p, "recommendations": library.recommend(p, k=3)}

    def read_document(query: str = "") -> dict:
        if ctx.doc is None:
            return {"error": "User belum upload dokumen. Pakai topik dari chat saja."}
        return {"doc": ctx.doc.name, "type": ctx.doc.doc_type, "chars": len(ctx.doc.text),
                "passages": retrieve(ctx.doc, query or ctx.doc.name, k=3)}

    def design_custom_game(idea: str) -> dict:
        spec, prov = design_game(idea, ctx.use_llm)
        ok, errs, clean = validate(spec)
        return {"spec": clean, "valid": ok, "errors": errs, "designer": prov}

    def validate_custom_game(spec: dict) -> dict:
        ok, errs, clean = validate(spec)
        return {"valid": ok, "errors": errs, "spec": clean, "template": TEMPLATE}

    def register_custom_game(spec: dict) -> dict:
        ok, errs, clean = validate(spec)
        if not ok:
            return {"valid": False, "errors": errs}
        gid, status = library.add_custom_game(clean)
        return {"valid": True, "game_id": gid, "slot_status": status}

    def create_room(game_id: str, players: int = 10, duration_min: int = 20, event_type: str = "",
                    mode: str = "offline", location: str = "", topic: str = "") -> dict:
        args = dict(locals())
        args.pop("game_id")
        if ctx.user_text is not None and not is_game_request(ctx.user_text, ctx.prev_answer):
            return {"error": "Pesan user belum berisi permintaan game. JANGAN buat room; sapa singkat lalu "
                             "tanyakan jumlah orang, durasi, jenis acara, online/offline, dan topik."}
        return create_room_service(ctx, game_id, args)

    params_schema = {
        "players": {"type": "integer", "description": "jumlah peserta (2-30)"},
        "duration_min": {"type": "integer", "description": "durasi main dalam menit (5-60)"},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "mode": {"type": "string", "enum": MODES},
        "location": {"type": "string", "description": "provinsi/kota acara"},
        "topic": {"type": "string", "description": "topik materi"},
    }
    spec_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}, "desc": {"type": "string"},
            "mechanic": {"type": "string", "enum": list(MECHANICS)},
            "roles": {"type": "object", "properties": {"host": {"type": "string"},
                                                        "peserta": {"type": "string"}}},
            "rules": {"type": "array", "items": {"type": "string"}},
            "max_players": {"type": "integer"}, "rounds": {"type": "integer"},
            "seconds_per_round": {"type": "integer"},
        },
        "required": ["name", "mechanic", "roles", "rules", "max_players", "rounds", "seconds_per_round"],
    }
    return {
        "get_trends": ("Lihat library game Top-10 + slot multiverse beserta skor ranking dan "
                       "sinyal tren real (Wikipedia pageviews, Google Trends) serta request in-app.",
                       {"type": "object", "properties": {"top_k": {"type": "integer"}}}, get_trends),
        "recommend_games": ("Rekomendasikan 3 game terbaik untuk parameter acara user.",
                            {"type": "object", "properties": params_schema}, recommend_games),
        "read_document": ("Ambil potongan dokumen yang diupload user yang relevan dengan query.",
                          {"type": "object", "properties": {"query": {"type": "string"}}}, read_document),
        "design_custom_game": ("Rancang spesifikasi game baru dari ide bebas user (memanggil chain desainer), "
                               "lalu langsung divalidasi aturan kunci.",
                               {"type": "object", "properties": {"idea": {"type": "string"}},
                                "required": ["idea"]}, design_custom_game),
        "validate_custom_game": ("Cek apakah spesifikasi game custom lolos aturan kunci (4-7 aturan, "
                                 "peran host & peserta, batas pemain/ronde/durasi).",
                                 {"type": "object", "properties": {"spec": spec_schema},
                                  "required": ["spec"]}, validate_custom_game),
        "register_custom_game": ("Daftarkan game custom yang valid ke slot multiverse. Return game_id.",
                                 {"type": "object", "properties": {"spec": spec_schema},
                                  "required": ["spec"]}, register_custom_game),
        "create_room": ("Buat room game (konten dari dokumen/topik) dan kembalikan link host & peserta.",
                        {"type": "object", "properties": {"game_id": {"type": "string"}, **params_schema},
                         "required": ["game_id"]}, create_room),
    }


SYSTEM = f"""Kamu "Game Master AI", asisten yang membuat game party edukatif dari dokumen/topik.
Masalah yang diselesaikan: acara kumpul sering berakhir semua main HP sendiri-sendiri, dan game party
yang ada kaku (itu-itu saja, harus gonta-ganti aplikasi). Kamu membuat game yang pas dengan acara.

Cara kerja:
0. Kalau pesan user BUKAN permintaan game/acara (mis. salam, "tes", "halo", pesan samar), JANGAN panggil tool
   apa pun. Sapa singkat lalu minta info: jumlah orang, durasi, jenis acara, online/offline, topik.
1. Ambil parameter dari pesan user: jumlah pemain, durasi, jenis acara ({', '.join(EVENT_TYPES)}),
   mode (offline/online/hybrid), lokasi, topik. Kalau ada yang kosong, pakai default wajar
   (10 pemain, 20 menit, offline) dan sebutkan asumsinya. Jangan bertanya lebih dari 1 kali.
2. Panggil recommend_games, lalu LANGSUNG panggil create_room dengan game_id rekomendasi teratas
   di giliran yang sama. JANGAN bertanya "mau langsung main?": user datang untuk dapat link.
   Kalau user menyebut nama game tertentu, langsung create_room dengan game itu.
   game_id adalah field "game_id" dari hasil tool (mis. voting_meme), bukan nama tampilan.
3. Kalau user ingin bikin game sendiri, panggil design_custom_game (atau susun spec lalu validate_custom_game). Kalau valid dan user
   setuju/minta, register_custom_game lalu create_room dengan game_id baru. Aturan kunci: 4-7 aturan,
   wajib peran host & peserta, maks 30 pemain, 3-15 ronde, 10-120 detik/ronde, total <= 60 menit.
4. Kalau user hanya tanya game apa yang lagi rame, panggil get_trends (tanpa create_room).
Jawab singkat, bahasa Indonesia santai. Saat room dibuat, WAJIB tampilkan host_url dan player_url apa adanya
dan jelaskan: host buka link host di laptop/HP host, bagikan link peserta ke teman.
Jangan mengarang link, skor, atau data: semua angka & link harus berasal dari hasil tool.
Tolak permintaan di luar urusan membuat/merekomendasikan game acara.
Maksimal {MAX_AGENT_STEPS} langkah."""


# ---------------------------------------------------------------------------
# planner offline (tanpa LLM)
# ---------------------------------------------------------------------------
def extract_params(text: str) -> dict:
    t = text.lower()
    p: dict = {}
    m = re.search(r"(\d{1,3})\s*(orang|org|pemain|peserta|anak|warga|santri|siswa)", t)
    if m:
        p["players"] = int(m.group(1))
    m = re.search(r"(\d{1,3})\s*(menit|mnt|min)", t)
    if m:
        p["duration_min"] = int(m.group(1))
    m = re.search(r"(\d)\s*jam", t)
    if m:
        p["duration_min"] = int(m.group(1)) * 60
    if "online" in t or "daring" in t or "zoom" in t or "meet" in t:
        p["mode"] = "online"
    elif "hybrid" in t:
        p["mode"] = "hybrid"
    elif "offline" in t or "luring" in t or "tatap muka" in t:
        p["mode"] = "offline"
    syn = {"kajian": "kajian", "pengajian": "kajian", "halaqah": "kajian", "mentoring": "kajian",
           "kelas": "kelas/sekolah", "sekolah": "kelas/sekolah", "kuliah": "kelas/sekolah",
           "kantor": "kantor", "gathering": "kantor", "keluarga": "keluarga", "arisan": "keluarga",
           "komunitas": "komunitas", "karang taruna": "komunitas", "desa": "sosialisasi desa",
           "sosialisasi": "sosialisasi desa", "posyandu": "sosialisasi desa", "seminar": "seminar",
           "webinar": "seminar"}
    for k, v in syn.items():
        if k in t:
            p["event_type"] = v
            break
    for loc in LOCATIONS:
        if loc.lower() in t:
            p["location"] = loc
    city = {"bandung": "Jawa Barat", "bogor": "Jawa Barat", "depok": "Jawa Barat", "bekasi": "Jawa Barat",
            "jakarta": "DKI Jakarta", "surabaya": "Jawa Timur", "malang": "Jawa Timur",
            "semarang": "Jawa Tengah", "solo": "Jawa Tengah", "jogja": "DI Yogyakarta",
            "yogyakarta": "DI Yogyakarta", "tangerang": "Banten", "denpasar": "Bali", "medan": "Sumatera Utara",
            "padang": "Sumatera Barat", "makassar": "Sulawesi Selatan", "mataram": "Nusa Tenggara Barat"}
    for c, prov in city.items():
        if c in t and "location" not in p:
            p["location"] = prov
    m = re.search(r"(?:topik|tema|tentang|materi)\s*[:\-]?\s*([^,.;\n]+)", t)
    if m:
        p["topic"] = m.group(1).strip()[:60]
    return p


def offline_agent(ctx: AgentContext, text: str) -> tuple[str, list[dict]]:
    tools = build_tools(ctx)
    trace = []

    def call(name, **kw):
        out = tools[name][2](**kw)
        trace.append({"tool": name, "args": kw, "result": out})
        return out

    t = text.lower()
    params = extract_params(text)
    if ctx.doc and not params.get("topic"):
        params["topic"] = ctx.doc.name.rsplit(".", 1)[0].replace("_", " ")
    wants_trend = any(w in t for w in ["rame", "ramai", "trend", "tren", "populer", "viral", "top"])
    if wants_trend and not any(k in params for k in ("players", "event_type")):
        tr = call("get_trends", top_k=10)
        def arrow(m):
            return "naik" if (m or 0) > 0.05 else ("turun" if (m or 0) < -0.05 else "stabil")
        lines = [f"{i+1}. **{g['name']}** ({g['mechanic']}) · skor {g['skor']} · tren {arrow(g['momentum'])}"
                 for i, g in enumerate(tr["library"])]
        note = (f"\n\nSkor = tren real (Wikipedia + Google Trends) + daya tarik (BoardGameGeek)"
                f" + request in-app (bobot sekarang {tr['bobot_request_in_app']:.0%}).")
        return "Ini library game yang lagi rame minggu ini:\n\n" + "\n".join(lines) + note, trace
    if not is_game_request(text, ctx.prev_answer):
        return ASK_DETAILS, trace
    rec = call("recommend_games", **params)
    if not rec["recommendations"]:
        return "Belum ada game yang cocok dengan parameter itu. Coba ubah jumlah pemain atau mode.", trace
    best = rec["recommendations"][0]
    if ctx.doc:
        call("read_document", query=params.get("topic", ""))
    room = call("create_room", game_id=best["game_id"], **params)
    p = rec["params"]
    alt = ", ".join(r["name"] for r in rec["recommendations"][1:])
    if "error" in room:
        return f"Gagal bikin room: {room['error']}", trace
    msg = (f"Asumsi acara: **{p['players']} orang, {p['duration_min']} menit, {p['event_type']}, {p['mode']}**"
           f"{', ' + p['location'] if p.get('location') else ''}.\n\n"
           f"Rekomendasi: **{best['name']}** ({', '.join(best['reasons'])}). Alternatif: {alt}.\n\n"
           f"Room **{room['room_code']}** siap, {room['rounds']} ronde.\n\n"
           f"- Link HOST (buka di laptop/HP kamu): {room['host_url']}\n"
           f"- Link PESERTA (bagikan ke teman): {room['player_url']}")
    return msg, trace


def chat(ctx: AgentContext, history: list[dict]) -> tuple[str, list[dict], str]:
    """Return (jawaban, trace_tool, mesin_yang_dipakai)."""
    text = history[-1]["content"]
    ctx.user_text = text
    ctx.prev_answer = next((m["content"] for m in reversed(history[:-1]) if m["role"] == "assistant"), "")
    if ctx.use_llm:
        doc_note = (f"\n[Konteks: user sudah upload dokumen '{ctx.doc.name}' ({ctx.doc.doc_type}, "
                    f"{len(ctx.doc.text)} karakter). Gunakan read_document bila perlu.]") if ctx.doc else ""
        hist = history[:-1] + [{"role": "user", "content": text + doc_note}]
        models = chat_models(temperature=0.2)
        if models:
            from .lc_agent import run_agent
            try:
                answer, trace = run_agent(models, build_tools(ctx), SYSTEM, hist[-10:])
                if answer or trace:
                    answer = ensure_room(ctx, history, answer, trace)
                    return answer, trace, f"LangChain agent · {status_label()}"
                fallback_reason = "agent tidak memberi jawaban"
            except Exception as e:
                fallback_reason = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            fallback_reason = "tanpa API key"
    else:
        fallback_reason = "LLM dimatikan"
    answer, trace = offline_agent(ctx, text)
    return answer, trace, f"offline planner ({fallback_reason})"


TREND_WORDS = ("rame", "ramai", "trend", "tren", "populer", "viral")
DESIGN_WORDS = ("custom", "desain", "rancang", "game sendiri", "game baru", "bikin aturan", "multiverse")
ASK_WORDS = ("game", "games", "gim", "main", "mainan", "permainan", "room", "link", "kuis", "quiz", "acara")
CONFIRM_WORDS = ("gas", "ayo", "oke", "ok", "okay", "ya", "iya", "yuk", "boleh", "mau", "lanjut", "jadi",
                 "setuju", "sip", "itu", "aja", "buat", "bikin", "buatin", "bikinin", "langsung")


def _has_word(text: str, words) -> bool:
    return bool(re.search(r"\b(" + "|".join(map(re.escape, words)) + r")\b", text.lower()))


def is_game_request(text: str, prev_answer: str = "") -> bool:
    """Apakah pesan user benar-benar minta dibuatkan game/room?

    Ya kalau: ada parameter acara (jumlah orang, durasi, jenis acara, mode, topik...), menyebut nama game,
    memakai kata seperti "game/main/room/link/kuis", atau mengiyakan ("gas", "oke") game yang
    disebut AI di jawaban sebelumnya. Salam / "tes" / pesan kosong makna -> tidak.
    """
    t = (text or "").lower()
    if any(w in t for w in DESIGN_WORDS):
        return False
    params = extract_params(text or "")
    if any(w in t for w in TREND_WORDS) and not any(k in params for k in ("players", "event_type")):
        return False
    if params or library.game_mentioned(text or "") or _has_word(t, ASK_WORDS):
        return True
    return bool(prev_answer and library.game_mentioned(prev_answer) and _has_word(t, CONFIRM_WORDS))


ASK_DETAILS = ("Halo! Mau bikin game buat acara apa? Ceritain singkat aja, misalnya: jumlah orang, durasi, "
               "jenis acara (kajian, kelas, kantor, keluarga...), online/offline, dan topiknya. "
               "Contoh: *arisan keluarga 12 orang, 20 menit, offline, topik sejarah kemerdekaan*.")


def _room_block(room: dict) -> str:
    return (f"Room **{room['room_code']}** siap ({room['game']}, {room['rounds']} ronde).\n\n"
            f"- Link HOST (buka di laptop/HP kamu): {room['host_url']}\n"
            f"- Link PESERTA (bagikan ke teman / scan QR): {room['player_url']}")


def ensure_room(ctx: AgentContext, history: list[dict], answer: str, trace: list[dict]) -> str:
    """Pengaman deterministik: user minta game -> pasti dapat room + link host/peserta.

    LLM kadang berhenti di rekomendasi ("mau langsung main?") atau salah membaca nama game.
    Kalau itu terjadi, kode yang membuat room (game yang disebut user > rekomendasi teratas),
    dan link selalu ditempel dari hasil tool, bukan dari tulisan LLM.
    """
    text = history[-1]["content"]
    ok_rooms = [t["result"] for t in trace if t["tool"] == "create_room" and "room_code" in t["result"]]
    if ok_rooms:
        room = ok_rooms[-1]
        return answer if room["host_url"] in answer else (answer + "\n\n" + _room_block(room)).strip()
    if any(t["tool"] in ("design_custom_game", "register_custom_game", "validate_custom_game") for t in trace):
        return answer
    params = extract_params(text)
    prev = next((m["content"] for m in reversed(history[:-1]) if m["role"] == "assistant"), "")
    if not is_game_request(text, prev):
        # salam / "tes" / pesan samar: jangan bikin room, minta detail acara
        return answer or ASK_DETAILS
    recs = next((t["result"].get("recommendations") for t in reversed(trace)
                 if t["tool"] == "recommend_games" and t["result"].get("recommendations")), None)
    named = library.game_mentioned(text)
    if recs is None and not named and prev:
        named = library.game_mentioned(prev)
    tools = build_tools(ctx)
    if recs and not named:
        game = recs[0]
        params = next((t["args"] for t in reversed(trace) if t["tool"] == "recommend_games"), params)
    elif named:
        game = {"game_id": named["id"], "name": named["name"]}
    else:
        rec = tools["recommend_games"][2](**params)
        trace.append({"tool": "recommend_games", "args": params, "result": rec})
        if not rec["recommendations"]:
            return answer
        game = rec["recommendations"][0]
    if ctx.doc and not params.get("topic"):
        params = {**params, "topic": ctx.doc.name.rsplit(".", 1)[0].replace("_", " ")}
    args = {k: v for k, v in params.items() if k in ("players", "duration_min", "event_type", "mode",
                                                       "location", "topic")}
    room = tools["create_room"][2](game_id=game["game_id"], **args)
    trace.append({"tool": "create_room", "args": {"game_id": game["game_id"], **args}, "result": room,
                  "auto": True})
    if "error" in room:
        return (answer + f"\n\nGagal bikin room: {room['error']}").strip()
    lead = f"Langsung aku buatin room **{game['name']}** ya."
    if named and recs is None:
        lead = f"Siap, pakai **{game['name']}**."
    # teks LLM dibuang kalau isinya menolak/membantah game yang barusan dibuat
    keep = answer and not re.search(r"(tidak|belum|gak|nggak) (ada|terdaftar|ditemukan)|maaf", answer.lower())
    body = (answer.split("Mau langsung")[0].strip() + "\n\n") if keep else ""
    return f"{body}{lead}\n\n{_room_block(room)}".strip()


def sample_doc_for(topic: str) -> Document:
    from .config import ROOT
    from .documents import parse_bytes
    files = sorted((ROOT / "samples").glob("*.md"))
    t = (topic or "").lower()
    best, score = files[0], -1
    for f in files:
        text = f.read_text(encoding="utf-8").lower()
        s = sum(text.count(w) for w in re.findall(r"\w{4,}", t))
        if s > score:
            best, score = f, s
    return parse_bytes(best.name, best.read_bytes())
