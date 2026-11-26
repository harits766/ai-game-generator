"""Agent GenAI: chatbot yang memakai tools untuk merekomendasikan & membuat room game.

Arsitektur agentic: LLM (Gemini/OpenAI) memutuskan tool mana yang dipanggil,
hasil tool dikembalikan ke LLM, sampai LLM memberi jawaban akhir (maks
MAX_AGENT_STEPS langkah). Tanpa API key, planner rule-based memanggil tools yang
sama secara berurutan, jadi aplikasi tetap jalan saat demo.

Tools (6):
  get_trends, recommend_games, read_document, validate_custom_game,
  register_custom_game, create_room
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

from . import library
from .catalog import MECHANICS
from .config import EVENT_TYPES, LOCATIONS, MAX_AGENT_STEPS, MODES
from .documents import Document, from_text, retrieve
from .engine import STORE
from .generator import generate_rounds
from .llm import run_tools
from .rules import TEMPLATE, validate


@dataclass
class AgentContext:
    base_url: str = ""
    doc: Document | None = None
    use_llm: bool = True
    rooms: list[dict] = field(default_factory=list)


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
        return {"error": f"Game '{game_id}' tidak ada di library."}
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
    from .llm import providers
    return bool(providers())


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def build_tools(ctx: AgentContext) -> dict:
    def get_trends(top_k: int = 10) -> dict:
        state = library.load_state()
        if not state.get("top"):
            state = library.refresh_library()
        pred = library.predict_demand().set_index("game_id")
        rows = []
        for g in library.active_games(state)[: int(top_k) + 1]:
            r = pred.loc[g["id"]] if g["id"] in pred.index else None
            rows.append({"game_id": g["id"], "name": g["name"], "mechanic": g["mechanic"],
                         "pred_requests_next_week": round(float(r["pred"]), 1) if r is not None else 0,
                         "google_trends": float(r["gtrend_t"]) if r is not None else 0,
                         "tiktok_mentions": int(r["tiktok_t"]) if r is not None else 0,
                         "multiverse": bool(g.get("custom"))})
        return {"library": rows, "multiverse": state.get("multiverse"),
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
        "get_trends": ("Lihat library game Top-10 + slot multiverse beserta prediksi permintaan dan "
                       "sinyal tren (Google Trends, TikTok, X).",
                       {"type": "object", "properties": {"top_k": {"type": "integer"}}}, get_trends),
        "recommend_games": ("Rekomendasikan 3 game terbaik untuk parameter acara user.",
                            {"type": "object", "properties": params_schema}, recommend_games),
        "read_document": ("Ambil potongan dokumen yang diupload user yang relevan dengan query.",
                          {"type": "object", "properties": {"query": {"type": "string"}}}, read_document),
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
1. Ambil parameter dari pesan user: jumlah pemain, durasi, jenis acara ({', '.join(EVENT_TYPES)}),
   mode (offline/online/hybrid), lokasi, topik. Kalau ada yang kosong, pakai default wajar
   (10 pemain, 20 menit, offline) dan sebutkan asumsinya. Jangan bertanya lebih dari 1 kali.
2. Panggil recommend_games, pilih game teratas (boleh sebut 2 alternatif singkat).
3. Kalau user ingin langsung main / minta room / link, panggil create_room.
4. Kalau user ingin bikin game sendiri, susun spec lalu validate_custom_game. Kalau valid dan user
   setuju/minta, register_custom_game lalu create_room dengan game_id baru. Aturan kunci: 4-7 aturan,
   wajib peran host & peserta, maks 30 pemain, 3-15 ronde, 10-120 detik/ronde, total <= 60 menit.
5. Kalau user tanya game apa yang lagi rame, panggil get_trends.
Jawab singkat, bahasa Indonesia santai. Saat room dibuat, WAJIB tampilkan host_url dan player_url apa adanya
dan jelaskan: host buka link host di laptop/HP host, bagikan link peserta ke teman.
Maksimal {MAX_AGENT_STEPS} langkah tool."""


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
        lines = [f"{i+1}. **{g['name']}** ({g['mechanic']}) - prediksi {g['pred_requests_next_week']} request"
                 for i, g in enumerate(tr["library"])]
        return "Ini library game yang lagi rame minggu ini:\n\n" + "\n".join(lines), trace
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
    if ctx.use_llm:
        doc_note = (f"\n[Konteks: user sudah upload dokumen '{ctx.doc.name}' ({ctx.doc.doc_type}, "
                    f"{len(ctx.doc.text)} karakter). Gunakan read_document bila perlu.]") if ctx.doc else ""
        hist = history[:-1] + [{"role": "user", "content": text + doc_note}]
        answer, trace, engine = run_tools(SYSTEM, hist[-10:], build_tools(ctx), MAX_AGENT_STEPS)
        if answer is not None:
            return answer, trace, engine
        fallback_reason = engine
    else:
        fallback_reason = "LLM dimatikan"
    answer, trace = offline_agent(ctx, text)
    return answer, trace, f"offline planner ({fallback_reason})"


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
