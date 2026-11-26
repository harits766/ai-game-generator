"""Generator konten game dari dokumen (GenAI + fallback offline) dengan cache TTL.

Output selalu berupa list ronde dengan skema per mekanik:
  quiz      {"q", "options"[4], "answer": idx}
  truefalse {"q", "answer": bool, "explain"}
  majority  {"q", "options"[3-4]}
  estimate  {"q", "answer": float, "unit"}
  open_vote {"q"}
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading

from cachetools import TTLCache

from .config import CACHE_MAXSIZE, CACHE_TTL_SEC, MAX_LLM_CALLS_PER_ROOM
from .documents import Document, keywords, retrieve, split_sentences
from .llm import generate_json

_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SEC)
_cache_lock = threading.Lock()
_stats = {"hit": 0, "miss": 0}

STYLE = {
    "emoji_tebak": "Pertanyaan berupa rangkaian emoji yang menyandikan istilah penting dari materi.",
    "tebak_kata_kunci": "Kalimat dari materi dengan satu kata kunci diganti ____.",
    "survival_kuis": "Pertanyaan makin sulit di ronde akhir.",
    "tim_vs_tim": "Pertanyaan ala cerdas cermat.",
    "mitos_sains": "Pernyataan populer di masyarakat terkait topik: mitos atau terbukti.",
    "fakta_hoaks": "Campuran pernyataan benar dari materi dan pernyataan yang sedikit diplintir.",
    "satu_frekuensi": "Pertanyaan opini ringan; tidak ada jawaban benar, yang dicari jawaban mayoritas.",
    "siapa_paling": "Pertanyaan 'mana yang paling ...' dengan opsi dari istilah materi.",
    "tebak_angka": "Angka (tahun, persen, jumlah) yang ada di materi.",
    "harga_pas": "Angka statistik/biaya terkait topik.",
    "adu_receh": "Minta peserta menjelaskan istilah dari materi dengan gaya lucu tapi sopan.",
    "debat_kilat": "Mosi debat singkat dari isi materi; peserta menulis 1 argumen.",
    "cerita_berantai": "Kalimat pembuka cerita berlatar materi; peserta melanjutkan 1-2 kalimat.",
    "voting_meme": "Situasi lucu terkait materi; peserta membuat caption meme.",
}


def cache_stats() -> dict:
    return {**_stats, "size": len(_cache), "maxsize": _cache.maxsize, "ttl_sec": int(_cache.ttl)}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _key(doc: Document, game: dict, n: int, topic: str) -> str:
    raw = json.dumps([doc.doc_hash, game["id"], game["mechanic"], n, topic.lower().strip()])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def generate_rounds(doc: Document, game: dict, n: int, topic: str = "",
                    use_llm: bool = True) -> tuple[list[dict], dict]:
    """Return (rounds, meta). meta berisi sumber (cache/llm/offline) dan jumlah panggilan LLM."""
    key = _key(doc, game, n, topic)
    with _cache_lock:
        if key in _cache:
            _stats["hit"] += 1
            rounds, meta = _cache[key]
            return rounds, {**meta, "source": "cache", "cache_key": key}
    _stats["miss"] += 1

    rounds: list[dict] = []
    meta = {"llm_calls": 0, "provider": "offline", "errors": []}
    if use_llm:
        context = "\n---\n".join(retrieve(doc, topic or doc.name, k=4))
        for _ in range(MAX_LLM_CALLS_PER_ROOM):
            need = n - len(rounds)
            if need <= 0:
                break
            data, prov = generate_json(_prompt(game, need, topic, context), SYSTEM)
            meta["llm_calls"] += 1
            if data is None:
                meta["errors"].append(prov)
                break
            meta["provider"] = prov
            items = data.get("rounds", data if isinstance(data, list) else [])
            rounds += [r for r in (_clean_item(game["mechanic"], it) for it in items) if r]
    if len(rounds) < n:
        rounds += offline_rounds(doc, game, n - len(rounds), seed=key)
        if meta["provider"] == "offline" or meta["llm_calls"] == 0:
            meta["provider"] = "offline"
        else:
            meta["provider"] += " + offline"
    rounds = rounds[:n]
    with _cache_lock:
        _cache[key] = (rounds, meta)
    return rounds, {**meta, "source": "generated", "cache_key": key}


SYSTEM = ("Kamu perancang game edukasi untuk acara kumpul warga Indonesia. Buat konten yang "
          "akurat berdasarkan MATERI, sopan, bahasa Indonesia santai. Jawab hanya JSON.")


def _prompt(game: dict, n: int, topic: str, context: str) -> str:
    schema = {
        "quiz": '{"q": "...", "options": ["A","B","C","D"], "answer": 0}',
        "truefalse": '{"q": "pernyataan", "answer": true, "explain": "1 kalimat"}',
        "majority": '{"q": "...", "options": ["...","...","...","..."]}',
        "estimate": '{"q": "...?", "answer": 1945, "unit": "tahun"}',
        "open_vote": '{"q": "instruksi/prompt untuk peserta"}',
    }[game["mechanic"]]
    style = game.get("style") or STYLE.get(game["id"], game.get("desc", ""))
    rules = "\n".join(f"- {r}" for r in game.get("rules", []))
    rules_block = f"ATURAN GAME:\n{rules}\n" if rules else ""
    return (f"GAME: {game['name']} (mekanik: {game['mechanic']})\nGAYA: {style}\n"
            f"{rules_block}"
            f"TOPIK: {topic or '-'}\nMATERI:\n{context}\n\n"
            f"Buat tepat {n} ronde. Format: {{\"rounds\": [{schema}, ...]}}. "
            "Untuk quiz, 'answer' adalah indeks opsi benar (0-3) dan acak posisinya.")


def _clean_item(mech: str, it: dict) -> dict | None:
    if not isinstance(it, dict) or not str(it.get("q", "")).strip():
        return None
    q = str(it["q"]).strip()[:300]
    try:
        if mech == "quiz":
            opts = [str(o)[:80] for o in it["options"]][:4]
            ans = int(it["answer"])
            return {"q": q, "options": opts, "answer": ans} if len(opts) == 4 and 0 <= ans < 4 else None
        if mech == "truefalse":
            a = it["answer"]
            a = a if isinstance(a, bool) else str(a).lower() in ("true", "benar", "fakta", "1")
            return {"q": q, "answer": a, "explain": str(it.get("explain", ""))[:200]}
        if mech == "majority":
            opts = [str(o)[:60] for o in it["options"]][:4]
            return {"q": q, "options": opts} if len(opts) >= 2 else None
        if mech == "estimate":
            return {"q": q, "answer": float(str(it["answer"]).replace(",", ".")),
                    "unit": str(it.get("unit", ""))[:20]}
        if mech == "open_vote":
            return {"q": q}
    except (KeyError, ValueError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Offline generator (tanpa LLM): ekstraktif + template
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)(?!\w)(\s?%|\s?persen)?")


def offline_rounds(doc: Document, game: dict, n: int, seed: str = "x") -> list[dict]:
    rnd = random.Random(seed)
    sents = [s for s in split_sentences(doc.text)
             if 35 <= len(s) <= 220 and s[-1] in ".!?"] or split_sentences(doc.text)
    kws = keywords(doc, 30) or ["materi", "topik", "diskusi", "contoh"]
    while len(kws) < 4:
        kws.append(["materi", "topik", "diskusi", "contoh"][len(kws) % 4])
    mech, gid = game["mechanic"], game["id"]
    out: list[dict] = []

    def with_kw():
        pairs = []
        used = {}
        for s in sents:
            hits = [k for k in kws if re.search(rf"\b{re.escape(k)}\b", s, re.I)]
            if hits:
                k = min(hits, key=lambda h: used.get(h, 0))   # variasikan kata kunci
                used[k] = used.get(k, 0) + 1
                pairs.append((s, k))
        rnd.shuffle(pairs)
        return pairs

    if mech == "quiz":
        for s, k in with_kw():
            blank = re.sub(rf"\b{re.escape(k)}\b", "____", s, count=1, flags=re.I)
            opts = [k] + rnd.sample([x for x in kws if x != k], 3)
            rnd.shuffle(opts)
            out.append({"q": f"Lengkapi: {blank}", "options": opts, "answer": opts.index(k)})
    elif mech == "truefalse":
        for i, (s, k) in enumerate(with_kw()):
            if i % 2 == 0:
                out.append({"q": s, "answer": True, "explain": "Sesuai materi."})
            else:
                other = rnd.choice([x for x in kws if x != k])
                fake = re.sub(rf"\b{re.escape(k)}\b", other, s, count=1, flags=re.I)
                out.append({"q": fake, "answer": False,
                            "explain": f"Seharusnya '{k}', bukan '{other}'."})
    elif mech == "majority":
        temps = ["Menurut kalian, istilah mana yang PALING penting dari materi ini?",
                 "Mana yang paling sering kalian temui di kehidupan sehari-hari?",
                 "Kalau harus dijelaskan ke anak kecil, mana yang paling gampang?",
                 "Mana yang paling bikin penasaran buat dibahas lebih lanjut?",
                 "Mana yang paling susah dipraktikkan?"]
        for i in range(n):
            out.append({"q": temps[i % len(temps)], "options": rnd.sample(kws[:12], 4)})
    elif mech == "estimate":
        for s in sents:
            m = NUM_RE.search(s)
            if not m:
                continue
            raw = m.group(1)
            val = float(raw.replace(".", "").replace(",", ".")) if re.search(r"\d[.,]\d{3}", raw) \
                else float(raw.replace(",", "."))
            q = s[:m.start()] + "**berapa?**" + s[m.end():]
            out.append({"q": q, "answer": val, "unit": "%" if m.group(2) else ""})
        rnd.shuffle(out)
        words = re.findall(r"\w+", doc.text.lower())
        for k in kws:
            if len(out) >= n:
                break
            out.append({"q": f"Berapa kali kata '{k}' muncul di materi ini?",
                        "answer": float(words.count(k.lower())), "unit": "kali"})
    elif mech == "open_vote":
        temp = {
            "adu_receh": "Jelaskan '{k}' ke anak TK dengan gaya paling receh (tetap sopan)!",
            "debat_kilat": "Mosi: \"{s}\" Tulis 1 argumen paling kuat (setuju/tidak)!",
            "cerita_berantai": "Lanjutkan cerita ini dalam 1-2 kalimat: \"{s}\"",
            "voting_meme": "Bikin caption meme untuk situasi: ketika semua orang lupa soal '{k}'.",
        }.get(gid, "Tulis jawaban paling kreatif tentang '{k}' dari materi ini!")
        pairs = with_kw() or [(s, rnd.choice(kws)) for s in sents]
        for s, k in pairs:
            out.append({"q": temp.format(k=k, s=s[:160])})
    rnd.shuffle(out)
    # kalau materi terlalu pendek, ulangi variasi agar jumlah ronde terpenuhi
    while out and len(out) < n:
        out.append(dict(rnd.choice(out)))
    return out[:n]
