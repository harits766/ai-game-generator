"""Katalog template game (pool). Library aktif = Top-10 dari pool ini + 1 slot multiverse.

Setiap game memakai salah satu MEKANIK yang didukung engine (core/engine.py).
Game baru (custom/AI) juga wajib memakai salah satu mekanik ini, sehingga AI
tidak pernah menulis kode, hanya mengisi spesifikasi yang tervalidasi.
"""
from __future__ import annotations

MECHANICS = {
    "quiz": "Pilihan ganda, poin benar + bonus kecepatan",
    "truefalse": "Benar/Salah (fakta atau hoaks), poin benar + bonus kecepatan",
    "majority": "Pilih jawaban; poin kalau sama dengan mayoritas",
    "estimate": "Tebak angka; paling dekat menang",
    "open_vote": "Semua tulis jawaban bebas lalu voting jawaban terbaik",
}

# keywords dipakai untuk mencocokkan sinyal tren (Google Trends / TikTok / X)
GAMES: list[dict] = [
    {"id": "kuis_kilat", "name": "Kuis Kilat", "mechanic": "quiz",
     "desc": "Kuis pilihan ganda cepat ala acara TV, makin cepat makin banyak poin.",
     "keywords": ["kuis", "trivia"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "fakta_hoaks", "name": "Fakta atau Hoaks", "mechanic": "truefalse",
     "desc": "Tebak pernyataan dari materi: fakta atau hoaks?",
     "keywords": ["hoaks", "cek fakta"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "satu_frekuensi", "name": "Satu Frekuensi", "mechanic": "majority",
     "desc": "Jawab sesuai isi kepala mayoritas. Beda sendiri = 0 poin.",
     "keywords": ["satu frekuensi", "survey bilang"], "min_players": 4, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "tebak_angka", "name": "Tebak Angka", "mechanic": "estimate",
     "desc": "Tebak angka dari materi (tahun, persen, jumlah). Paling dekat menang.",
     "keywords": ["tebak angka"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "adu_receh", "name": "Adu Receh", "mechanic": "open_vote",
     "desc": "Jelaskan istilah dari materi dengan gaya paling receh, lalu voting.",
     "keywords": ["receh", "jokes bapak"], "min_players": 3, "max_players": 20,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "debat_kilat", "name": "Debat Kilat", "mechanic": "open_vote",
     "desc": "Tulis argumen 1 kalimat paling kuat untuk mosi dari materi, lalu voting.",
     "keywords": ["debat", "mosi"], "min_players": 3, "max_players": 20,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "survival_kuis", "name": "Survival Kuis", "mechanic": "quiz",
     "desc": "Kuis gugur: salah dua kali, kamu jadi penonton.", "variant": "survival",
     "keywords": ["survival", "squid game"], "min_players": 4, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "tim_vs_tim", "name": "Kubu vs Kubu", "mechanic": "quiz",
     "desc": "Kuis antar dua kubu, skor kubu = rata-rata anggota.", "variant": "teams",
     "keywords": ["lomba 17an", "cerdas cermat"], "min_players": 4, "max_players": 30,
     "modes": ["offline", "hybrid"], "launch_week": 0},
    {"id": "emoji_tebak", "name": "Tebak Emoji", "mechanic": "quiz",
     "desc": "Tebak istilah dari materi yang disandikan jadi emoji.",
     "keywords": ["tebak emoji"], "min_players": 2, "max_players": 30,
     "modes": ["online", "hybrid", "offline"], "launch_week": 0},
    {"id": "siapa_paling", "name": "Siapa Paling", "mechanic": "majority",
     "desc": "Pilih opsi yang paling cocok menurut kelompok, cari yang sefrekuensi.",
     "keywords": ["siapa paling", "most likely to"], "min_players": 4, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "cerita_berantai", "name": "Cerita Berantai", "mechanic": "open_vote",
     "desc": "Lanjutkan cerita dari kalimat materi, voting lanjutan terbaik.",
     "keywords": ["cerita berantai", "storytelling"], "min_players": 3, "max_players": 20,
     "modes": ["online", "hybrid", "offline"], "launch_week": 0},
    {"id": "mitos_sains", "name": "Mitos vs Sains", "mechanic": "truefalse",
     "desc": "Pernyataan populer di masyarakat: mitos atau terbukti?",
     "keywords": ["mitos", "fakta unik"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    {"id": "harga_pas", "name": "Harga Pas", "mechanic": "estimate",
     "desc": "Tebak angka statistik terkait topik (biaya, jumlah, persen).",
     "keywords": ["tebak harga"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 0},
    # dua game yang "lahir" belakangan, untuk mensimulasikan game baru yang naik daun
    {"id": "tebak_kata_kunci", "name": "Tebak Kata Kunci", "mechanic": "quiz",
     "desc": "Kalimat materi dengan kata kunci hilang, tebak katanya.",
     "keywords": ["tebak kata", "teka teki"], "min_players": 2, "max_players": 30,
     "modes": ["offline", "online", "hybrid"], "launch_week": 60},
    {"id": "voting_meme", "name": "Caption Meme", "mechanic": "open_vote",
     "desc": "Bikin caption meme dari situasi di materi, voting yang paling kena.",
     "keywords": ["meme", "caption challenge"], "min_players": 3, "max_players": 20,
     "modes": ["online", "hybrid", "offline"], "launch_week": 75},
]

GAME_BY_ID = {g["id"]: g for g in GAMES}


def get_game(game_id: str, extra: dict | None = None) -> dict | None:
    """Cari game di pool, atau di dict tambahan (mis. game multiverse/custom)."""
    if game_id in GAME_BY_ID:
        return GAME_BY_ID[game_id]
    if extra and game_id in extra:
        return extra[game_id]
    return None
