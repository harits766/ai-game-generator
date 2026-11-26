"""Pemetaan tiap game di katalog ke sumber data REAL.

- wiki   : artikel Wikipedia ("en:Judul" / "id:Judul") yang mewakili jenis game itu.
           Jumlah view mingguan = proksi minat publik terhadap jenis game tersebut.
- gtrends: keyword Google Trends (geo Indonesia).
- bgg    : profil BoardGameGeek (mekanik/kategori) untuk model daya tarik (cold start).
"""
from __future__ import annotations

GAME_SOURCES: dict[str, dict] = {
    "kuis_kilat":       {"wiki": ["en:Kahoot!"], "gtrends": "kahoot"},
    "fakta_hoaks":      {"wiki": ["en:Fact-checking", "id:Hoaks"], "gtrends": "hoaks"},
    "satu_frekuensi":   {"wiki": ["en:Family_Feud"], "gtrends": "family 100"},
    "tebak_angka":      {"wiki": ["en:Wits_&_Wagers"], "gtrends": "tebak angka"},
    "adu_receh":        {"wiki": ["en:Cards_Against_Humanity"], "gtrends": "jokes bapak bapak"},
    "debat_kilat":      {"wiki": ["en:Debate", "id:Debat"], "gtrends": "debat"},
    "survival_kuis":    {"wiki": ["en:Squid_Game"], "gtrends": "squid game"},
    "tim_vs_tim":       {"wiki": ["en:Quiz_bowl", "id:Cerdas_cermat"], "gtrends": "cerdas cermat"},
    "emoji_tebak":      {"wiki": ["en:Emoji"], "gtrends": "tebak emoji"},
    "siapa_paling":     {"wiki": ["en:Never_have_I_ever"], "gtrends": "never have i ever"},
    "cerita_berantai":  {"wiki": ["en:Exquisite_corpse"], "gtrends": "cerita berantai"},
    "mitos_sains":      {"wiki": ["en:MythBusters", "id:Mitos"], "gtrends": "mitos atau fakta"},
    "harga_pas":        {"wiki": ["en:The_Price_Is_Right"], "gtrends": "tebak harga"},
    "tebak_kata_kunci": {"wiki": ["en:Wordle"], "gtrends": "teka teki"},
    "voting_meme":      {"wiki": ["en:Internet_meme", "id:Meme"], "gtrends": "meme"},
}

# Game party populer di luar katalog, dipantau sebagai konteks pasar (tidak masuk ranking).
CONTEXT_WIKI: dict[str, str] = {
    "Werewolf": "en:Werewolf_(social_deduction_game)",
    "Mafia": "en:Mafia_(party_game)",
    "Among Us": "en:Among_Us",
    "Codenames": "en:Codenames_(board_game)",
    "Jackbox": "en:The_Jackbox_Party_Pack",
    "Kahoot": "en:Kahoot!",
}

# Profil mekanik ala BoardGameGeek untuk tiap mekanik engine (dipakai model cold start).
MECHANIC_TO_BGG: dict[str, dict] = {
    "quiz":      {"categories": ["Trivia", "Party Game"], "mechanics": ["Team-Based Game"]},
    "truefalse": {"categories": ["Trivia", "Party Game", "Bluffing"], "mechanics": []},
    "majority":  {"categories": ["Party Game"], "mechanics": ["Voting"]},
    "estimate":  {"categories": ["Trivia", "Party Game", "Math"], "mechanics": ["Betting and Bluffing"]},
    "open_vote": {"categories": ["Party Game", "Humor"], "mechanics": ["Voting", "Storytelling"]},
}
