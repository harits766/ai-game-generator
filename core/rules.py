"""Validator ATURAN KUNCI untuk game custom / hasil AI.

Game custom hanya boleh lewat kalau:
- punya 4-7 aturan main (masing-masing kalimat pendek)
- mendefinisikan peran host DAN peserta
- memakai salah satu mekanik engine yang didukung
- jumlah pemain, ronde, detik/ronde, dan total durasi ada dalam batas komputasi
"""
from __future__ import annotations

import re

from .catalog import MECHANICS
from .config import KEY_RULES as K

TEMPLATE = {
    "name": "Nama Game",
    "mechanic": "quiz",
    "desc": "Deskripsi singkat 1 kalimat.",
    "roles": {"host": "Apa yang dilakukan host", "peserta": "Apa yang dilakukan peserta"},
    "rules": ["Aturan 1", "Aturan 2", "Aturan 3", "Aturan 4"],
    "max_players": 20,
    "rounds": 8,
    "seconds_per_round": 20,
}


def validate(spec: dict) -> tuple[bool, list[str], dict]:
    """Return (ok, daftar_error, spec_bersih)."""
    errs: list[str] = []
    s = dict(spec or {})

    name = str(s.get("name", "")).strip()
    if not name:
        errs.append("Nama game wajib diisi.")
    elif len(name) > K["max_name_chars"]:
        errs.append(f"Nama game maksimal {K['max_name_chars']} karakter.")
    s["name"] = name[: K["max_name_chars"]]

    if s.get("mechanic") not in MECHANICS:
        errs.append(f"Mekanik harus salah satu dari: {', '.join(MECHANICS)}.")

    rules = [str(r).strip() for r in (s.get("rules") or []) if str(r).strip()]
    if not K["min_rules"] <= len(rules) <= K["max_rules"]:
        errs.append(f"Jumlah aturan harus {K['min_rules']}-{K['max_rules']} (sekarang {len(rules)}).")
    long = [i + 1 for i, r in enumerate(rules) if len(r) > K["max_rule_chars"]]
    if long:
        errs.append(f"Aturan nomor {long} terlalu panjang (maks {K['max_rule_chars']} karakter).")
    if any(re.search(r"<script|https?://|```", r, re.I) for r in rules):
        errs.append("Aturan tidak boleh berisi kode atau link.")
    s["rules"] = rules

    roles = s.get("roles") or {}
    roles = {str(k).lower().strip(): str(v).strip() for k, v in roles.items()}
    if "participant" in roles and "peserta" not in roles:
        roles["peserta"] = roles.pop("participant")
    for r in K["required_roles"]:
        if not roles.get(r):
            errs.append(f"Peran '{r}' wajib didefinisikan.")
    s["roles"] = roles

    def _int(key, lo, hi, label):
        try:
            v = int(s.get(key))
        except (TypeError, ValueError):
            errs.append(f"{label} harus angka.")
            return None
        if not lo <= v <= hi:
            errs.append(f"{label} harus {lo}-{hi} (sekarang {v}).")
        s[key] = v
        return v

    _int("max_players", K["min_players"], K["max_players"], "Maks pemain")
    rounds = _int("rounds", K["min_rounds"], K["max_rounds"], "Jumlah ronde")
    sec = _int("seconds_per_round", K["min_seconds_per_round"], K["max_seconds_per_round"],
               "Detik per ronde")
    if rounds and sec:
        # perkiraan total: waktu jawab + 15 dtk reveal (+ waktu voting untuk open_vote)
        extra = 15 + (sec if s.get("mechanic") == "open_vote" else 0)
        total_min = rounds * (sec + extra) / 60
        if total_min > K["max_session_minutes"]:
            errs.append(f"Perkiraan durasi {total_min:.0f} menit melebihi batas "
                        f"{K['max_session_minutes']} menit.")
    s["desc"] = str(s.get("desc", "")).strip()[:200]
    return (not errs), errs, s
