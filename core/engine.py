"""Engine room multiplayer (host + peserta). Logika murni Python, tanpa UI.

State room disimpan di memori proses (1 instance Streamlit Cloud = 1 proses), dengan
lock agar aman dipakai banyak sesi sekaligus. Timer berbasis timestamp deadline
(tidak ada thread/tulis per detik): setiap kali ada yang membaca state, `tick()`
mengecek apakah deadline lewat lalu memindah fase.

Fase: lobby -> question -> (vote, khusus open_vote) -> reveal -> question ... -> final
"""
from __future__ import annotations

import random
import secrets
import string
import threading
import time
from dataclasses import dataclass, field

from .config import MAX_ACTIVE_ROOMS, ROOM_IDLE_TTL_MIN

REVEAL_SEC = 12
VOTE_SEC = 30


@dataclass
class Player:
    pid: str
    name: str
    score: int = 0
    team: str | None = None
    strikes: int = 0
    alive: bool = True
    last_seen: float = field(default_factory=time.time)
    last_gain: int = 0


@dataclass
class Room:
    code: str
    host_key: str
    game: dict
    rounds: list[dict]
    seconds: int
    params: dict
    meta: dict
    phase: str = "lobby"
    idx: int = -1
    deadline: float = 0.0
    phase_started: float = 0.0
    players: dict[str, Player] = field(default_factory=dict)
    answers: dict[int, dict[str, dict]] = field(default_factory=dict)
    votes: dict[int, dict[str, str]] = field(default_factory=dict)
    results: dict[int, dict] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    version: int = 0

    # ------------------------------------------------------------------ util
    @property
    def mechanic(self) -> str:
        return self.game["mechanic"]

    @property
    def variant(self) -> str | None:
        return self.game.get("variant")

    @property
    def current(self) -> dict | None:
        return self.rounds[self.idx] if 0 <= self.idx < len(self.rounds) else None

    def time_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def _bump(self):
        self.version += 1
        self.touched = time.time()

    def eligible(self) -> list[Player]:
        return [p for p in self.players.values() if p.alive]

    # ------------------------------------------------------------- lifecycle
    def join(self, name: str) -> Player:
        name = (name or "").strip()[:20] or "Tanpa Nama"
        max_p = int(self.game.get("max_players", 30))
        if len(self.players) >= max_p:
            raise ValueError(f"Room penuh (maks {max_p} pemain).")
        existing = {p.name.lower() for p in self.players.values()}
        base, n = name, 2
        while name.lower() in existing:
            name, n = f"{base} {n}", n + 1
        pid = secrets.token_urlsafe(6)
        p = Player(pid=pid, name=name)
        if self.variant == "teams":
            a = sum(1 for x in self.players.values() if x.team == "A")
            b = sum(1 for x in self.players.values() if x.team == "B")
            p.team = "A" if a <= b else "B"
        self.players[pid] = p
        self._bump()
        return p

    def kick(self, pid: str):
        self.players.pop(pid, None)
        self._bump()

    def start(self):
        if self.phase != "lobby":
            return
        if len(self.players) < 1:
            raise ValueError("Belum ada peserta yang join.")
        self._goto_question(0)

    def _goto_question(self, i: int):
        self.idx = i
        self.phase = "question"
        self.phase_started = time.time()
        self.deadline = self.phase_started + self.seconds
        self.answers.setdefault(i, {})
        self._bump()

    def submit(self, pid: str, value) -> str:
        self.tick()
        p = self.players.get(pid)
        if not p:
            return "Kamu belum terdaftar di room ini."
        p.last_seen = time.time()
        if self.phase != "question":
            return "Waktu menjawab sudah habis."
        if not p.alive:
            return "Kamu sudah gugur, tonton dulu ya."
        if pid in self.answers[self.idx]:
            return "Jawabanmu sudah tercatat."
        if self.mechanic == "open_vote":
            value = str(value).strip()[:200]
            if not value:
                return "Jawaban kosong."
        self.answers[self.idx][pid] = {"value": value, "t": time.time() - self.phase_started}
        self._bump()
        if len(self.answers[self.idx]) >= len(self.eligible()):
            self._end_question()
        return "ok"

    def vote(self, voter: str, target: str) -> str:
        self.tick()
        if self.phase != "vote":
            return "Voting sudah ditutup."
        if voter == target:
            return "Tidak boleh memilih jawaban sendiri."
        if voter not in self.players:
            return "Kamu belum terdaftar."
        self.votes.setdefault(self.idx, {})
        if voter in self.votes[self.idx]:
            return "Vote kamu sudah tercatat."
        self.votes[self.idx][voter] = target
        self._bump()
        if len(self.votes[self.idx]) >= len(self.players):
            self._reveal()
        return "ok"

    def _answered_pids(self) -> set[str]:
        return set(self.answers.get(self.idx, {}))

    def _end_question(self):
        if self.mechanic == "open_vote" and len(self.answers.get(self.idx, {})) >= 2:
            self.phase = "vote"
            self.phase_started = time.time()
            self.deadline = self.phase_started + VOTE_SEC
            self._bump()
        else:
            self._reveal()

    def _reveal(self):
        self._score()
        self.phase = "reveal"
        self.phase_started = time.time()
        self.deadline = self.phase_started + REVEAL_SEC
        self._bump()

    def next(self):
        """Dipanggil host (atau otomatis setelah reveal)."""
        if self.phase in ("question", "vote"):
            if self.phase == "question":
                self._end_question()
            else:
                self._reveal()
            return
        if self.phase == "reveal":
            alive = self.eligible()
            if self.idx + 1 >= len(self.rounds) or (self.variant == "survival" and len(alive) <= 1):
                self.phase = "final"
                self.deadline = 0
                self._bump()
            else:
                self._goto_question(self.idx + 1)

    def tick(self):
        """Pindah fase otomatis berdasarkan deadline."""
        if self.phase in ("question", "vote", "reveal") and time.time() >= self.deadline:
            self.next()

    # --------------------------------------------------------------- scoring
    def _score(self):
        r = self.current
        ans = self.answers.get(self.idx, {})
        gains: dict[str, int] = {pid: 0 for pid in self.players}
        info: dict = {}
        mech = self.mechanic
        if mech in ("quiz", "truefalse"):
            correct = r["answer"]
            for pid, a in ans.items():
                ok = (a["value"] == correct)
                if ok:
                    speed = max(0.0, 1 - a["t"] / max(self.seconds, 1))
                    gains[pid] = 500 + int(500 * speed)
            if self.variant == "survival":
                for p in self.eligible():
                    if gains.get(p.pid, 0) == 0:
                        p.strikes += 1
                        if p.strikes >= 2:
                            p.alive = False
            info["correct"] = correct
            info["n_correct"] = sum(1 for v in gains.values() if v > 0)
        elif mech == "majority":
            counts: dict = {}
            for a in ans.values():
                counts[a["value"]] = counts.get(a["value"], 0) + 1
            if counts:
                top = max(counts.values())
                for pid, a in ans.items():
                    gains[pid] = int(1000 * counts[a["value"]] / max(len(ans), 1))
                    if counts[a["value"]] == top:
                        gains[pid] += 200
            info["counts"] = counts
        elif mech == "estimate":
            truth = float(r["answer"])
            diffs = {}
            for pid, a in ans.items():
                try:
                    diffs[pid] = abs(float(a["value"]) - truth)
                except (TypeError, ValueError):
                    continue
            if diffs:
                ranked = sorted(diffs, key=diffs.get)
                for rank, pid in enumerate(ranked):
                    rel = diffs[pid] / (abs(truth) + 1)
                    gains[pid] = int(max(0, 1000 * (1 - min(rel, 1))) + (300 if rank == 0 else 0))
            info["truth"] = truth
            info["diffs"] = diffs
        elif mech == "open_vote":
            tally: dict[str, int] = {}
            for target in self.votes.get(self.idx, {}).values():
                tally[target] = tally.get(target, 0) + 1
            for pid in ans:
                gains[pid] = 100 + 300 * tally.get(pid, 0)
            info["tally"] = tally
        for pid, g in gains.items():
            if pid in self.players:
                self.players[pid].score += g
                self.players[pid].last_gain = g
        info["gains"] = gains
        self.results[self.idx] = info

    # ------------------------------------------------------------ read model
    def leaderboard(self) -> list[Player]:
        return sorted(self.players.values(), key=lambda p: -p.score)

    def team_scores(self) -> dict[str, float]:
        out = {}
        for t in ("A", "B"):
            m = [p.score for p in self.players.values() if p.team == t]
            out[t] = round(sum(m) / len(m), 1) if m else 0.0
        return out

    def answers_for_vote(self, exclude: str | None = None) -> list[tuple[str, str]]:
        items = [(pid, a["value"]) for pid, a in self.answers.get(self.idx, {}).items() if pid != exclude]
        random.Random(f"{self.code}{self.idx}").shuffle(items)
        return items


class RoomStore:
    def __init__(self):
        self._rooms: dict[str, Room] = {}
        self.lock = threading.RLock()

    def _gc(self):
        now = time.time()
        for code in [c for c, r in self._rooms.items() if now - r.touched > ROOM_IDLE_TTL_MIN * 60]:
            del self._rooms[code]

    def create(self, game: dict, rounds: list[dict], seconds: int, params: dict, meta: dict) -> Room:
        with self.lock:
            self._gc()
            if len(self._rooms) >= MAX_ACTIVE_ROOMS:
                oldest = min(self._rooms.values(), key=lambda r: r.touched)
                del self._rooms[oldest.code]
            while True:
                code = "".join(random.choices(string.ascii_uppercase.replace("O", "").replace("I", ""), k=5))
                if code not in self._rooms:
                    break
            room = Room(code=code, host_key=secrets.token_urlsafe(8), game=game, rounds=rounds,
                        seconds=int(seconds), params=params, meta=meta)
            self._rooms[code] = room
            return room

    def get(self, code: str) -> Room | None:
        with self.lock:
            r = self._rooms.get((code or "").upper())
            if r:
                r.tick()
            return r

    def delete(self, code: str):
        with self.lock:
            self._rooms.pop(code, None)

    def all(self) -> list[Room]:
        with self.lock:
            self._gc()
            return list(self._rooms.values())


STORE = RoomStore()
