"""Lapisan LLM berbasis LangChain.

API key HANYA dipakai di 3 tempat (semuanya lewat fungsi di file ini):
  1. Agent Game Master   (core/agent.py)     -> memilih tool & menulis jawaban chat
  2. Chain Konten Game    (core/generator.py) -> membuat soal dari potongan dokumen
  3. Chain Desainer Game  (core/agent.py)     -> draft spesifikasi game custom
Model ML (prediksi tren, daya tarik), retrieval dokumen, validasi aturan, skor, dan
room TIDAK memakai API sama sekali.

Pengaman supaya tidak "ngaco" dan hemat kuota:
- temperature rendah, max output token dibatasi, timeout 30 dtk, retry 1x
- output konten & desain dipaksa ke skema Pydantic (structured output) lalu divalidasi lagi
- LLM hanya melihat <= 4 potongan dokumen (bukan seluruh dokumen)
- urutan: Gemini 2.5 Flash -> Gemini 2.5 Flash-Lite (kuota gratis terpisah) -> OpenAI (cadangan)
  -> mode offline tanpa API
- model yang kena 429 / kuota habis "diistirahatkan" sementara (cooldown), jadi pesan berikutnya
  langsung ke model cadangan tanpa membuang panggilan
"""
from __future__ import annotations

import os
import re
import time
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MAX_OUTPUT_TOKENS = 1500
TIMEOUT = 30


def _secret(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is not None and v != "":
        return v
    if name in os.environ:          # di-set kosong secara eksplisit (mis. saat tes)
        return default
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def provider_names() -> list[str]:
    pref = _secret("LLM_PROVIDER", "auto").lower()
    have = {"gemini": bool(_secret("GEMINI_API_KEY")), "openai": bool(_secret("OPENAI_API_KEY"))}
    order = ["openai", "gemini"] if pref == "openai" else ["gemini", "openai"]
    return [p for p in order if have[p]]


def providers() -> list[str]:          # kompatibilitas lama
    return provider_names()


# ------------------------------------------------------------------ cooldown kuota (429)
_COOLDOWN: dict[str, float] = {}
COOLDOWN_MIN_SEC = 60          # limit per menit
COOLDOWN_DAY_SEC = 30 * 60     # limit harian: coba lagi 30 menit kemudian


def is_rate_limit(err: BaseException | str) -> bool:
    t = str(err)
    return "429" in t or "RESOURCE_EXHAUSTED" in t or "RateLimit" in type(err).__name__ \
        or "rate limit" in t.lower() or "quota" in t.lower()


def mark_rate_limited(err: BaseException | str, model: str | None = None) -> None:
    """Catat model yang kena 429 supaya dilewati sementara."""
    t = str(err)
    if model is None:
        m = re.search(r"model '([^']+)'", t)
        model = m.group(1) if m else None
    if not model:
        return
    _COOLDOWN[model] = time.time() + (COOLDOWN_DAY_SEC if "PerDay" in t or "per day" in t.lower()
                                      else COOLDOWN_MIN_SEC)


def _cooling(model: str) -> bool:
    return _COOLDOWN.get(model, 0) > time.time()


# ------------------------------------------------------------------ budget OpenAI (berbayar)
# OpenAI hanya cadangan dan memakai kredit berbayar, jadi dibatasi per hari (bisa diatur lewat secrets).
# Estimasi gpt-4o-mini: 1 pesan chat lengkap (agent + soal) ~ 4-6 panggilan ~ USD 0,003-0,005.
import json
import threading
from datetime import date

_budget_lock = threading.Lock()


def _budget_file():
    from .config import RUNTIME_DIR
    return RUNTIME_DIR / "llm_usage.json"


def openai_daily_limit() -> int:
    try:
        return int(_secret("OPENAI_DAILY_CALLS", "150"))
    except ValueError:
        return 150


def _usage() -> dict:
    try:
        u = json.loads(_budget_file().read_text())
        if u.get("date") == date.today().isoformat():
            return u
    except Exception:
        pass
    return {"date": date.today().isoformat(), "openai": 0}


def openai_calls_today() -> int:
    return int(_usage().get("openai", 0))


def openai_budget_ok() -> bool:
    return openai_calls_today() < openai_daily_limit()


def record_openai_call() -> None:
    with _budget_lock:
        u = _usage()
        u["openai"] = int(u.get("openai", 0)) + 1
        try:
            f = _budget_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(u))
        except Exception:
            pass


class BudgetExceeded(RuntimeError):
    pass


def is_openai(model_or_name) -> bool:
    n = model_or_name if isinstance(model_or_name, str) else (_model_id(model_or_name) or type(model_or_name).__name__)
    return n.startswith(("gpt", "o1", "o3", "o4")) or "OpenAI" in n


def model_names() -> list[tuple[str, str]]:
    """(provider, nama_model) sesuai key yang ada, urut prioritas, tanpa yang sedang cooldown."""
    out = []
    for p in provider_names():
        if p == "gemini":
            names = [_secret("GEMINI_MODEL", "gemini-2.5-flash"),
                     _secret("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")]
        else:
            names = [_secret("OPENAI_MODEL", "gpt-4o-mini")]
        for n in dict.fromkeys(x for x in names if x):
            out.append((p, n))
    if not openai_budget_ok():
        out = [x for x in out if x[0] != "openai"]       # kredit berbayar dijaga: batas harian tercapai
    ready = [x for x in out if not _cooling(x[1])]
    return ready or out[-1:]       # semua cooldown: tetap coba model terakhir


def status_label() -> str:
    m = model_names()
    if not m:
        return "offline (tanpa API key)"
    return " -> ".join(f"{p} ({n})" for p, n in m)


def chat_models(temperature: float = 0.3) -> list:
    """Daftar chat model LangChain sesuai key yang tersedia (urut prioritas)."""
    out = []
    for p, name in model_names():
        if p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            out.append(ChatGoogleGenerativeAI(
                model=name, google_api_key=_secret("GEMINI_API_KEY"), temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS, timeout=TIMEOUT, max_retries=0))
        else:
            from langchain_openai import ChatOpenAI
            out.append(ChatOpenAI(
                model=name, api_key=_secret("OPENAI_API_KEY"), temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS, timeout=TIMEOUT, max_retries=1))
    return out


def _model_id(m) -> str:
    return str(getattr(m, "model", "") or getattr(m, "model_name", "")).replace("models/", "")


def structured(schema: type[T], system: str, prompt: str, temperature: float = 0.6) -> tuple[T | None, str]:
    """Panggil LLM dengan output terstruktur (Pydantic). Return (objek | None, provider/error)."""
    errors = []
    for m in chat_models(temperature):
        try:
            if is_openai(m):
                if not openai_budget_ok():
                    raise BudgetExceeded("batas harian OpenAI tercapai")
                record_openai_call()
            obj = m.with_structured_output(schema).invoke([("system", system), ("human", prompt)])
            if obj is not None:
                return obj, _model_id(m) or type(m).__name__
        except Exception as e:
            if is_rate_limit(e):
                mark_rate_limited(e, _model_id(m))
            errors.append(f"{_model_id(m)}: {type(e).__name__}: {str(e)[:120]}")
    return None, "; ".join(errors) or "offline"
