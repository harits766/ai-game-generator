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
- urutan: Gemini (gratis) -> OpenAI (cadangan) -> mode offline tanpa API
"""
from __future__ import annotations

import os
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


def status_label() -> str:
    p = provider_names()
    if not p:
        return "offline (tanpa API key)"
    names = {"gemini": _secret("GEMINI_MODEL", "gemini-2.5-flash"), "openai": _secret("OPENAI_MODEL", "gpt-4o-mini")}
    return " -> ".join(f"{x} ({names[x]})" for x in p)


def chat_models(temperature: float = 0.3) -> list:
    """Daftar chat model LangChain sesuai key yang tersedia (urut prioritas)."""
    out = []
    for p in provider_names():
        if p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            out.append(ChatGoogleGenerativeAI(
                model=_secret("GEMINI_MODEL", "gemini-2.5-flash"), google_api_key=_secret("GEMINI_API_KEY"),
                temperature=temperature, max_output_tokens=MAX_OUTPUT_TOKENS, timeout=TIMEOUT, max_retries=1))
        else:
            from langchain_openai import ChatOpenAI
            out.append(ChatOpenAI(
                model=_secret("OPENAI_MODEL", "gpt-4o-mini"), api_key=_secret("OPENAI_API_KEY"),
                temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS, timeout=TIMEOUT, max_retries=1))
    return out


def structured(schema: type[T], system: str, prompt: str, temperature: float = 0.6) -> tuple[T | None, str]:
    """Panggil LLM dengan output terstruktur (Pydantic). Return (objek | None, provider/error)."""
    errors = []
    for m in chat_models(temperature):
        try:
            obj = m.with_structured_output(schema).invoke([("system", system), ("human", prompt)])
            if obj is not None:
                return obj, type(m).__name__
        except Exception as e:
            errors.append(f"{type(m).__name__}: {type(e).__name__}: {str(e)[:120]}")
    return None, "; ".join(errors) or "offline"
