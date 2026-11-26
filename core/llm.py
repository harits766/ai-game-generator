"""Lapisan LLM: Gemini (gratis, utama) -> OpenAI (cadangan) -> offline (tanpa API).

Dua kemampuan:
- generate_json(): bikin konten game terstruktur (JSON).
- run_tools():     loop tool-calling (agentic) dengan function calling native.
Kalau tidak ada API key / kuota habis, pemanggil memakai jalur offline.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable


def _secret(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


@dataclass
class LLMInfo:
    provider: str            # gemini | openai | offline
    model: str
    errors: list[str] = field(default_factory=list)


def providers() -> list[LLMInfo]:
    """Urutan provider yang tersedia sesuai key yang di-set."""
    order = []
    pref = _secret("LLM_PROVIDER", "auto").lower()
    gem = _secret("GEMINI_API_KEY")
    oai = _secret("OPENAI_API_KEY")
    if gem and pref in ("auto", "gemini"):
        order.append(LLMInfo("gemini", _secret("GEMINI_MODEL", "gemini-2.5-flash")))
    if oai and pref in ("auto", "openai", "gemini"):
        order.append(LLMInfo("openai", _secret("OPENAI_MODEL", "gpt-4o-mini")))
    if gem and pref == "openai":
        order.append(LLMInfo("gemini", _secret("GEMINI_MODEL", "gemini-2.5-flash")))
    return order


def status_label() -> str:
    p = providers()
    return f"{p[0].provider} ({p[0].model})" if p else "offline (tanpa API key)"


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return json.loads(text[start:])


# ---------------------------------------------------------------------------
# JSON generation
# ---------------------------------------------------------------------------
def generate_json(prompt: str, system: str = "", temperature: float = 0.7) -> tuple[dict | None, str]:
    """Return (json|None, provider_yang_dipakai_atau_error)."""
    errors = []
    for p in providers():
        try:
            if p.provider == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
                r = client.models.generate_content(
                    model=p.model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system or None, temperature=temperature,
                        response_mime_type="application/json"))
                return _extract_json(r.text), f"gemini:{p.model}"
            if p.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=_secret("OPENAI_API_KEY"))
                r = client.chat.completions.create(
                    model=p.model, temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system or "Jawab dalam JSON."},
                              {"role": "user", "content": prompt}])
                return _extract_json(r.choices[0].message.content), f"openai:{p.model}"
        except Exception as e:  # kuota habis, jaringan, JSON rusak -> coba provider berikutnya
            errors.append(f"{p.provider}: {type(e).__name__}: {str(e)[:120]}")
    return None, "; ".join(errors) or "offline"


# ---------------------------------------------------------------------------
# Tool calling (agentic loop)
# ---------------------------------------------------------------------------
ToolFn = Callable[..., dict]


def run_tools(system: str, history: list[dict], tools: dict[str, tuple[str, dict, ToolFn]],
              max_steps: int = 6) -> tuple[str | None, list[dict], str]:
    """Jalankan loop agent. history = [{"role": "user"|"assistant", "content": str}].

    tools: nama -> (deskripsi, json_schema_parameter, fungsi)
    Return (jawaban_akhir | None, trace_tool, provider/error).
    """
    errors = []
    for p in providers():
        trace: list[dict] = []
        try:
            if p.provider == "gemini":
                return (*_run_gemini(p.model, system, history, tools, max_steps, trace), f"gemini:{p.model}")
            if p.provider == "openai":
                return (*_run_openai(p.model, system, history, tools, max_steps, trace), f"openai:{p.model}")
        except Exception as e:
            errors.append(f"{p.provider}: {type(e).__name__}: {str(e)[:160]}")
    return None, [], "; ".join(errors) or "offline"


def _call_tool(tools, name, args, trace) -> dict:
    if name not in tools:
        out = {"error": f"tool {name} tidak ada"}
    else:
        try:
            out = tools[name][2](**(args or {}))
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
    trace.append({"tool": name, "args": args, "result": out})
    return out


def _run_gemini(model, system, history, tools, max_steps, trace):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
    decls = [types.FunctionDeclaration(name=n, description=d, parameters_json_schema=s)
             for n, (d, s, _) in tools.items()]
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=0.3,
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part.from_text(text=m["content"])]) for m in history]
    for _ in range(max_steps):
        r = client.models.generate_content(model=model, contents=contents, config=cfg)
        calls = r.function_calls or []
        if not calls:
            return r.text or "", trace
        contents.append(r.candidates[0].content)
        parts = []
        for c in calls:
            out = _call_tool(tools, c.name, dict(c.args or {}), trace)
            parts.append(types.Part.from_function_response(name=c.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=parts))
    return "Batas langkah agent tercapai. Coba perjelas permintaanmu.", trace


def _run_openai(model, system, history, tools, max_steps, trace):
    from openai import OpenAI
    client = OpenAI(api_key=_secret("OPENAI_API_KEY"))
    schema = [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
              for n, (d, s, _) in tools.items()]
    msgs = [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"]} for m in history]
    for _ in range(max_steps):
        r = client.chat.completions.create(model=model, messages=msgs, tools=schema, temperature=0.3)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", trace
        msgs.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = _call_tool(tools, tc.function.name, args, trace)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(out, ensure_ascii=False, default=str)[:6000]})
    return "Batas langkah agent tercapai. Coba perjelas permintaanmu.", trace
