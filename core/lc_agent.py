"""Agent LangChain (create_agent = ReAct loop di atas LangGraph) + pengaman pemakaian API.

Middleware yang dipasang:
- ModelFallbackMiddleware : kalau Gemini gagal/kuota habis -> otomatis pakai OpenAI
- ModelCallLimitMiddleware: maks MAX_AGENT_STEPS panggilan LLM per pesan user
- ToolCallLimitMiddleware : create_room maks 1x per pesan, register_custom_game maks 1x,
                            total tool call maks 8 per pesan
"""
from __future__ import annotations

import json
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (AgentMiddleware, ModelCallLimitMiddleware,
                                         ModelFallbackMiddleware, ToolCallLimitMiddleware)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from .config import MAX_AGENT_STEPS
from .llm import (BudgetExceeded, _model_id, is_openai, is_rate_limit, mark_rate_limited,
                  openai_budget_ok, record_openai_call)


class RateLimitMarker(AgentMiddleware):
    """Pengaman kuota & biaya di setiap percobaan model:
    - model yang kena 429 dicatat supaya pesan berikutnya langsung pakai cadangan
    - panggilan OpenAI (berbayar) dihitung dan ditolak kalau batas harian tercapai
    """

    def wrap_model_call(self, request, handler):
        if is_openai(request.model):
            if not openai_budget_ok():
                raise BudgetExceeded("batas harian OpenAI tercapai")
            record_openai_call()
        try:
            return handler(request)
        except Exception as e:
            if is_rate_limit(e):
                mark_rate_limited(e, _model_id(request.model))
            raise


def to_langchain_tools(spec: dict[str, tuple[str, dict, Callable]], trace: list[dict]) -> list[StructuredTool]:
    """Bungkus fungsi Python biasa jadi tool LangChain + catat setiap pemanggilan ke `trace`."""
    tools = []
    for name, (desc, schema, fn) in spec.items():
        def _run(_fn=fn, _name=name, **kwargs):
            try:
                out = _fn(**kwargs)
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": _name, "args": kwargs, "result": out})
            return json.dumps(out, ensure_ascii=False, default=str)[:6000]
        tools.append(StructuredTool.from_function(func=_run, name=name, description=desc,
                                                  args_schema=schema))
    return tools


def build_agent(models: list, tools: list, system: str):
    middleware = [
        ModelCallLimitMiddleware(run_limit=MAX_AGENT_STEPS, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="create_room", run_limit=1),
        ToolCallLimitMiddleware(tool_name="register_custom_game", run_limit=1),
        ToolCallLimitMiddleware(run_limit=8),
    ]
    if len(models) > 1:
        middleware.insert(0, ModelFallbackMiddleware(*models[1:]))
    middleware.append(RateLimitMarker())     # paling dalam: melihat setiap percobaan model
    return create_agent(models[0], tools=tools, system_prompt=system, middleware=middleware,
                        name="game_master")


def run_agent(models: list, spec: dict, system: str, history: list[dict]) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    agent = build_agent(models, to_langchain_tools(spec, trace), system)
    msgs = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]
    result = agent.invoke({"messages": msgs})
    final = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), None)
    text = final.content if final else ""
    if isinstance(text, list):   # beberapa model mengembalikan list blok konten
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    return text, trace
