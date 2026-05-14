"""OpenAI 兼容 provider — 用于 WhiteDream 等代理。"""
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import LLMCallTrace
from ..logging_setup import get_logger
from .protocol import LLMProvider, LLMResponseError
from .tracing import TraceWriter

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, trace_writer: TraceWriter, base_url: str, api_key: str,
                 max_retries: int = 3):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.trace = trace_writer
        self.max_retries = max_retries

    async def call(self, system, user, *, model, max_tokens=4096,
                   temperature=1.0, purpose="") -> str:
        return await self._call_text(system, user, model=model,
                                     max_tokens=max_tokens,
                                     temperature=temperature, purpose=purpose)

    async def call_json(self, system, user, *, model, max_tokens=4096,
                        temperature=1.0, purpose="") -> dict[str, Any]:
        text = await self._call_text(system, user, model=model,
                                     max_tokens=max_tokens,
                                     temperature=temperature, purpose=purpose)
        try:
            return _extract_json(text)
        except ValueError as e:
            raise LLMResponseError(str(e), raw_response=text) from e

    async def call_with_tools(self, system, user, tools, tool_handler, *,
                              model, max_tokens=4096, temperature=1.0,
                              max_iterations=5, purpose="") -> str:
        tool_schemas = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema", {})}} for t in tools]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        for iteration in range(max_iterations):
            resp = await self._raw_call(
                messages=messages, tools=tool_schemas, model=model,
                max_tokens=max_tokens, temperature=temperature,
                purpose=f"{purpose}#iter{iteration}")
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments}}
                             for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await tool_handler(tc.function.name, args)
                except Exception as e:
                    result = f"工具调用错误: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)})
        raise LLMResponseError(f"Tool loop exceeded {max_iterations} iterations")

    async def _call_text(self, system, user, *, model, max_tokens,
                         temperature, purpose) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        resp = await self._raw_call(
            messages=messages, tools=None, model=model, max_tokens=max_tokens,
            temperature=temperature, purpose=purpose)
        content = resp.choices[0].message.content
        if not content:
            raise LLMResponseError("Empty response", raw_response=str(resp))
        return content

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def _raw_call(self, *, messages, tools, model, max_tokens,
                        temperature, purpose):
        start = time.perf_counter()
        kwargs = {"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        error_str: str | None = None
        resp = None
        try:
            resp = await self.client.chat.completions.create(**kwargs)
            return resp
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            trace = LLMCallTrace(
                timestamp=datetime.now(timezone.utc).isoformat(),
                purpose=purpose, model=model,
                input_tokens=getattr(resp.usage, "prompt_tokens", 0) if resp else 0,
                output_tokens=getattr(resp.usage, "completion_tokens", 0) if resp else 0,
                duration_ms=duration_ms, system=str(messages[0]["content"])[:2000] if messages else "",
                user=str(messages[-1]["content"] if messages else "")[:4000],
                response=resp.choices[0].message.content[:4000] if resp else "",
                error=error_str)
            self.trace.write(trace)

    async def close(self):
        await self.client.close()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse failed: {e}")
        raise ValueError(f"No JSON found in response")