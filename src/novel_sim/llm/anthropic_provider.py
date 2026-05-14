"""真实 Anthropic 实现。带重试、tracing、JSON 容错。"""
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import anthropic
from anthropic import AsyncAnthropic, APIError, APITimeoutError, RateLimitError
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type)

from ..models import LLMCallTrace
from ..logging_setup import get_logger
from .protocol import LLMProvider, LLMResponseError
from .tracing import TraceWriter

logger = get_logger(__name__)
_RETRYABLE = (RateLimitError, APITimeoutError, APIError)


class AnthropicProvider(LLMProvider):
    def __init__(self, trace_writer: TraceWriter, max_retries: int = 3):
        self.client = AsyncAnthropic()
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
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for iteration in range(max_iterations):
            resp = await self._raw_call(
                system=system, messages=messages, tools=tools,
                model=model, max_tokens=max_tokens, temperature=temperature,
                purpose=f"{purpose}#iter{iteration}")
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                texts = [b.text for b in resp.content if hasattr(b, "text")]
                return "\n".join(texts)
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        result = await tool_handler(block.name, block.input)
                    except Exception as e:
                        result = f"工具调用错误: {e}"
                    tool_results.append({"type": "tool_result",
                                         "tool_use_id": block.id,
                                         "content": str(result)})
            messages.append({"role": "user", "content": tool_results})
        raise LLMResponseError(f"Tool loop exceeded {max_iterations} iterations")

    async def _call_text(self, system, user, *, model, max_tokens,
                         temperature, purpose) -> str:
        resp = await self._raw_call(
            system=system, messages=[{"role": "user", "content": user}],
            tools=None, model=model, max_tokens=max_tokens,
            temperature=temperature, purpose=purpose)
        texts = [b.text for b in resp.content if hasattr(b, "text")]
        if not texts:
            raise LLMResponseError("Empty response", raw_response=str(resp))
        return "\n".join(texts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30),
           retry=retry_if_exception_type(_RETRYABLE), reraise=True)
    async def _raw_call(self, *, system, messages, tools, model,
                        max_tokens, temperature, purpose):
        start = time.perf_counter()
        kwargs = {"model": model, "max_tokens": max_tokens,
                  "temperature": temperature, "system": system,
                  "messages": messages}
        if tools:
            kwargs["tools"] = tools
        error_str: str | None = None
        resp = None
        try:
            resp = await self.client.messages.create(**kwargs)
            return resp
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            trace = LLMCallTrace(
                timestamp=datetime.now(timezone.utc).isoformat(),
                purpose=purpose, model=model,
                input_tokens=getattr(resp.usage, "input_tokens", 0) if resp else 0,
                output_tokens=getattr(resp.usage, "output_tokens", 0) if resp else 0,
                duration_ms=duration_ms, system=system[:2000],
                user=str(messages)[:4000],
                response=str(resp.content)[:4000] if resp else "",
                error=error_str)
            self.trace.write(trace)


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