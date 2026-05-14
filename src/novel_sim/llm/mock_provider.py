"""测试用 mock LLM。"""
import json
from typing import Any, Callable, Awaitable
from .protocol import LLMProvider


class MockProvider(LLMProvider):
    def __init__(self, scripts: dict[str, list[str]] | None = None,
                 default_text: str = "[MOCK RESPONSE]",
                 default_json: dict[str, Any] | None = None):
        self.scripts = scripts or {}
        self._cursor: dict[str, int] = {}
        self.default_text = default_text
        self.default_json = default_json or {}
        self.call_log: list[dict[str, Any]] = []

    def _match(self, purpose: str) -> str | None:
        for prefix, responses in self.scripts.items():
            if purpose.startswith(prefix):
                idx = self._cursor.get(prefix, 0)
                if idx < len(responses):
                    self._cursor[prefix] = idx + 1
                    return responses[idx]
                return responses[-1]
        return None

    async def call(self, system, user, *, model, max_tokens=4096,
                   temperature=1.0, purpose="") -> str:
        self.call_log.append({"type": "text", "purpose": purpose,
                              "system": system[:200], "user": user[:200]})
        return self._match(purpose) or self.default_text

    async def call_json(self, system, user, *, model, max_tokens=4096,
                        temperature=1.0, purpose="") -> dict[str, Any]:
        text = await self.call(system, user, model=model, max_tokens=max_tokens,
                               temperature=temperature, purpose=purpose)
        try:
            return json.loads(text) if isinstance(text, str) else text
        except json.JSONDecodeError:
            return self.default_json

    async def call_with_tools(self, system, user, tools, tool_handler, *,
                              model, max_tokens=4096, temperature=1.0,
                              max_iterations=5, purpose="") -> str:
        return await self.call(system, user, model=model, max_tokens=max_tokens,
                               temperature=temperature, purpose=purpose)