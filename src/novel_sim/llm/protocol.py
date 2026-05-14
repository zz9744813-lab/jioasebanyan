"""LLM 抽象接口。"""
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def call(self, system: str, user: str, *, model: str,
                   max_tokens: int = 4096, temperature: float = 1.0,
                   purpose: str = "") -> str: ...
    async def call_json(self, system: str, user: str, *, model: str,
                        max_tokens: int = 4096, temperature: float = 1.0,
                        purpose: str = "") -> dict[str, Any]: ...
    async def call_with_tools(self, system: str, user: str,
                              tools: list[dict[str, Any]],
                              tool_handler: Callable[[str, dict[str, Any]],
                              Awaitable[str]], *, model: str,
                              max_tokens: int = 4096, temperature: float = 1.0,
                              max_iterations: int = 5, purpose: str = "") -> str: ...


class LLMResponseError(Exception):
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response