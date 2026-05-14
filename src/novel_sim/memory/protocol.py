"""记忆库抽象接口。"""
from typing import Protocol, runtime_checkable
from ..models import RetrievedMemory, CharacterTurnRecord


@runtime_checkable
class SubjectiveStore(Protocol):
    def add(self, char_id: str, turn: int, memory_entry: str) -> None: ...
    def retrieve(self, char_id: str, query: str, n: int = 5) -> list[RetrievedMemory]: ...


@runtime_checkable
class VerbatimStore(Protocol):
    def append(self, char_id: str, record: CharacterTurnRecord,
               observation: str) -> None: ...
    def search_keyword(self, char_id: str, keyword: str,
                       n: int = 5) -> list[dict]: ...