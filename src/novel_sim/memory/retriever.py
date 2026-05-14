"""统一检索门面。依赖 Protocol，能注入 mock。"""
from ..models import RetrievedMemory
from .protocol import SubjectiveStore, VerbatimStore


class Retriever:
    def __init__(self, subjective: SubjectiveStore, verbatim: VerbatimStore):
        self.subjective = subjective
        self.verbatim = verbatim

    def auto_retrieve(self, char_id: str, query: str,
                      n: int = 5) -> list[RetrievedMemory]:
        return self.subjective.retrieve(char_id, query, n)

    def recall_subjective(self, char_id: str, query: str,
                          n: int = 5) -> list[RetrievedMemory]:
        return self.subjective.retrieve(char_id, query, n)

    def recall_verbatim(self, char_id: str, query: str,
                        n: int = 5) -> list[dict]:
        return self.verbatim.search_keyword(char_id, query, n)