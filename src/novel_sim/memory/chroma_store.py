"""ChromaDB 实现主观摘要库。"""
from pathlib import Path
import chromadb
from ..models import RetrievedMemory


class ChromaSubjectiveStore:
    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))

    def _coll(self, char_id: str):
        return self.client.get_or_create_collection(
            name=f"subjective_{char_id}",
            metadata={"hnsw:space": "cosine"})

    def add(self, char_id: str, turn: int, memory_entry: str) -> None:
        self._coll(char_id).add(documents=[memory_entry],
                                ids=[f"{char_id}_t{turn}"],
                                metadatas=[{"turn": turn}])

    def retrieve(self, char_id: str, query: str, n: int = 5) -> list[RetrievedMemory]:
        col = self._coll(char_id)
        count = col.count()
        if count == 0:
            return []
        r = col.query(query_texts=[query], n_results=min(n, count))
        return [RetrievedMemory(turn=r["metadatas"][0][i]["turn"],
                                summary=r["documents"][0][i])
                for i in range(len(r["ids"][0]))]