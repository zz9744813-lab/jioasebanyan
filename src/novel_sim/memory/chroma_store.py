"""ChromaDB 实现主观摘要库。"""
from pathlib import Path
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from ..models import RetrievedMemory


class _DeterministicEmbedding(EmbeddingFunction[Documents]):
    """离线可用 embedding，避免测试时下载默认 ONNX 模型。"""
    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "deterministic_test"

    def get_config(self) -> dict:
        return {"backend": "deterministic"}

    def __call__(self, input: Documents) -> Embeddings:
        out: Embeddings = []
        for text in input:
            vec = [0.0] * 8
            for i, b in enumerate(text.encode("utf-8")):
                vec[i % 8] += float(b)
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class _SentenceTransformerEmbedding(EmbeddingFunction[Documents]):
    """生产用 sentence-transformers embedding。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._name = f"st_{model_name}"

    def name(self) -> str:
        return self._name

    def get_config(self) -> dict:
        return {"backend": "sentence_transformers", "model": self._name}

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), convert_to_numpy=True).tolist()


def _build_embedding(backend: str, model_name: str) -> EmbeddingFunction[Documents]:
    if backend == "deterministic":
        return _DeterministicEmbedding()
    return _SentenceTransformerEmbedding(model_name)


class ChromaSubjectiveStore:
    def __init__(self, persist_dir: Path,
                 backend: str = "sentence_transformers",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_fn = _build_embedding(backend, model_name)

    def _coll(self, char_id: str):
        return self.client.get_or_create_collection(
            name=f"subjective_{char_id}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn)

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
