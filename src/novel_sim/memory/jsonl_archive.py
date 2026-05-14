"""每角色原文档案，JSONL 存储。"""
import json
from pathlib import Path
from ..models import CharacterTurnRecord


class JsonlVerbatimStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, char_id: str) -> Path:
        return self.data_dir / f"{char_id}.jsonl"

    def append(self, char_id: str, record: CharacterTurnRecord,
               observation: str) -> None:
        payload = {
            "turn": record.turn if hasattr(record, "turn") else 0,
            "observation": observation,
            "thinking": record.thinking,
            "action": record.action,
            "memory_entry": record.memory_entry,
        }
        with open(self._path(char_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def search_keyword(self, char_id: str, keyword: str,
                       n: int = 5) -> list[dict]:
        path = self._path(char_id)
        if not path.exists():
            return []
        hits = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if keyword in " ".join(str(v) for v in r.values()):
                    hits.append(r)
        return hits[-n:]