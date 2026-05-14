"""中央素材库 - 记录者写章时的 RAG 源。"""
import json
from pathlib import Path
from ..models import TurnRecord


class CentralArchive:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: TurnRecord):
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def get_turns(self, start: int, end: int) -> list[TurnRecord]:
        if not self.save_path.exists():
            return []
        results = []
        with open(self.save_path, encoding="utf-8") as f:
            for line in f:
                r = TurnRecord.model_validate_json(line)
                if start <= r.turn <= end:
                    results.append(r)
        return results