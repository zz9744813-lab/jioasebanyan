"""事件日志。追加 JSONL。"""
import json
from pathlib import Path
from ..models import ObjectiveEvent


class EventLog:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, turn: int, events: list[ObjectiveEvent]):
        with open(self.save_path, "a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps({"turn": turn, **ev.model_dump()},
                                   ensure_ascii=False) + "\n")

    def tail(self, n: int = 6) -> list[dict]:
        if not self.save_path.exists():
            return []
        with open(self.save_path, encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]

    def all_events(self) -> list[dict]:
        if not self.save_path.exists():
            return []
        with open(self.save_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f]