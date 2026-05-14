"""剧情大事记。每场景一条 JSONL。"""
import json
from pathlib import Path


class Chronicle:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, turn: int, summary: str):
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"turn": turn, "summary": summary},
                               ensure_ascii=False) + "\n")

    def all(self) -> list[dict]:
        if not self.save_path.exists():
            return []
        with open(self.save_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    def to_markdown(self) -> str:
        lines = ["# 剧情大事记\n"]
        for i, e in enumerate(self.all(), 1):
            lines.append(f"{i}. （第 {e['turn']} 回合）{e['summary']}")
        return "\n".join(lines)