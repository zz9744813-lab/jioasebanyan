"""每次 LLM 调用写一行 JSONL trace。"""
import json
from pathlib import Path
from threading import Lock
from ..models import LLMCallTrace


class TraceWriter:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, trace: LLMCallTrace):
        date = trace.timestamp[:10]
        path = self.trace_dir / f"{date}.jsonl"
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(trace.model_dump_json() + "\n")

    def query(self, purpose_contains: str = "", date: str | None = None) -> list[LLMCallTrace]:
        results = []
        files = ([self.trace_dir / f"{date}.jsonl"] if date
                 else sorted(self.trace_dir.glob("*.jsonl")))
        for path in files:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = LLMCallTrace.model_validate_json(line)
                    if purpose_contains in t.purpose:
                        results.append(t)
        return results