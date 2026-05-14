"""测试记忆库读写循环。"""
import pytest
from novel_sim.memory.chroma_store import ChromaSubjectiveStore
from novel_sim.memory.jsonl_archive import JsonlVerbatimStore
from novel_sim.memory.retriever import Retriever
from novel_sim.models import CharacterTurnRecord


def test_subjective_write_retrieve(tmp_path):
    store = ChromaSubjectiveStore(tmp_path / "chroma")
    store.add("alice", 1, "在客栈初次见到黑衣人")
    store.add("alice", 2, "黑衣人说了一句奇怪的话")
    results = store.retrieve("alice", "黑衣人", n=5)
    assert len(results) >= 1


def test_verbatim_keyword(tmp_path):
    store = JsonlVerbatimStore(tmp_path / "verbatim")
    cr = CharacterTurnRecord(char_id="alice", observation="雨夜", thinking="疑心",
                             action="推门", memory_entry="记录")
    store.append("alice", cr, "雨夜入店")
    results = store.search_keyword("alice", "雨夜")
    assert len(results) == 1


def test_retriever_unified(tmp_path):
    sub = ChromaSubjectiveStore(tmp_path / "chroma")
    ver = JsonlVerbatimStore(tmp_path / "verbatim")
    r = Retriever(sub, ver)
    sub.add("alice", 1, "见到黑衣人")
    cr = CharacterTurnRecord(char_id="alice", observation="...", thinking="...",
                             action="...", memory_entry="见到黑衣人")
    ver.append("alice", cr, "...")
    auto = r.auto_retrieve("alice", "陌生人", n=5)
    assert len(auto) >= 1
    verbatim = r.recall_verbatim("alice", "黑衣人")
    assert len(verbatim) == 1