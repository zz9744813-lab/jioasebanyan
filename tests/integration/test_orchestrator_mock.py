"""用 mock LLM 跑完整流程。"""
import pytest
from pathlib import Path
import yaml


@pytest.mark.asyncio
async def test_orchestrator_with_mock_llm(tmp_settings, mock_llm, tmp_path):
    from novel_sim.orchestrator import Orchestrator
    from novel_sim.prompts.loader import PromptLoader
    from novel_sim.world.state import WorldStore
    from novel_sim.world.event_log import EventLog
    from novel_sim.world.chronicle import Chronicle
    from novel_sim.world.central_archive import CentralArchive
    from novel_sim.memory.chroma_store import ChromaSubjectiveStore
    from novel_sim.memory.jsonl_archive import JsonlVerbatimStore

    scenario = tmp_path / "minimal.yaml"
    scenario.write_text("""
title: test
characters:
  - name: A
  - name: B
""", encoding="utf-8")

    tmp_settings.ensure_dirs()
    orch = Orchestrator(
        settings=tmp_settings, llm=mock_llm,
        prompts=PromptLoader(tmp_settings.prompts_dir),
        world=WorldStore(tmp_settings.storage.data_dir / "world_state.json"),
        events=EventLog(tmp_settings.storage.data_dir / "event_log.jsonl"),
        chronicle=Chronicle(tmp_settings.storage.data_dir / "chronicle.jsonl"),
        archive=CentralArchive(tmp_settings.storage.data_dir / "archive" / "raw.jsonl"),
        subjective=ChromaSubjectiveStore(
            tmp_settings.storage.chroma_dir, backend="deterministic"),
        verbatim=JsonlVerbatimStore(tmp_settings.storage.data_dir / "archive" / "per_char"))
    await orch.initialize(scenario)
    assert orch.world.turn == 0
    assert set(orch.sub_agents.keys()) == {"alice", "bob"}
    await orch.run_turn()
    assert orch.world.turn == 1
    assert len(orch.archive.get_turns(0, 0)) == 1
