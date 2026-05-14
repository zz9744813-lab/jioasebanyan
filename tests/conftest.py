"""全局测试 fixtures。"""
import asyncio
import inspect
import pytest
from pathlib import Path
from novel_sim.settings import Settings


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    import yaml
    cfg = {
        "anthropic_api_key": "sk-test",
        "models": {"master_agent": "claude-opus-4-5",
                    "sub_agent": "claude-sonnet-4-6",
                    "recorder": "claude-opus-4-5",
                    "reviewer": "claude-sonnet-4-6"},
        "storage": {"data_dir": str(tmp_path / "data"),
                    "output_dir": str(tmp_path / "outputs"),
                    "chroma_dir": str(tmp_path / "data" / "chroma"),
                    "trace_dir": str(tmp_path / "data" / "traces")},
        "prompts_dir": "./prompts",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return Settings(**cfg)


@pytest.fixture
def mock_llm():
    from novel_sim.llm.mock_provider import MockProvider
    return MockProvider(scripts={
        "master.init": ['{"world_rules": "测试规则", "initial_state": {'
                        '"time": "测试时间", "locations": {'
                        '"test_loc": {"description": "...", "present_chars": ["alice", "bob"]}'
                        '}, "environment": {}}, "characters": ['
                        '{"id": "alice", "name": "爱丽丝", "persona": "测试人设A",'
                        ' "initial_location": "test_loc", "initial_status": "正常"},'
                        '{"id": "bob", "name": "鲍勃", "persona": "测试人设B",'
                        ' "initial_location": "test_loc", "initial_status": "正常"}]}'],
        "master.pov": ["测试 POV 观察"],
        "master.adjudicate": ['{"adjudications": {"alice": "ok"},'
                              '"events": [], "world_state_patch": {},'
                              '"scene_ended": false, "chronicle_entry": null}'],
        "sub.": ['{"thinking": "思考", "action": "行动", "memory_entry": "记忆"}'],
    })


def pytest_pyfunc_call(pyfuncitem):
    """在缺少 pytest-asyncio 插件时兜底执行 async 测试。"""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {name: pyfuncitem.funcargs[name]
                  for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None
