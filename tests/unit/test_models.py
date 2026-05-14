"""验证 Pydantic 模型 schema。"""
import pytest
from novel_sim.models import (SubAgentOutput, ReviewResult, DimensionReview,
                              Adjudication, ChapterPlan, NovelPlan)


def test_sub_agent_output():
    out = SubAgentOutput(thinking="想", action="做", memory_entry="记")
    assert out.thinking == "想"


def test_review_result_failed_dimensions():
    r = ReviewResult(
        literary=DimensionReview(passed=False, issues=["文笔差"]),
        factual=DimensionReview(passed=True),
        world_rules=DimensionReview(passed=True),
        coherence=DimensionReview(passed=False, issues=["不连贯"]),
        overall_passed=False)
    assert set(r.failed_dimensions()) == {"literary", "coherence"}


def test_adjudication_strict():
    with pytest.raises(Exception):
        Adjudication.model_validate({"wrong": "schema"})


def test_all_passed():
    r = ReviewResult(
        literary=DimensionReview(passed=True),
        factual=DimensionReview(passed=True),
        world_rules=DimensionReview(passed=True),
        coherence=DimensionReview(passed=True),
        overall_passed=True)
    assert r.failed_dimensions() == []