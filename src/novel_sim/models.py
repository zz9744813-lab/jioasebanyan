"""所有跨模块流转的数据结构。Pydantic 模型。"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class CharacterPersona(BaseModel):
    """角色档案。persona 是常驻 system prompt。"""
    model_config = ConfigDict(frozen=False)
    id: str
    name: str
    persona: str
    location: str
    status: str
    inventory: list[str] = Field(default_factory=list)


class WorldLocation(BaseModel):
    description: str
    present_chars: list[str] = Field(default_factory=list)


class WorldStateSnapshot(BaseModel):
    turn: int = 0
    time: str = "初始"
    world_rules: str = ""
    locations: dict[str, WorldLocation] = Field(default_factory=dict)
    characters: dict[str, CharacterPersona] = Field(default_factory=dict)
    environment: dict[str, str | int | float] = Field(default_factory=dict)


class POVObservation(BaseModel):
    char_id: str
    turn: int
    text: str


class RetrievedMemory(BaseModel):
    turn: int
    summary: str


class SubAgentOutput(BaseModel):
    thinking: str
    action: str
    memory_entry: str


class ObjectiveEvent(BaseModel):
    description: str
    actors: list[str] = Field(default_factory=list)
    location: str = ""


class Adjudication(BaseModel):
    adjudications: dict[str, str] = Field(default_factory=dict)
    events: list[ObjectiveEvent] = Field(default_factory=list)
    world_state_patch: dict = Field(default_factory=dict)
    scene_ended: bool = False
    chronicle_entry: str | None = None


class Foreshadowing(BaseModel):
    id: str
    description: str
    planted_in_chapter: int
    recovered_in_chapter: int


class ChapterPlan(BaseModel):
    number: int
    title: str
    scene_indices: list[int]
    key_events: list[str]
    plant_foreshadowing: list[str] = Field(default_factory=list)
    recover_foreshadowing: list[str] = Field(default_factory=list)


class NovelPlan(BaseModel):
    protagonist_id: str
    protagonist_name: str
    rationale: str
    chapters: list[ChapterPlan]
    foreshadowing_table: list[Foreshadowing] = Field(default_factory=list)


class DimensionReview(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    literary: DimensionReview
    factual: DimensionReview
    world_rules: DimensionReview
    coherence: DimensionReview
    overall_passed: bool

    def failed_dimensions(self) -> list[str]:
        out = []
        for name in ("literary", "factual", "world_rules", "coherence"):
            if not getattr(self, name).passed:
                out.append(name)
        return out


class CharacterTurnRecord(BaseModel):
    char_id: str
    observation: str
    thinking: str
    action: str
    memory_entry: str


class TurnRecord(BaseModel):
    turn: int
    characters: dict[str, CharacterTurnRecord]
    objective_events: list[ObjectiveEvent]


class LLMCallTrace(BaseModel):
    timestamp: str
    purpose: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    system: str = ""
    user: str = ""
    response: str = ""
    error: str | None = None