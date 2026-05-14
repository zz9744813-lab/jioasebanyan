"""主 Agent - 初始化、POV 生成、行动裁定。"""
import json
from ..models import (WorldStateSnapshot, POVObservation, Adjudication,
                      CharacterPersona, WorldLocation)
from ..llm.protocol import LLMProvider
from ..prompts.loader import PromptLoader
from ..world.state import WorldStore
from ..world.event_log import EventLog
from ..settings import Settings
from ..logging_setup import get_logger

logger = get_logger(__name__)


class MasterAgent:
    def __init__(self, llm: LLMProvider, prompts: PromptLoader,
                 world: WorldStore, events: EventLog, settings: Settings):
        self.llm = llm
        self.prompts = prompts
        self.world = world
        self.events = events
        self.settings = settings

    async def initialize_from_outline(self, outline_text: str) -> dict:
        prompt = self.prompts.render("master_init", outline=outline_text)
        result = await self.llm.call_json(
            system="你是负责系统初始化的主 Agent。严格按要求输出 JSON。",
            user=prompt, model=self.settings.models.master_agent,
            max_tokens=8192, temperature=self.settings.llm.temperature,
            purpose="master.init")
        characters = {
            c["id"]: CharacterPersona(
                id=c["id"], name=c["name"], persona=c["persona"],
                location=c["initial_location"], status=c["initial_status"])
            for c in result["characters"]}
        locations = {
            k: WorldLocation(**v)
            for k, v in result["initial_state"].get("locations", {}).items()}
        self.world.initialize(
            world_rules=result["world_rules"],
            time=result["initial_state"].get("time", "初始"),
            locations=locations, characters=characters,
            environment=result["initial_state"].get("environment", {}))
        logger.info("master.initialized", char_count=len(characters))
        return result

    async def generate_pov_observation(self, char_id: str) -> POVObservation:
        char = self.world.get_char(char_id)
        if char is None:
            raise ValueError(f"unknown char_id {char_id}")
        snapshot = self.world.snapshot
        prompt = self.prompts.render(
            "master_pov", char_id=char_id, char_name=char.name,
            world_rules=snapshot.world_rules,
            world_state=snapshot.model_dump_json(indent=2),
            recent_events=json.dumps(self.events.tail(6), ensure_ascii=False, indent=2),
            char_location=char.location,
            char_profile=char.model_dump_json(indent=2))
        text = await self.llm.call(
            system="你是为子 Agent 生成 POV 观察的主 Agent。严格按 POV 视角描述。",
            user=prompt, model=self.settings.models.master_agent,
            max_tokens=self.settings.llm.max_tokens,
            temperature=self.settings.llm.temperature,
            purpose=f"master.pov.{char_id}.t{snapshot.turn}")
        return POVObservation(char_id=char_id, turn=snapshot.turn, text=text)

    async def adjudicate(self, turn: int, actions: dict) -> Adjudication:
        snapshot = self.world.snapshot
        prompt = self.prompts.render(
            "master_adjudicate", world_rules=snapshot.world_rules,
            world_state=snapshot.model_dump_json(indent=2),
            actions=json.dumps(actions, ensure_ascii=False, indent=2))
        result_dict = await self.llm.call_json(
            system="你是裁判主 Agent。完全客观，不带创作意图。",
            user=prompt, model=self.settings.models.master_agent,
            max_tokens=8192, temperature=self.settings.llm.temperature,
            purpose=f"master.adjudicate.t{turn}")
        adj = Adjudication.model_validate(result_dict)
        patch = dict(adj.world_state_patch)
        patch["turn"] = turn + 1
        self.world.apply_patch(patch)
        self.events.append(turn, adj.events)
        return adj