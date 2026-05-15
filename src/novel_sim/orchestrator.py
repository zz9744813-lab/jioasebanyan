"""异步主流程。子 Agent 并发跑。"""
import asyncio
import json
from difflib import SequenceMatcher
from pathlib import Path

from .models import (SubAgentOutput, CharacterTurnRecord, TurnRecord,
                     POVObservation, ReviewResult, ChapterPlan, NovelPlan)
from .llm.protocol import LLMProvider
from .prompts.loader import PromptLoader
from .settings import Settings
from .world.state import WorldStore
from .world.event_log import EventLog
from .world.chronicle import Chronicle
from .world.central_archive import CentralArchive
from .memory.protocol import SubjectiveStore, VerbatimStore
from .memory.retriever import Retriever
from .entities.master_agent import MasterAgent
from .entities.sub_agent import SubAgent
from .entities.recorder import Recorder
from .entities.reviewer import Reviewer
from .logging_setup import get_logger
from .events import emit

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, *, settings: Settings, llm: LLMProvider,
                 prompts: PromptLoader, world: WorldStore, events: EventLog,
                 chronicle: Chronicle, archive: CentralArchive,
                 subjective: SubjectiveStore, verbatim: VerbatimStore):
        self.settings = settings
        self.world = world
        self.events = events
        self.chronicle = chronicle
        self.archive = archive
        self.subjective = subjective
        self.verbatim = verbatim
        retriever = Retriever(subjective, verbatim)
        self.master = MasterAgent(llm, prompts, world, events, settings)
        self.recorder = Recorder(llm, prompts, settings)
        self.reviewer = Reviewer(llm, prompts, settings)
        self.sub_agents: dict[str, SubAgent] = {}
        self._sub_factory = lambda cid, p: SubAgent(
            cid, p, llm, prompts, retriever, settings)

    async def initialize(self, scenario_path: Path):
        import yaml
        with open(scenario_path, encoding="utf-8") as f:
            scenario = yaml.safe_load(f)
        outline_text = yaml.dump(scenario, allow_unicode=True, sort_keys=False)
        await self.master.initialize_from_outline(outline_text)
        for cid in self.world.char_ids():
            char = self.world.get_char(cid)
            assert char is not None
            self.sub_agents[cid] = self._sub_factory(cid, char.persona)
        logger.info("init.done", chars=list(self.sub_agents.keys()))

    def attach_existing(self):
        for cid in self.world.char_ids():
            char = self.world.get_char(cid)
            assert char is not None
            self.sub_agents[cid] = self._sub_factory(cid, char.persona)

    async def run_turn(self):
        turn = self.world.turn
        emit("turn.started", turn=turn)
        logger.info("turn.start", turn=turn)
        char_ids = list(self.sub_agents.keys())
        if self.settings.concurrency.sub_agent_parallel:
            observations = await asyncio.gather(*[
                self.master.generate_pov_observation(cid) for cid in char_ids])
        else:
            observations = [await self.master.generate_pov_observation(cid)
                            for cid in char_ids]
        obs_map = {o.char_id: o for o in observations}
        emit("master.observation", turn=turn,
             observations={cid: obs_map[cid].text for cid in char_ids})
        if self.settings.concurrency.sub_agent_parallel:
            outputs = await asyncio.gather(*[
                self.sub_agents[cid].run_turn(obs_map[cid]) for cid in char_ids])
        else:
            outputs = [await self.sub_agents[cid].run_turn(obs_map[cid])
                       for cid in char_ids]
        output_map: dict[str, SubAgentOutput] = dict(zip(char_ids, outputs))
        for cid, out in output_map.items():
            self.subjective.add(cid, turn, out.memory_entry)
            ct = CharacterTurnRecord(char_id=cid, observation=obs_map[cid].text,
                                     thinking=out.thinking, action=out.action,
                                     memory_entry=out.memory_entry)
            self.verbatim.append(cid, turn, ct, obs_map[cid].text)
        actions_for_judge = {cid: {"thinking": out.thinking, "action": out.action}
                             for cid, out in output_map.items()}
        adj = await self.master.adjudicate(turn, actions_for_judge)
        emit("master.adjudication",
             turn=turn,
             adjudications=adj.adjudications,
             events=[e.model_dump() for e in adj.events],
             world_state_patch=adj.world_state_patch,
             scene_ended=adj.scene_ended)
        turn_record = TurnRecord(
            turn=turn,
            characters={cid: CharacterTurnRecord(
                char_id=cid, observation=obs_map[cid].text,
                thinking=out.thinking, action=out.action,
                memory_entry=out.memory_entry)
                for cid, out in output_map.items()},
            objective_events=adj.events)
        self.archive.record(turn_record)
        if adj.scene_ended and adj.chronicle_entry:
            self.chronicle.add(turn, adj.chronicle_entry)
            logger.info("scene.ended", turn=turn, summary=adj.chronicle_entry)
        logger.info("turn.done", turn=turn, next=turn + 1)
        emit("turn.done", turn=turn)
        return adj

    async def writing_phase(self) -> Path | None:
        emit("writing.started")
        chronicle_entries = self.chronicle.all()
        if not chronicle_entries:
            logger.warning("writing.no_chronicle")
            return None
        snapshot = self.world.snapshot
        plan = await self.recorder.plan(
            world_rules=snapshot.world_rules,
            characters=snapshot.characters, chronicle=chronicle_entries)
        (self.settings.storage.output_dir / "plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("writing.plan_ready", chapters=len(plan.chapters),
                    protagonist=plan.protagonist_name)
        novel_path = self.settings.storage.output_dir / "novel.md"
        chapters_md = [f"# 小说\n\n*主角: {plan.protagonist_name}*\n"]
        prev_summaries: list[str] = []
        for ch in plan.chapters:
            logger.info("chapter.start", number=ch.number, title=ch.title)
            scene_indices = ch.scene_indices
            turn_ranges = self._scene_to_turn_ranges(scene_indices, chronicle_entries)
            materials: list = []
            for start, end in turn_ranges:
                materials.extend(self.archive.get_turns(start, end))
            if not materials:
                logger.warning("chapter.no_materials", number=ch.number)
                continue
            draft = await self._write_chapter_with_loop(ch, plan, materials, prev_summaries)
            if draft is None:
                continue
            chapters_md.append(f"\n## {ch.title}\n\n{draft}\n")
            prev_summaries.append(draft[:200].replace("\n", " ") + "...")
            novel_path.write_text("\n".join(chapters_md), encoding="utf-8")
            logger.info("chapter.done", number=ch.number)
        chron_md = self.settings.storage.output_dir / "chronicle.md"
        chron_md.write_text(self.chronicle.to_markdown(), encoding="utf-8")
        logger.info("writing.complete")
        emit("writing.done", final_draft=novel_path.read_text(encoding="utf-8") if novel_path.exists() else "",
             iterations=0, terminated_reason="completed")
        return novel_path

    async def _write_chapter_with_loop(self, ch: ChapterPlan, plan: NovelPlan,
                                        materials: list, prev_summaries: list[str]
                                        ) -> str | None:
        loop_history: list[str] = []
        draft = await self.recorder.write_chapter(ch, plan, materials, prev_summaries)
        attempt = 0
        snapshot = self.world.snapshot
        while True:
            attempt += 1
            emit("recorder.draft", iteration=attempt, draft=draft)
            review = await self.reviewer.review(
                draft=draft, world_rules=snapshot.world_rules,
                materials=materials, prev_summaries=prev_summaries,
                chapter_number=ch.number)
            if review.overall_passed:
                emit("reviewer.result", iteration=attempt,
                     literary=review.literary.model_dump(),
                     factual=review.factual.model_dump(),
                     world_rules=review.world_rules.model_dump(),
                     coherence=review.coherence.model_dump(),
                     overall_passed=True)
                logger.info("chapter.passed", number=ch.number, attempts=attempt)
                return draft
            emit("reviewer.result", iteration=attempt,
                 literary=review.literary.model_dump(),
                 factual=review.factual.model_dump(),
                 world_rules=review.world_rules.model_dump(),
                 coherence=review.coherence.model_dump(),
                 overall_passed=False)
            failed = review.failed_dimensions()
            logger.info("chapter.review_failed", number=ch.number,
                        attempt=attempt, dimensions=failed)
            feedback_str = review.model_dump_json()
            loop_history.append(feedback_str)
            if self._is_stuck(loop_history):
                logger.warning("chapter.stuck", number=ch.number)
                emit("writing.done", final_draft=draft, iterations=attempt,
                     terminated_reason="stuck")
                return draft
            draft = await self.recorder.revise(ch, draft, review, materials)

    def _is_stuck(self, history: list[str]) -> bool:
        cfg = self.settings.review_loop
        if len(history) < cfg.window:
            return False
        recent = history[-cfg.window:]
        for i in range(len(recent) - 1):
            if SequenceMatcher(None, recent[i], recent[i+1]).ratio() < cfg.similarity_threshold:
                return False
        return True

    @staticmethod
    def _scene_to_turn_ranges(scene_indices: list[int],
                              chronicle_entries: list[dict]) -> list[tuple[int, int]]:
        ranges = []
        for i in scene_indices:
            if i >= len(chronicle_entries):
                continue
            start = chronicle_entries[i]["turn"]
            end = (chronicle_entries[i+1]["turn"] - 1
                   if i + 1 < len(chronicle_entries) else 10**9)
            ranges.append((start, end))
        return ranges
