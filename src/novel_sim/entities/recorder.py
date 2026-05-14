"""记录者 - 阶段二上线。规划 + 写章 + 修订。"""
import json
from ..models import (NovelPlan, ChapterPlan, ReviewResult, TurnRecord,
                      Foreshadowing)
from ..llm.protocol import LLMProvider
from ..prompts.loader import PromptLoader
from ..settings import Settings


class Recorder:
    def __init__(self, llm: LLMProvider, prompts: PromptLoader, settings: Settings):
        self.llm = llm
        self.prompts = prompts
        self.settings = settings

    async def plan(self, world_rules: str, characters: dict,
                   chronicle: list[dict]) -> NovelPlan:
        chars_str = json.dumps(
            {cid: {"name": c.name, "status": c.status}
             for cid, c in characters.items()},
            ensure_ascii=False, indent=2)
        chron_str = "\n".join(
            f"{i}. （第 {e['turn']} 回合）{e['summary']}"
            for i, e in enumerate(chronicle))
        prompt = self.prompts.render(
            "recorder_plan", world_rules=world_rules,
            characters=chars_str, chronicle=chron_str)
        data = await self.llm.call_json(
            system="你是负责小说全局规划的记录者。严格按 JSON 输出。",
            user=prompt, model=self.settings.models.recorder,
            max_tokens=8192, temperature=self.settings.llm.temperature,
            purpose="recorder.plan")
        return NovelPlan(
            protagonist_id=data["protagonist"]["char_id"],
            protagonist_name=data["protagonist"]["name"],
            rationale=data["protagonist"]["rationale"],
            chapters=[ChapterPlan.model_validate(c) for c in data["chapters"]],
            foreshadowing_table=[Foreshadowing.model_validate(f)
                                  for f in data.get("foreshadowing_table", [])])

    async def write_chapter(self, chapter: ChapterPlan, plan: NovelPlan,
                            materials: list[TurnRecord],
                            prev_summaries: list[str]) -> str:
        prompt = self.prompts.render(
            "recorder_write", chapter_title=chapter.title,
            chapter_number=chapter.number,
            total_chapters=len(plan.chapters),
            protagonist_id=plan.protagonist_id,
            protagonist_name=plan.protagonist_name,
            plant_list=", ".join(chapter.plant_foreshadowing) or "无",
            recover_list=", ".join(chapter.recover_foreshadowing) or "无",
            materials=self._format_materials(materials),
            prev_summaries="\n".join(f"第 {i+1} 章: {s}"
                                      for i, s in enumerate(prev_summaries))
            or "（这是第一章，无前文）")
        return await self.llm.call(
            system="你是这部小说的记录者。文白夹杂古典字句 + 交叉剪辑。",
            user=prompt, model=self.settings.models.recorder,
            max_tokens=8192, temperature=self.settings.llm.temperature,
            purpose=f"recorder.write.ch{chapter.number}")

    async def revise(self, chapter: ChapterPlan, previous_draft: str,
                     feedback: ReviewResult,
                     materials: list[TurnRecord]) -> str:
        prompt = self.prompts.render(
            "recorder_revise", chapter_title=chapter.title,
            previous_draft=previous_draft,
            review_feedback=self._format_feedback(feedback),
            materials=self._format_materials(materials))
        return await self.llm.call(
            system="你是这部小说的记录者，正在根据审核反馈修订文字层。剧情不可改。",
            user=prompt, model=self.settings.models.recorder,
            max_tokens=8192, temperature=self.settings.llm.temperature,
            purpose=f"recorder.revise.ch{chapter.number}")

    @staticmethod
    def _format_materials(materials: list[TurnRecord]) -> str:
        out = []
        for tr in materials:
            out.append(f"\n=== 第 {tr.turn} 回合 ===")
            for ev in tr.objective_events:
                out.append(f"  [客观] {ev.description}")
            for cid, parts in tr.characters.items():
                out.append(f"  [{cid}] POV:{parts.observation[:200]} | "
                           f"内心:{parts.thinking[:300]} | 行动:{parts.action}")
        return "\n".join(out)

    @staticmethod
    def _format_feedback(r: ReviewResult) -> str:
        cn = {"literary": "文笔", "factual": "事实一致性",
              "world_rules": "世界规则", "coherence": "连贯性"}
        out = []
        for k in ("literary", "factual", "world_rules", "coherence"):
            d = getattr(r, k)
            if not d.passed:
                out.append(f"\n【{cn[k]}】未通过：")
                for iss in d.issues:
                    out.append(f"  - {iss}")
        return "\n".join(out) if out else "（无反馈）"