"""审核 - 四维独立判定，完全中立。"""
from ..models import ReviewResult, ChapterPlan, TurnRecord
from ..llm.protocol import LLMProvider
from ..prompts.loader import PromptLoader
from ..settings import Settings
from .recorder import Recorder


class Reviewer:
    def __init__(self, llm: LLMProvider, prompts: PromptLoader, settings: Settings):
        self.llm = llm
        self.prompts = prompts
        self.settings = settings

    async def review(self, draft: str, world_rules: str,
                     materials: list[TurnRecord],
                     prev_summaries: list[str],
                     chapter_number: int) -> ReviewResult:
        prompt = self.prompts.render(
            "reviewer", world_rules=world_rules,
            materials=Recorder._format_materials(materials),
            prev_summaries="\n".join(
                f"第 {i+1} 章: {s}"
                for i, s in enumerate(prev_summaries)) or "（这是第一章）",
            draft=draft)
        data = await self.llm.call_json(
            system="你是完全中立的审核员。独立判定四个维度，不带创作偏好。",
            user=prompt, model=self.settings.models.reviewer,
            max_tokens=4096,
            temperature=self.settings.llm.reviewer_temperature,
            purpose=f"reviewer.ch{chapter_number}")
        return ReviewResult.model_validate(data)