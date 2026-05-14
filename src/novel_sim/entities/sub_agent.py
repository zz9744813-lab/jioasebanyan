"""子 Agent - 每个角色独立实例，POV 隔离。"""
import json
from ..models import POVObservation, SubAgentOutput, RetrievedMemory
from ..llm.protocol import LLMProvider, LLMResponseError
from ..llm.anthropic_provider import _extract_json
from ..prompts.loader import PromptLoader
from ..memory.retriever import Retriever
from ..settings import Settings


class SubAgent:
    def __init__(self, char_id: str, persona: str, llm: LLMProvider,
                 prompts: PromptLoader, retriever: Retriever, settings: Settings):
        self.char_id = char_id
        self.persona = persona
        self.llm = llm
        self.prompts = prompts
        self.retriever = retriever
        self.settings = settings

    async def run_turn(self, obs: POVObservation) -> SubAgentOutput:
        retrieved = self.retriever.auto_retrieve(
            self.char_id, obs.text, self.settings.retrieval.auto_retrieve_n)
        system = self.prompts.render("sub_agent_system", persona=self.persona)
        user = self.prompts.render(
            "sub_agent_user",
            turn=obs.turn,
            observation=obs.text,
            memories=self._fmt_memories(retrieved))
        tools = self._tool_specs()
        try:
            response_text = await self.llm.call_with_tools(
                system=system, user=user, tools=tools,
                tool_handler=self._handle_tool,
                model=self.settings.models.sub_agent,
                max_tokens=self.settings.llm.max_tokens,
                temperature=self.settings.llm.temperature,
                max_iterations=5,
                purpose=f"sub.{self.char_id}.t{obs.turn}")
        except LLMResponseError:
            response_text = await self.llm.call(
                system=system, user=user,
                model=self.settings.models.sub_agent,
                max_tokens=self.settings.llm.max_tokens,
                temperature=self.settings.llm.temperature,
                purpose=f"sub.{self.char_id}.t{obs.turn}.fallback")
        data = _extract_json(response_text)
        return SubAgentOutput.model_validate(data)

    @staticmethod
    def _fmt_memories(memories: list[RetrievedMemory]) -> str:
        if not memories:
            return "（暂无相关记忆——这件事对你可能是新的）"
        return "\n".join(f"- 第 {m.turn} 回合左右：{m.summary}"
                         for m in memories)

    @staticmethod
    def _tool_specs() -> list[dict]:
        return [
            {"name": "recall_subjective",
             "description": "从你的主观记忆里按语义检索过去事件（可能不准确）",
             "input_schema": {"type": "object",
                              "properties": {"query": {"type": "string"}},
                              "required": ["query"]}},
            {"name": "recall_verbatim",
             "description": "从原文档案查询当时究竟说了/做了什么。严肃求证用。",
             "input_schema": {"type": "object",
                              "properties": {"query": {"type": "string"}},
                              "required": ["query"]}}]

    async def _handle_tool(self, name: str, inp: dict) -> str:
        query = inp.get("query", "")
        if name == "recall_subjective":
            r = self.retriever.recall_subjective(self.char_id, query)
            return self._fmt_memories(r) if r else "（无相关记忆）"
        elif name == "recall_verbatim":
            r = self.retriever.recall_verbatim(self.char_id, query)
            if not r:
                return "（档案无匹配）"
            return "\n".join(
                f"[第{x['turn']}回合] 观察:{x['observation'][:100]} | 行动:{x['action'][:100]}"
                for x in r)
        return f"未知工具: {name}"
