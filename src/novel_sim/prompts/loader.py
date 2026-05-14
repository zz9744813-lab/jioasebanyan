"""提示词加载。jinja2 渲染，StrictUndefined 防漏变量。"""
from pathlib import Path
from functools import cache
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        self._prompts_dir = prompts_dir

    @cache
    def get_raw(self, name: str) -> str:
        path = self._prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, name: str, **kwargs) -> str:
        template = self.env.get_template(f"{name}.txt")
        return template.render(**kwargs)