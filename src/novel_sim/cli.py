"""CLI 入口 - typer + rich。"""
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .settings import Settings
from .logging_setup import configure_logging, get_logger
from .checkpoints import CheckpointManager
from .orchestrator import Orchestrator
from .llm.anthropic_provider import AnthropicProvider
from .llm.tracing import TraceWriter
from .prompts.loader import PromptLoader
from .world.state import WorldStore
from .world.event_log import EventLog
from .world.chronicle import Chronicle
from .world.central_archive import CentralArchive
from .memory.chroma_store import ChromaSubjectiveStore
from .memory.jsonl_archive import JsonlVerbatimStore

app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)
console = Console()
logger = get_logger()


def _build_orchestrator(settings: Settings) -> Orchestrator:
    settings.ensure_dirs()
    trace = TraceWriter(settings.storage.trace_dir)
    if settings.provider.type == "openai":
        from .llm.openai_provider import OpenAICompatibleProvider
        llm = OpenAICompatibleProvider(
            trace, base_url=settings.provider.base_url,
            api_key=settings.provider.api_key,
            max_retries=settings.llm.max_retries)
    else:
        llm = AnthropicProvider(trace, max_retries=settings.llm.max_retries)
    prompts = PromptLoader(settings.prompts_dir)
    world = WorldStore(settings.storage.data_dir / "world_state.json")
    events = EventLog(settings.storage.data_dir / "event_log.jsonl")
    chronicle = Chronicle(settings.storage.data_dir / "chronicle.jsonl")
    archive = CentralArchive(settings.storage.data_dir / "archive" / "raw.jsonl")
    subj = ChromaSubjectiveStore(settings.storage.chroma_dir)
    verb = JsonlVerbatimStore(settings.storage.data_dir / "archive" / "per_char")
    return Orchestrator(
        settings=settings, llm=llm, prompts=prompts, world=world,
        events=events, chronicle=chronicle, archive=archive,
        subjective=subj, verbatim=verb)


@app.command()
def run(scenario: Path = typer.Option(..., help="场景 YAML 路径"),
        config: Path = typer.Option(Path("config.yaml")),
        reset: bool = typer.Option(False, help="清空数据重新开始"),
        log_level: str = typer.Option("INFO")):
    """启动一次模拟会话。"""
    configure_logging(log_level)
    settings = Settings.load(str(config))
    cm = CheckpointManager(settings)
    if reset:
        cm.reset()
        console.print("[yellow]已清空所有数据[/yellow]")
    if cm.has_existing_session() and not reset:
        info = cm.session_info()
        console.print(f"[cyan]检测到已有会话: 第 {info['turn']} 回合, "
                      f"{info['chronicle_entries']} 个场景[/cyan]")
        choice = Prompt.ask("选择", choices=["resume", "reset", "abort"], default="resume")
        if choice == "abort":
            raise typer.Exit(0)
        elif choice == "reset":
            cm.reset()
    orch = _build_orchestrator(settings)
    if cm.has_existing_session():
        orch.attach_existing()
        console.print(f"[green]恢复会话 (第 {orch.world.turn} 回合)[/green]")
    else:
        console.print(f"[cyan]初始化: {scenario}[/cyan]")
        asyncio.run(orch.initialize(scenario))
    _simulation_loop(orch)
    if Confirm.ask("进入成稿期？"):
        novel_path = asyncio.run(orch.writing_phase())
        if novel_path:
            console.print(f"[green]完成: {novel_path}[/green]")


def _simulation_loop(orch: Orchestrator):
    console.print("[bold cyan]模拟期 · 命令: Enter=下一回合 c=大事记 s=状态 q=成稿[/bold cyan]")
    while True:
        cmd = Prompt.ask(">>>", default="").strip().lower()
        if cmd == "q":
            return
        elif cmd == "c":
            for i, e in enumerate(orch.chronicle.all(), 1):
                console.print(f"  {i}. (第 {e['turn']} 回合) {e['summary']}")
            continue
        elif cmd == "s":
            console.print(orch.world.snapshot.model_dump_json(indent=2))
            continue
        try:
            asyncio.run(orch.run_turn())
        except KeyboardInterrupt:
            console.print("[yellow]中断[/yellow]")
            return
        except Exception as e:
            logger.error("turn_error", error=str(e), exc_info=True)
            if not Confirm.ask("继续？", default=True):
                return


@app.command()
def trace(query: str = typer.Argument(..., help="purpose 关键词"),
          config: Path = typer.Option(Path("config.yaml")),
          show_response: bool = typer.Option(False)):
    """查询 LLM 调用 trace。"""
    settings = Settings.load(str(config))
    tw = TraceWriter(settings.storage.trace_dir)
    results = tw.query(purpose_contains=query)
    console.print(f"[cyan]找到 {len(results)} 条匹配 '{query}'[/cyan]")
    for t in results:
        console.print(f"\n[bold]{t.timestamp}[/bold] [yellow]{t.purpose}[/yellow]")
        console.print(f"  model={t.model} tokens={t.input_tokens}+{t.output_tokens} {t.duration_ms}ms")
        if t.error:
            console.print(f"  [red]error: {t.error}[/red]")
        if show_response:
            console.print(f"  response: {t.response[:500]}")


@app.command()
def status(config: Path = typer.Option(Path("config.yaml"))):
    """显示当前会话状态。"""
    settings = Settings.load(str(config))
    cm = CheckpointManager(settings)
    info = cm.session_info()
    if not info:
        console.print("[dim]无会话[/dim]")
    else:
        console.print(info)


if __name__ == "__main__":
    app()