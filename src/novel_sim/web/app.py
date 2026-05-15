"""FastAPI Web UI。本地单用户使用，不做鉴权。"""
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from ..settings import Settings
from ..checkpoints import CheckpointManager
from ..events import get_bus
from ..cli import _build_orchestrator

_state = {"orchestrator": None, "settings": None, "scenario": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
HERE = Path(__file__).parent
STATIC = HERE / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/init")
async def init(body: dict):
    config_path = body.get("config", "config.yaml")
    scenario_path = body["scenario"]
    reset = body.get("reset", False)
    settings = Settings.load(config_path)
    cm = CheckpointManager(settings)
    if reset:
        cm.reset()
    orch = _build_orchestrator(settings)
    await orch.initialize(Path(scenario_path))
    _state["orchestrator"] = orch
    _state["settings"] = settings
    _state["scenario"] = scenario_path
    return {"ok": True, "world": orch.world.snapshot.model_dump()}


@app.post("/api/turn")
async def turn():
    orch = _state["orchestrator"]
    if orch is None:
        raise HTTPException(400, "未初始化，请先调用 /api/init")
    await orch.run_turn()
    return {"ok": True, "world": orch.world.snapshot.model_dump()}


@app.post("/api/writing")
async def writing():
    orch = _state["orchestrator"]
    if orch is None:
        raise HTTPException(400, "未初始化")
    result = await orch.writing_phase()
    return {"ok": True, "result": str(result) if result else None}


@app.get("/api/status")
async def status():
    orch = _state["orchestrator"]
    if orch is None:
        return {"initialized": False}
    return {"initialized": True, "scenario": _state["scenario"], "world": orch.world.snapshot.model_dump()}


@app.get("/api/events")
async def events():
    bus = get_bus()
    q = bus.subscribe()

    async def gen():
        try:
            yield {"event": "hello", "data": json.dumps({"ok": True})}
            while True:
                evt = await q.get()
                yield {"event": evt["type"], "data": json.dumps(evt["payload"], ensure_ascii=False)}
        except asyncio.CancelledError:
            raise
        finally:
            bus.unsubscribe(q)

    return EventSourceResponse(gen())
