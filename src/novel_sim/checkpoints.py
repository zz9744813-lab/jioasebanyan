"""会话可恢复。"""
import shutil
from pathlib import Path
from .settings import Settings


class CheckpointManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def has_existing_session(self) -> bool:
        return (self.settings.storage.data_dir / "world_state.json").exists()

    def session_info(self) -> dict:
        from .world.state import WorldStore
        from .world.chronicle import Chronicle
        path = self.settings.storage.data_dir / "world_state.json"
        if not path.exists():
            return {}
        world = WorldStore(path)
        chron = Chronicle(self.settings.storage.data_dir / "chronicle.jsonl")
        return {"turn": world.turn, "char_count": len(world.char_ids()),
                "chronicle_entries": len(chron.all())}

    def reset(self):
        for d in (self.settings.storage.data_dir, self.settings.storage.output_dir):
            if d.exists():
                shutil.rmtree(d)
        self.settings.ensure_dirs()