"""世界状态。Pydantic-typed，深合并 patch。"""
import json
from pathlib import Path
from ..models import WorldStateSnapshot, CharacterPersona, WorldLocation


class WorldStore:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> WorldStateSnapshot:
        if self.save_path.exists():
            return WorldStateSnapshot.model_validate_json(
                self.save_path.read_text(encoding="utf-8"))
        return WorldStateSnapshot()

    def save(self):
        self.save_path.write_text(self._state.model_dump_json(indent=2),
                                  encoding="utf-8")

    @property
    def snapshot(self) -> WorldStateSnapshot:
        return self._state.model_copy(deep=True)

    @property
    def turn(self) -> int:
        return self._state.turn

    def char_ids(self) -> list[str]:
        return list(self._state.characters.keys())

    def get_char(self, char_id: str) -> CharacterPersona | None:
        return self._state.characters.get(char_id)

    def apply_patch(self, patch: dict):
        as_dict = self._state.model_dump()
        _deep_merge(as_dict, patch)
        try:
            self._state = WorldStateSnapshot.model_validate(as_dict)
        except Exception as e:
            from ..logging_setup import get_logger
            get_logger().error("world_patch_invalid", error=str(e), patch=patch)
            return
        self.save()

    def initialize(self, world_rules: str, time: str,
                   locations: dict[str, WorldLocation],
                   characters: dict[str, CharacterPersona],
                   environment: dict):
        self._state = WorldStateSnapshot(
            turn=0, time=time, world_rules=world_rules,
            locations=locations, characters=characters,
            environment=environment)
        self.save()


def _deep_merge(target: dict, patch: dict):
    for k, v in patch.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v