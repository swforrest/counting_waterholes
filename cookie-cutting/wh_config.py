"""Load waterhole_seg_config.yaml and resolve its paths.

Usage from a notebook:

    import wh_config
    cfg = wh_config.load()
    cfg.paths["tiles"]          # absolute Path
    cfg.classes                 # list of ClassDef
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# cookie-cutting/wh_config.py -> cookie-cutting -> repository root
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "waterhole_seg_config.yaml"


@dataclass(frozen=True)
class ClassDef:
    """One entry in the class scheme."""

    id: int
    name: str
    colour: str
    key: str
    ignore: bool
    description: str


@dataclass(frozen=True)
class Config:
    """Parsed config: raw dict, resolved paths, and the class scheme."""

    raw: dict[str, Any]
    paths: dict[str, Path]
    classes: list[ClassDef]
    source_path: Path
    hash: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def class_by_id(self, class_id: int) -> ClassDef:
        for definition in self.classes:
            if definition.id == class_id:
                return definition
        raise KeyError(f"no class with id {class_id} in {self.source_path}")

    def class_by_name(self, name: str) -> ClassDef:
        for definition in self.classes:
            if definition.name == name:
                return definition
        raise KeyError(f"no class named {name!r} in {self.source_path}")

    @property
    def trainable_classes(self) -> list[ClassDef]:
        """Classes that take part in training and metrics (everything but ignore)."""
        return [definition for definition in self.classes if not definition.ignore]


def load(config_path: str | Path | None = None) -> Config:
    """Read the config, resolve paths against the repo root, validate the scheme.

    Paths are resolved here rather than at point of use so that notebooks behave
    identically whether they are run from the repo root or from cookie-cutting/.
    Directories are not created as a side effect of loading; the modules that
    write to them create them.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    text = path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    paths = {
        name: (REPO_ROOT / value).resolve()
        for name, value in raw["paths"].items()
    }

    classes = _parse_classes(raw, path)
    config_hash = hashlib.sha256(text.encode()).hexdigest()[:12]

    return Config(
        raw=raw,
        paths=paths,
        classes=classes,
        source_path=path,
        hash=config_hash,
    )


def _parse_classes(raw: dict[str, Any], path: Path) -> list[ClassDef]:
    """Turn the YAML class definitions into ClassDefs, checking they are coherent."""
    definitions = [
        ClassDef(
            id=int(entry["id"]),
            name=str(entry["name"]),
            colour=str(entry["colour"]),
            key=str(entry["key"]),
            ignore=bool(entry.get("ignore", False)),
            description=str(entry.get("description", "")).strip(),
        )
        for entry in raw["classes"]["definitions"]
    ]

    ids = [definition.id for definition in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate class ids {sorted(ids)}")

    names = [definition.name for definition in definitions]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate class names {sorted(names)}")

    keys = [definition.key for definition in definitions]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate keybindings {sorted(keys)}")

    if min(ids) < 0 or max(ids) > 255:
        raise ValueError(f"{path}: class ids must fit in uint8, got {sorted(ids)}")

    ignore_ids = [definition.id for definition in definitions if definition.ignore]
    if ignore_ids != [0]:
        raise ValueError(
            f"{path}: class 0 and only class 0 must be the ignore class, "
            f"got ignore ids {ignore_ids}"
        )

    return definitions
