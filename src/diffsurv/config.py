from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, root: str = ".") -> str:
    value = Path(path)
    if value.is_absolute():
        return str(value)
    return str(Path(root) / value)
