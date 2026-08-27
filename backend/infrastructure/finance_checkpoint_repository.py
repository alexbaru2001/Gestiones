from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import get_finance_checkpoint_path


class JsonFinanceCheckpointRepository:
    def __init__(self, path: Path | None = None):
        self.path = path or get_finance_checkpoint_path()

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def save(self, checkpoint: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
