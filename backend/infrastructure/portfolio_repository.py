from __future__ import annotations

import re
from pathlib import Path

from backend.config import get_investments_path
from backend.domain.portfolio import UploadedInvestmentFile, build_snapshot_from_files, load_snapshot, save_snapshot


class LocalPortfolioRepository:
    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or get_investments_path()
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.snapshot_path = self.processed_path / "portfolio_snapshot.json"

    def save_uploads_and_rebuild(self, files: list[UploadedInvestmentFile]) -> dict:
        self.raw_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            target = self.raw_path / safe_filename(file.filename)
            target.write_bytes(file.content)
        return self.rebuild()

    def rebuild(self) -> dict:
        files = self.load_raw_files()
        snapshot = build_snapshot_from_files(files)
        save_snapshot(self.snapshot_path, snapshot)
        return snapshot

    def load(self) -> dict | None:
        return load_snapshot(self.snapshot_path)

    def load_raw_files(self) -> list[UploadedInvestmentFile]:
        if not self.raw_path.exists():
            return []
        files = []
        for path in sorted(self.raw_path.iterdir()):
            if path.suffix.lower() not in {".pdf", ".xlsx"}:
                continue
            files.append(UploadedInvestmentFile(filename=path.name, content=path.read_bytes()))
        return files


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", name).strip() or "documento"
