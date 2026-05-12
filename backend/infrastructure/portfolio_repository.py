from __future__ import annotations

from datetime import date
import re
from pathlib import Path

from backend.config import get_investments_path
from backend.domain.portfolio import UploadedInvestmentFile, build_snapshot_from_files, load_snapshot, save_snapshot


class LocalPortfolioRepository:
    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or get_investments_path()
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.snapshots_path = self.base_path / "snapshots"
        self.snapshot_path = self.processed_path / "portfolio_snapshot.json"

    def save_uploads_and_rebuild(self, files: list[UploadedInvestmentFile]) -> dict:
        snapshot = build_snapshot_from_files(files)
        month_key = snapshot.get("snapshot_month") or "sin-fecha"
        import_path = self.raw_path / month_key
        import_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            target = import_path / safe_filename(file.filename)
            target.write_bytes(file.content)
        self.save_current_and_monthly_snapshot(snapshot)
        return snapshot

    def rebuild(self) -> dict:
        files = self.load_raw_files()
        snapshot = build_snapshot_from_files(files)
        self.save_current_and_monthly_snapshot(snapshot)
        return snapshot

    def load(self) -> dict | None:
        return load_snapshot(self.snapshot_path)

    def load_snapshots(self) -> list[dict]:
        if not self.snapshots_path.exists():
            return []
        snapshots = []
        today = date.today().isoformat()
        for path in sorted(self.snapshots_path.glob("*.json")):
            snapshot = load_snapshot(path)
            snapshot_date = snapshot.get("snapshot_date") if snapshot else None
            if snapshot and (not snapshot_date or snapshot_date <= today):
                snapshots.append(snapshot)
        return sorted(snapshots, key=lambda item: item.get("snapshot_month") or "")

    def save_current_and_monthly_snapshot(self, snapshot: dict) -> None:
        save_snapshot(self.snapshot_path, snapshot)
        month_key = snapshot.get("snapshot_month") or "sin-fecha"
        save_snapshot(self.snapshots_path / f"{month_key}.json", snapshot)

    def load_raw_files(self) -> list[UploadedInvestmentFile]:
        if not self.raw_path.exists():
            return []
        files = []
        for path in sorted(self.raw_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pdf", ".xlsx"}:
                continue
            files.append(UploadedInvestmentFile(filename=path.name, content=path.read_bytes()))
        return files


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", name).strip() or "documento"
