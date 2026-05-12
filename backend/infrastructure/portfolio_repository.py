from __future__ import annotations

import csv
from datetime import date
import re
from pathlib import Path

from backend.config import get_finance_history_path, get_investments_path
from backend.domain.portfolio import UploadedInvestmentFile, build_snapshot_from_files, load_snapshot, save_snapshot


class LocalPortfolioRepository:
    def __init__(self, base_path: Path | None = None, finance_history_path: Path | None = None):
        self.base_path = base_path or get_investments_path()
        self.finance_history_path = finance_history_path or get_finance_history_path()
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.snapshots_path = self.base_path / "snapshots"
        self.snapshot_path = self.processed_path / "portfolio_snapshot.json"

    def save_uploads_and_rebuild(self, files: list[UploadedInvestmentFile]) -> dict:
        snapshot = self.enrich_with_finance_history(build_snapshot_from_files(files))
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
        snapshot = self.enrich_with_finance_history(build_snapshot_from_files(files))
        self.save_current_and_monthly_snapshot(snapshot)
        return snapshot

    def load(self) -> dict | None:
        snapshot = load_snapshot(self.snapshot_path)
        return self.enrich_with_finance_history(snapshot) if snapshot else None

    def load_snapshots(self) -> list[dict]:
        if not self.snapshots_path.exists():
            return []
        snapshots = []
        today = date.today().isoformat()
        for path in sorted(self.snapshots_path.glob("*.json")):
            snapshot = load_snapshot(path)
            snapshot_date = snapshot.get("snapshot_date") if snapshot else None
            if snapshot and (not snapshot_date or snapshot_date <= today):
                snapshots.append(self.enrich_with_finance_history(snapshot))
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

    def enrich_with_finance_history(self, snapshot: dict) -> dict:
        month = snapshot.get("snapshot_month")
        finance_row = self.load_finance_history_row(month)
        if not finance_row:
            return snapshot
        summary = {**snapshot.get("summary", {})}
        summary["finance_invested"] = finance_row["finance_invested"]
        summary["investment_bucket"] = finance_row["investment_bucket"]
        summary["investment_net_worth"] = round(finance_row["finance_invested"] + finance_row["investment_bucket"], 2)
        return {**snapshot, "summary": summary}

    def load_finance_history_row(self, month: str | None) -> dict | None:
        if not month or not self.finance_history_path.exists():
            return None
        with self.finance_history_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("Mes") != month:
                    continue
                return {
                    "finance_invested": parse_csv_number(row.get("Dinero Invertido")),
                    "investment_bucket": parse_csv_number(row.get("Inversiones") or row.get("📈 Inversiones")),
                }
        return None


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", name).strip() or "documento"


def parse_csv_number(value: str | None) -> float:
    if value is None or not str(value).strip():
        return 0.0
    normalized = str(value).strip().replace(".", "").replace(",", ".") if "," in str(value) else str(value).strip()
    try:
        return round(float(normalized), 2)
    except ValueError:
        return 0.0
