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
        snapshot_key = snapshot_storage_key(snapshot)
        import_path = self.raw_path / snapshot_key
        import_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            target = import_path / safe_filename(file.filename)
            target.write_bytes(file.content)
        self.save_current_and_snapshot(snapshot)
        return snapshot

    def rebuild(self) -> dict:
        files = self.load_raw_files()
        snapshot = self.enrich_with_finance_history(build_snapshot_from_files(files))
        self.save_current_and_snapshot(snapshot)
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
        return sorted(snapshots, key=snapshot_storage_key)

    def save_current_and_snapshot(self, snapshot: dict) -> None:
        save_snapshot(self.snapshot_path, snapshot)
        snapshot_key = snapshot_storage_key(snapshot)
        save_snapshot(self.snapshots_path / f"{snapshot_key}.json", snapshot)

    def update_snapshot_date(self, snapshot_key: str, snapshot_date: str) -> dict:
        try:
            parsed_date = date.fromisoformat(snapshot_date)
        except ValueError as exc:
            raise ValueError("Indica una fecha válida con formato YYYY-MM-DD.") from exc
        current_key = safe_filename(snapshot_key)
        previous_path = self.snapshots_path / f"{current_key}.json"
        if not previous_path.exists():
            raise FileNotFoundError("No se encontró la foto seleccionada.")

        new_key = parsed_date.isoformat()
        next_path = self.snapshots_path / f"{new_key}.json"
        if next_path.exists() and next_path != previous_path:
            raise FileExistsError("Ya existe una foto con esa fecha.")

        snapshot = load_snapshot(previous_path)
        if not snapshot:
            raise ValueError("La foto seleccionada no se pudo leer.")

        updated_snapshot = {
            **snapshot,
            "snapshot_key": new_key,
            "snapshot_date": new_key,
            "snapshot_month": new_key[:7],
        }
        save_snapshot(next_path, updated_snapshot)
        if next_path != previous_path:
            previous_path.unlink()
            self.rename_raw_snapshot_folder(current_key, new_key)

        current_snapshot = load_snapshot(self.snapshot_path)
        if current_snapshot and snapshot_storage_key(current_snapshot) == current_key:
            save_snapshot(self.snapshot_path, updated_snapshot)
        return self.enrich_with_finance_history(updated_snapshot)

    def rename_raw_snapshot_folder(self, current_key: str, new_key: str) -> None:
        previous_path = self.raw_path / current_key
        next_path = self.raw_path / new_key
        if previous_path.exists() and not next_path.exists():
            previous_path.rename(next_path)

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
        previous_summary = self.load_latest_previous_summary(snapshot)
        if not finance_row:
            if not previous_summary:
                return snapshot
            summary = {**snapshot.get("summary", {})}
            summary["dividends"] = summary.get("dividends") or previous_summary.get("dividends", 0.0)
            summary["fees"] = summary.get("fees") or previous_summary.get("fees", 0.0)
            return {**snapshot, "summary": summary}
        summary = {**snapshot.get("summary", {})}
        summary["finance_invested"] = finance_row["finance_invested"]
        summary["finance_source_month"] = finance_row["month"]
        summary["investment_bucket"] = finance_row["investment_bucket"]
        summary["investment_net_worth"] = round(finance_row["finance_invested"] + finance_row["investment_bucket"], 2)
        summary["dividends"] = finance_row["dividends"]
        summary["fees"] = finance_row["fees"]
        if finance_row["finance_invested"] > 0:
            invested = summary.get("invested") or 0.0
            fees = summary.get("fees") or 0.0
            gain = round(invested - finance_row["finance_invested"] - fees, 2)
            summary["known_cost"] = finance_row["finance_invested"]
            summary["known_unrealized_gain"] = gain
            summary["known_unrealized_gain_pct"] = round(gain / finance_row["finance_invested"] * 100.0, 2)
        return {**snapshot, "summary": summary}

    def load_finance_history_row(self, month: str | None) -> dict | None:
        if not month or not self.finance_history_path.exists():
            return None
        fallback = None
        with self.finance_history_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = sorted(csv.DictReader(file), key=lambda row: row.get("Mes") or "")
            for row in rows:
                row_month = row.get("Mes")
                if not row_month:
                    continue
                if row_month > month:
                    continue
                parsed = {
                    "month": row_month,
                    "finance_invested": parse_csv_number(row.get("Dinero Invertido")),
                    "investment_bucket": parse_csv_number(row.get("Inversiones") or row.get("📈 Inversiones")),
                    "dividends": parse_csv_number(first_present(row, ["Dividendos netos", "Dividendos", "dividendos"])),
                    "fees": parse_csv_number(first_present(row, ["Comisiones", "Comisiones inversión", "comisiones"])),
                }
                if row_month == month:
                    return parsed
                fallback = parsed
        return fallback

    def load_latest_previous_summary(self, snapshot: dict) -> dict:
        if not self.snapshots_path.exists():
            return {}
        current_key = snapshot_storage_key(snapshot)
        previous = []
        for path in sorted(self.snapshots_path.glob("*.json")):
            stored = load_snapshot(path)
            if not stored:
                continue
            stored_key = snapshot_storage_key(stored)
            if stored_key and current_key and stored_key < current_key:
                previous.append((stored_key, stored.get("summary", {})))
        for _key, summary in reversed(previous):
            if summary.get("dividends") or summary.get("fees"):
                return summary
        return {}


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ._ -]", "_", name).strip() or "documento"


def snapshot_storage_key(snapshot: dict) -> str:
    return safe_filename(snapshot.get("snapshot_key") or snapshot.get("snapshot_date") or snapshot.get("snapshot_month") or "sin-fecha")


def first_present(row: dict, columns: list[str]) -> str | None:
    for column in columns:
        if column in row:
            return row.get(column)
    return None


def parse_csv_number(value: str | None) -> float:
    if value is None or not str(value).strip():
        return 0.0
    normalized = str(value).strip().replace(".", "").replace(",", ".") if "," in str(value) else str(value).strip()
    try:
        return round(float(normalized), 2)
    except ValueError:
        return 0.0
