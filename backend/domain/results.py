from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MovementSummary:
    gastos: int
    ingresos: int
    transferencias: int
    cuentas: int

    def to_dict(self) -> dict:
        return {
            "gastos": self.gastos,
            "ingresos": self.ingresos,
            "transferencias": self.transferencias,
            "cuentas": self.cuentas,
        }


@dataclass(frozen=True)
class HistoryResult:
    meses: int
    ultimo_mes: dict[str, Any] | None
    resumen: list[dict[str, Any]]
    objetivos: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "meses": self.meses,
            "ultimo_mes": self.ultimo_mes,
            "resumen": self.resumen,
            "objetivos": self.objetivos,
        }


@dataclass(frozen=True)
class PipelineResult:
    params: dict[str, Any]
    movimientos: MovementSummary
    historial: HistoryResult

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "movimientos": self.movimientos.to_dict(),
            "historial": self.historial.to_dict(),
        }
