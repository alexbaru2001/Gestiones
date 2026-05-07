from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    fecha_inicio: str = "2024-10-01"
    porcentaje_gasto: float = 0.3
    porcentaje_inversion: float = 0.1
    porcentaje_vacaciones: float = 0.05

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.fecha_inicio)
        except ValueError as exc:
            raise ValueError("fecha_inicio debe tener formato YYYY-MM-DD") from exc
        self._validate_percentage("porcentaje_gasto", self.porcentaje_gasto)
        self._validate_percentage("porcentaje_inversion", self.porcentaje_inversion)
        self._validate_percentage("porcentaje_vacaciones", self.porcentaje_vacaciones)

    def to_dict(self) -> dict:
        return {
            "fecha_inicio": self.fecha_inicio,
            "porcentaje_gasto": self.porcentaje_gasto,
            "porcentaje_inversion": self.porcentaje_inversion,
            "porcentaje_vacaciones": self.porcentaje_vacaciones,
        }

    def _validate_percentage(self, field: str, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(f"{field} debe estar entre 0 y 1")


@dataclass(frozen=True)
class ObjectiveConfig:
    nombre: str
    etiquetas: list[str]
    fraccion_presupuesto: float
    duracion_meses: int
    mes_inicio: str
    saldo_inicial: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "etiquetas": self.etiquetas,
            "fraccion_presupuesto": self.fraccion_presupuesto,
            "duracion_meses": self.duracion_meses,
            "mes_inicio": self.mes_inicio,
            "saldo_inicial": self.saldo_inicial,
        }


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
