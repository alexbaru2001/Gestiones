from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    fecha_inicio: str = "2024-10-01"
    porcentaje_gasto: float = 0.3
    porcentaje_inversion: float = 0.1
    porcentaje_vacaciones: float = 0.05


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
