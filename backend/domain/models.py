from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    fecha_inicio: str = "2024-10-01"
    porcentaje_gasto: float = 0.3
    porcentaje_inversion: float = 0.1
    porcentaje_vacaciones: float = 0.05
