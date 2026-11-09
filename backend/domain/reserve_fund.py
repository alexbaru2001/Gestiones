"""Cómo se cuenta el fondo de reserva de cara a quien lo mira.

El objetivo clásico de un fondo de emergencia son seis meses de gasto corriente, y el pipeline lo
recalcula mes a mes con la media de gasto de todos los meses vividos. Aquí no se repite esa
aritmética: este módulo solo recoge el resultado y deriva lo que hace falta para enseñarlo
(cuánto falta, qué porcentaje está cubierto), para que la regla tenga un único dueño y no pueda
desviarse de lo que de verdad se ha calculado.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReserveFundStatus:
    """Situación del fondo en un momento dado."""

    objetivo: float
    cargado: float
    meses_de_cobertura: int = 6
    cuota_mensual: float = 0.0

    @property
    def porcentaje(self) -> float:
        """Fracción cubierta, entre 0 y 1. Sin objetivo no hay progreso que medir.

        Se topa en 1: un gasto medio a la baja puede dejar el fondo por encima del objetivo, y
        'tienes más de seis meses cubiertos' no se enseña como un 118 %.
        """
        if self.objetivo <= 0:
            return 0.0
        return min(self.cargado / self.objetivo, 1.0)

    @property
    def pendiente(self) -> float:
        return max(0.0, self.objetivo - self.cargado)

    def to_dict(self) -> dict[str, float]:
        return {
            "objetivo": round(self.objetivo, 2),
            "cargado": round(self.cargado, 2),
            "porcentaje": round(self.porcentaje, 4),
            "pendiente": round(self.pendiente, 2),
            "meses_de_cobertura": self.meses_de_cobertura,
            "cuota_mensual": self.cuota_mensual,
        }
