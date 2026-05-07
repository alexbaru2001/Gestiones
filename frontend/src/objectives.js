export function createObjective(id = crypto.randomUUID()) {
  return {
    id,
    nombre: '',
    etiquetas: '',
    fraccion_presupuesto: 0.1,
    duracion_meses: 1,
    mes_inicio: '2024-10',
    saldo_inicial: 0,
  }
}

export function objectiveToPayload({ id, etiquetas, ...objective }) {
  return {
    ...objective,
    etiquetas: etiquetas
      .split(',')
      .map((tag) => tag.trim().toLowerCase())
      .filter(Boolean),
  }
}

export function objectiveFromApi(objective, id = crypto.randomUUID()) {
  return {
    id,
    nombre: objective.nombre ?? '',
    etiquetas: Array.isArray(objective.etiquetas) ? objective.etiquetas.join(', ') : String(objective.etiquetas ?? ''),
    fraccion_presupuesto: Number(objective.fraccion_presupuesto ?? 0),
    duracion_meses: Number(objective.duracion_meses ?? 1),
    mes_inicio: String(objective.mes_inicio ?? '2024-10').slice(0, 7),
    saldo_inicial: Number(objective.saldo_inicial ?? 0),
  }
}
