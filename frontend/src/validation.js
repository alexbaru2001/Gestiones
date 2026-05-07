export function validateFinanceInput(params, objectives) {
  const messages = []
  if (!params.fecha_inicio) {
    messages.push('La fecha de inicio es obligatoria.')
  }

  for (const [label, value] of [
    ['Gasto', params.porcentaje_gasto],
    ['Inversión', params.porcentaje_inversion],
    ['Vacaciones', params.porcentaje_vacaciones],
  ]) {
    if (!isValidFraction(value)) {
      messages.push(`${label} debe estar entre 0 y 1.`)
    }
  }

  const names = new Set()
  const repeatedNames = new Set()
  let totalFraction = 0
  objectives.forEach((objective) => {
    const name = objective.nombre.trim().toLowerCase()
    if (names.has(name)) repeatedNames.add(objective.nombre.trim())
    names.add(name)

    if (!isValidFraction(objective.fraccion_presupuesto)) {
      messages.push(`Objetivo "${objective.nombre}": la fracción debe estar entre 0 y 1.`)
    }
    totalFraction += Number(objective.fraccion_presupuesto) || 0

    if (!Number.isInteger(objective.duracion_meses) || objective.duracion_meses <= 0) {
      messages.push(`Objetivo "${objective.nombre}": los meses deben ser un entero mayor que 0.`)
    }

    if (!/^\d{4}-\d{2}$/.test(objective.mes_inicio)) {
      messages.push(`Objetivo "${objective.nombre}": el inicio debe tener formato YYYY-MM.`)
    }
  })

  if (repeatedNames.size > 0) {
    messages.push(`Hay objetivos repetidos: ${Array.from(repeatedNames).join(', ')}.`)
  }
  if (totalFraction > 1 + 1e-9) {
    messages.push('La suma de fracciones de objetivos no puede superar 1.')
  }

  return messages
}

function isValidFraction(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1
}
