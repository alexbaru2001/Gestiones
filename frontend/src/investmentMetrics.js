// Lógica pura del panel de Invertir: umbrales por sector, evaluación de métricas y lectura del
// histórico de seguimiento. Vive fuera del componente para poder probarla con `npm test`.

export const metricGroups = [
  ['rpd_ttm', 'RPD TTM', '%'],
  ['rpd_forward', 'RPD forward', '%'],
  ['rpd_avg5', 'RPD media 5a', '%'],
  ['dgr5', 'DGR 5 años', '%'],
  ['dgr10', 'DGR 10 años', '%'],
  ['payout', 'Payout', '%'],
  ['per_ttm', 'PER', ''],
  ['de_ratio', 'Deuda/Patrimonio', 'x'],
  ['roe', 'ROE', '%'],
  ['ev_ebitda', 'EV/EBITDA', 'x'],
  ['fcf_yield', 'FCF yield', '%'],
  ['streak_years', 'Racha pagos', 'años'],
  ['streak_growth', 'Racha crecimiento', 'años'],
]

export const metricHelp = {
  rpd_ttm: 'Rentabilidad por dividendo pagada en los últimos 12 meses. Mide la renta actual.',
  rpd_forward: 'Rentabilidad esperada con el dividendo anual previsto. Depende de estimaciones.',
  rpd_avg5: 'Rentabilidad media de los últimos 5 años cerrados. Es la referencia para saber si hoy cotiza cara o barata.',
  dgr5: 'Crecimiento anual compuesto del dividendo a 5 años.',
  dgr10: 'Crecimiento anual compuesto del dividendo a 10 años.',
  payout: 'Porcentaje del beneficio destinado a dividendos. Demasiado alto deja poco margen.',
  per_ttm: 'Precio entre beneficio por acción. Ayuda a ver si el precio es exigente frente a beneficios.',
  de_ratio: 'Deuda frente a fondos propios. Menos deuda da más margen en una crisis.',
  roe: 'Rentabilidad sobre fondos propios: calidad con la que convierte capital en beneficio.',
  ev_ebitda: 'Valor de empresa frente al EBITDA. Complementa al PER cuando hay deuda relevante.',
  fcf_yield: 'Flujo de caja libre frente a capitalización: cuánto efectivo genera respecto al precio.',
  streak_years: 'Años consecutivos pagando dividendo.',
  streak_growth: 'Años consecutivos aumentando el dividendo.',
}

// Guía de referencia visible al final del panel: la ayuda al pasar el ratón no sustituye a poder
// leer de corrido qué significa cada ratio.
export const ratioExplanations = [
  ['RPD TTM', 'Rentabilidad por dividendo pagada durante los últimos 12 meses. Sirve para medir renta actual.'],
  ['RPD forward', 'Rentabilidad esperada usando el dividendo anual previsto. Es útil, pero depende de estimaciones.'],
  ['RPD media 5 años', 'Rentabilidad media de los últimos cinco años cerrados, calculada con el precio medio de cada año. Comparar la RPD de hoy con ella indica si la empresa cotiza cara o barata frente a su propia historia.'],
  ['DGR 5/10 años', 'Crecimiento anual compuesto del dividendo. Para dividendos crecientes interesa que sea positivo y estable.'],
  ['Payout', 'Porcentaje del beneficio destinado a dividendos. Si es demasiado alto, el dividendo tiene menos margen.'],
  ['PER', 'Precio dividido entre beneficio por acción. Ayuda a valorar si el precio parece exigente frente a beneficios.'],
  ['Deuda/Patrimonio', 'Relación entre deuda y fondos propios. Menor deuda suele dar más margen en crisis.'],
  ['ROE', 'Rentabilidad sobre fondos propios. Mide la calidad con la que la empresa convierte capital en beneficios.'],
  ['EV/EBITDA', 'Valor de empresa frente al EBITDA. Complementa al PER, especialmente si hay deuda relevante.'],
  ['FCF yield', 'Flujo de caja libre frente a capitalización. Indica cuánto efectivo genera el negocio respecto al precio.'],
  ['Rachas', 'Años consecutivos pagando o aumentando dividendo. Dan contexto sobre disciplina y estabilidad histórica.'],
  ['Precio de referencia', 'Precio al que el dividendo actual rendiría lo que ha rendido de media estos cinco años. Es orientativo: no incorpora crecimiento futuro ni riesgo del negocio.'],
  ['Score', 'Nota de 0 a 100 que pondera Dividendo (40), Solidez (25), Valoración (15) e Historial (20), con umbrales distintos según el sector de la empresa.'],
]

// Cada métrica contra el umbral que usa el propio score, para poder enseñarlo junto al valor.
const METRIC_BANDS = {
  rpd_ttm: (rules) => ({ min: rules.rpd_min, max: rules.rpd_max }),
  rpd_forward: (rules) => ({ min: rules.rpd_min, max: rules.rpd_max }),
  dgr5: (rules) => ({ min: rules.dgr5_min }),
  dgr10: (rules) => ({ min: rules.dgr5_min }),
  payout: (rules) => ({ min: rules.payout_min, max: rules.payout_max }),
  payout_fcf: (rules) => ({ min: rules.payout_min, max: rules.payout_max }),
  per_ttm: (rules) => ({ min: rules.per_min, max: rules.per_max }),
  de_ratio: (rules) => ({ max: rules.de_max }),
  roe: (rules) => ({ min: rules.roe_min }),
  streak_years: (rules) => ({ min: rules.streak_min }),
  streak_growth: (rules) => ({ min: rules.streak_min }),
}

export function getMetricBand(key, rules) {
  if (!rules || !METRIC_BANDS[key]) return null
  const band = METRIC_BANDS[key](rules)
  const min = Number.isFinite(Number(band.min)) ? Number(band.min) : null
  const max = Number.isFinite(Number(band.max)) ? Number(band.max) : null
  if (min === null && max === null) return null
  return { min, max }
}

export function evaluateMetric(value, band) {
  // Number(null) es 0, y un 0 colado se juzgaría como "por debajo del mínimo" en vez de "sin dato".
  if (!band || value === null || value === undefined || value === '' || !Number.isFinite(Number(value))) return null
  const number = Number(value)
  if (band.min !== null && number < band.min) return 'off'
  if (band.max !== null && number > band.max) return 'off'
  return 'on'
}

export function formatBand(band, suffix = '') {
  if (!band) return ''
  const unit = suffix ? ` ${suffix}` : ''
  if (band.min !== null && band.max !== null) return `objetivo ${band.min}–${band.max}${unit}`
  if (band.min !== null) return `mín. ${band.min}${unit}`
  return `máx. ${band.max}${unit}`
}

// El bloque con más puntos perdidos es la respuesta a "¿qué le falta para subir de tramo?".
export const scoreBlockWeights = { Dividendo: 40, Solidez: 25, Valoración: 15, Historial: 20 }

export function getWeakestBlock(breakdown) {
  if (!breakdown) return null
  const rows = Object.entries(scoreBlockWeights)
    .map(([block, weight]) => ({ block, weight, score: Number(breakdown[block] ?? 0), lost: weight - Number(breakdown[block] ?? 0) }))
    .filter((row) => Number.isFinite(row.lost))
  if (!rows.length) return null
  return rows.sort((left, right) => right.lost - left.lost)[0]
}

export function getValuation(metrics) {
  const current = Number(metrics?.rpd_ttm)
  const average = Number(metrics?.rpd_avg5)
  if (!Number.isFinite(current) || !Number.isFinite(average) || average <= 0) return null
  const difference = current - average
  // Más RPD que su media histórica = se paga menos por el mismo dividendo.
  const verdict = difference >= average * 0.1 ? 'barata' : difference <= -average * 0.1 ? 'cara' : 'en su media'
  return {
    current,
    average,
    difference,
    verdict,
    referencePrice: Number.isFinite(Number(metrics?.reference_price)) ? Number(metrics.reference_price) : null,
  }
}

// Un recorte de dividendo se ve de un vistazo si la barra del año que baja va en rojo.
export function markDividendCuts(rows = []) {
  return rows.map((row, index) => {
    const previous = index > 0 ? Number(rows[index - 1].amount) : null
    const amount = Number(row.amount)
    if (!Number.isFinite(previous) || !Number.isFinite(amount)) return { ...row, trend: 'flat' }
    if (amount < previous * 0.99) return { ...row, trend: 'down' }
    if (amount > previous * 1.01) return { ...row, trend: 'up' }
    return { ...row, trend: 'flat' }
  })
}

export function getScoreTrend(history = []) {
  const points = history.filter((point) => Number.isFinite(Number(point?.score)))
  if (points.length < 2) return null
  const last = Number(points.at(-1).score)
  const previous = Number(points.at(-2).score)
  return { last, previous, delta: last - previous, since: points.at(-2).date }
}

export function extractPortfolioTickers(snapshot) {
  const positions = Array.isArray(snapshot?.positions) ? snapshot.positions : []
  const seen = new Map()
  for (const position of positions) {
    const ticker = typeof position?.ticker === 'string' ? position.ticker.trim().toUpperCase() : ''
    if (!ticker || seen.has(ticker)) continue
    seen.set(ticker, {
      ticker,
      name: position.name || position.isin || ticker,
      value: Number(position.current_value) || 0,
    })
  }
  return [...seen.values()].sort((left, right) => right.value - left.value)
}

// En la comparativa se destaca la mejor celda de cada fila; algunas métricas no tienen "mejor".
export function getBestTickerForMetric(columns, metric) {
  if (!metric?.better) return null
  const candidates = columns
    .map((column) => ({ ticker: column.ticker, value: Number(column.metrics?.[metric.key]) }))
    .filter((candidate) => Number.isFinite(candidate.value))
  if (candidates.length < 2) return null
  const sorted = candidates.sort((left, right) =>
    metric.better === 'high' ? right.value - left.value : left.value - right.value,
  )
  return sorted[0].ticker
}
