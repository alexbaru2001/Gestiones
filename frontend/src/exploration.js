// Lógica pura de la exploración de universos: repartir el trabajo en tandas, ordenar la criba y
// preparar las filas del resultado.

// Number(null) y Number('') valen 0: sin esta guarda, una empresa cuyo análisis falló entraría en
// el ranking con un cero en vez de quedarse fuera.
function isNumber(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))
}

export const DEFAULT_FINALISTS = 40
export const DEFAULT_TOP = 20

export function chunkTickers(tickers = [], size = 100) {
  if (size <= 0) return []
  const chunks = []
  for (let index = 0; index < tickers.length; index += size) {
    chunks.push(tickers.slice(index, index + size))
  }
  return chunks
}

export function rankByPrescore(rows = [], finalists = DEFAULT_FINALISTS) {
  return [...rows]
    .filter((row) => isNumber(row?.prescore) && Number(row.prescore) > 0)
    .sort((left, right) => Number(right.prescore) - Number(left.prescore))
    .slice(0, finalists)
}

// Del análisis completo solo se guarda lo que se enseña en la tabla: guardar el análisis entero
// de 40 empresas dispararía el tamaño del archivo sin que aporte nada.
export function toExplorationRow(analysis, prescoreValue = null) {
  const metrics = analysis?.metrics ?? {}
  return {
    ticker: analysis?.ticker,
    name: analysis?.name,
    sector: analysis?.sector,
    currency: analysis?.exchange?.currency ?? '',
    price: analysis?.price ?? null,
    score: analysis?.score ?? null,
    recommendation: analysis?.recommendation ?? '',
    flags: analysis?.flags ?? [],
    rpd_ttm: metrics.rpd_ttm ?? null,
    rpd_avg5: metrics.rpd_avg5 ?? null,
    dgr5: metrics.dgr5 ?? null,
    payout: metrics.payout ?? null,
    per_ttm: metrics.per_ttm ?? null,
    streak_years: metrics.streak_years ?? null,
    prescore: prescoreValue,
  }
}

export function rankByScore(rows = [], top = DEFAULT_TOP) {
  return [...rows]
    .filter((row) => isNumber(row?.score))
    .sort((left, right) => Number(right.score) - Number(left.score))
    .slice(0, top)
}

export function getValuationLabel(row) {
  const current = Number(row?.rpd_ttm)
  const average = Number(row?.rpd_avg5)
  if (!isNumber(row?.rpd_ttm) || !isNumber(row?.rpd_avg5) || average <= 0) return null
  const difference = current - average
  if (difference >= average * 0.1) return 'barata'
  if (difference <= -average * 0.1) return 'cara'
  return 'en su media'
}

export function describeProgress(progress) {
  if (!progress) return ''
  // Se cuenta la que se está haciendo, no las ya terminadas: "Criba 0 de 35" no dice nada.
  const current = Math.min(Number(progress.done) + 1, Number(progress.total) || 1)
  if (progress.phase === 'criba') {
    return `Criba ${current} de ${progress.total} empresas`
  }
  if (progress.phase === 'analisis') {
    return `Análisis a fondo ${current} de ${progress.total}${progress.ticker ? ` · ${progress.ticker}` : ''}`
  }
  return ''
}
