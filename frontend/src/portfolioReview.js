// Lógica pura de la revisión de cartera: qué posiciones se pueden analizar, cómo queda la nota
// del conjunto y qué ha cambiado desde la revisión anterior.

// Number(null) y Number('') son 0: sin esta guarda, una posición sin puntuar entraría en la media
// como un cero y hundiría la nota de la cartera.
function isNumber(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))
}

export function splitAnalyzablePositions(snapshot) {
  const positions = Array.isArray(snapshot?.positions) ? snapshot.positions : []
  const total = positions.reduce((sum, position) => sum + (Number(position?.current_value) || 0), 0)

  const analyzable = new Map()
  const skipped = []
  for (const position of positions) {
    const ticker = typeof position?.ticker === 'string' ? position.ticker.trim().toUpperCase() : ''
    const value = Number(position?.current_value) || 0
    if (!ticker) {
      skipped.push({ name: position?.name || position?.isin || 'Posición sin nombre', value, assetType: position?.asset_type })
      continue
    }
    // Una misma empresa puede estar en dos brókeres: se revisa una vez con el valor sumado.
    const existing = analyzable.get(ticker)
    if (existing) {
      existing.value += value
      existing.brokers.add(position?.broker)
      continue
    }
    analyzable.set(ticker, {
      ticker,
      name: position?.name || ticker,
      value,
      brokers: new Set([position?.broker].filter(Boolean)),
    })
  }

  const rows = [...analyzable.values()]
    .map((item) => ({
      ticker: item.ticker,
      name: item.name,
      value: item.value,
      weight: total > 0 ? (item.value / total) * 100 : 0,
      broker: [...item.brokers].join(' · '),
    }))
    .sort((left, right) => right.value - left.value)

  const skippedValue = skipped.reduce((sum, item) => sum + item.value, 0)
  return {
    analyzable: rows,
    skipped: skipped.sort((left, right) => right.value - left.value),
    skippedValue,
    skippedWeight: total > 0 ? (skippedValue / total) * 100 : 0,
    total,
  }
}

export function summarizeReview(entries = []) {
  const scored = entries.filter((entry) => isNumber(entry?.last_score))
  if (!scored.length) return null

  // Ponderada por valor: que una posición de 7 € flojee no vale lo mismo que una de 1.200 €.
  const weightSum = scored.reduce((sum, entry) => sum + (Number(entry.value) || 0), 0)
  const weightedScore = weightSum
    ? scored.reduce((sum, entry) => sum + Number(entry.last_score) * (Number(entry.value) || 0), 0) / weightSum
    : null
  const simpleScore = scored.reduce((sum, entry) => sum + Number(entry.last_score), 0) / scored.length

  const flagged = scored.filter((entry) => (entry.flags?.length ?? 0) > 0)
  const flaggedValue = flagged.reduce((sum, entry) => sum + (Number(entry.value) || 0), 0)

  return {
    analyzed: scored.length,
    weightedScore,
    simpleScore,
    flaggedCount: flagged.length,
    flaggedValue,
    flaggedWeight: weightSum ? (flaggedValue / weightSum) * 100 : 0,
    reviewedValue: weightSum,
  }
}

export function getValuationVerdict(point) {
  const current = Number(point?.rpd_ttm)
  const average = Number(point?.rpd_avg5)
  if (!isNumber(point?.rpd_ttm) || !isNumber(point?.rpd_avg5) || average <= 0) return null
  const difference = current - average
  if (difference >= average * 0.1) return 'barata'
  if (difference <= -average * 0.1) return 'cara'
  return 'en su media'
}

export function buildReviewRows(entries = []) {
  return entries.map((entry) => {
    const history = Array.isArray(entry.history) ? entry.history : []
    const last = history.at(-1) ?? null
    const previous = history.length > 1 ? history.at(-2) : null
    const delta =
      last && previous && isNumber(last.score) && isNumber(previous.score)
        ? Number(last.score) - Number(previous.score)
        : null
    return {
      ...entry,
      valuation: getValuationVerdict(last),
      delta,
      previousDate: previous?.date ?? null,
      flagCount: entry.flags?.length ?? 0,
    }
  })
}

const SCORE_DROP_THRESHOLD = 5

// Solo lo accionable: si no ha cambiado nada relevante hay que decirlo, no inventar movimiento.
export function detectReviewChanges(entries = []) {
  const changes = []
  for (const entry of entries) {
    const history = Array.isArray(entry.history) ? entry.history : []
    if (history.length < 2) continue
    const last = history.at(-1)
    const previous = history.at(-2)
    const label = entry.ticker

    const lastScore = Number(last.score)
    const previousScore = Number(previous.score)
    if (isNumber(last.score) && isNumber(previous.score)) {
      const delta = lastScore - previousScore
      if (delta <= -SCORE_DROP_THRESHOLD) {
        changes.push({ ticker: label, kind: 'score', severity: 'bad', text: `pierde ${Math.abs(delta).toFixed(1)} puntos de score` })
      } else if (delta >= SCORE_DROP_THRESHOLD) {
        changes.push({ ticker: label, kind: 'score', severity: 'good', text: `gana ${delta.toFixed(1)} puntos de score` })
      }
    }

    const lastFlags = Number(last.flags)
    const previousFlags = Number(previous.flags)
    if (isNumber(last.flags) && isNumber(previous.flags) && lastFlags > previousFlags) {
      changes.push({ ticker: label, kind: 'flags', severity: 'bad', text: `${lastFlags - previousFlags} bandera(s) roja(s) nueva(s)` })
    }

    const lastRpd = Number(last.rpd_ttm)
    const previousRpd = Number(previous.rpd_ttm)
    if (isNumber(last.rpd_ttm) && isNumber(previous.rpd_ttm) && previousRpd > 0 && lastRpd < previousRpd * 0.9) {
      changes.push({ ticker: label, kind: 'dividendo', severity: 'bad', text: 'el dividendo por acción o el precio han movido la RPD más de un 10 %' })
    }

    const lastVerdict = getValuationVerdict(last)
    const previousVerdict = getValuationVerdict(previous)
    if (lastVerdict && previousVerdict && lastVerdict !== previousVerdict) {
      changes.push({
        ticker: label,
        kind: 'valoracion',
        severity: lastVerdict === 'barata' ? 'good' : lastVerdict === 'cara' ? 'bad' : 'neutral',
        text: `pasa de ${previousVerdict} a ${lastVerdict} frente a su media de 5 años`,
      })
    }
  }
  return changes
}
