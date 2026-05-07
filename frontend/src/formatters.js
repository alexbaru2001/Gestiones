export function formatMoney(value) {
  if (typeof value !== 'number') return '-'
  return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(value)
}

export function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`
}

export function formatDelta(value) {
  if (typeof value !== 'number') return '-'
  const sign = value > 0 ? '+' : ''
  return `${sign}${formatMoney(value)}`
}
