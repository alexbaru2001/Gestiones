// Formato español en toda la app: miles agrupados también en cifras de cuatro dígitos
// (el valor por defecto de Intl no agrupa 1234) y signo menos tipográfico.
const MINUS = '−'

const euro = new Intl.NumberFormat('es-ES', {
  style: 'currency',
  currency: 'EUR',
  useGrouping: 'always',
})

function withRealMinus(text) {
  return text.replace(/-/g, MINUS)
}

export function formatMoney(value) {
  if (typeof value !== 'number') return '-'
  return withRealMinus(euro.format(value))
}

export function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`.replace('-', MINUS)
}

export function formatDelta(value) {
  if (typeof value !== 'number') return '-'
  const sign = value > 0 ? '+' : ''
  return `${sign}${formatMoney(value)}`
}
