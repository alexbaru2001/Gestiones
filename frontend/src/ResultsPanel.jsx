import { useId, useLayoutEffect, useRef, useState } from 'react'
import { CalendarDays, Download, FileJson, Maximize2, Table2 } from 'lucide-react'
import { formatDelta, formatMoney } from './formatters'
import { aggregateDividendsByCompany, comparisonRows, getDividendPayments } from './resultSelectors'

const moneyFields = [
  'total',
  '💰 Ahorros',
  'Dinero Invertido',
  '💳 Gasto del mes',
  '💸 Presupuesto Mes',
  '🧾 Presupuesto Disponible',
  '📉 Deuda Presupuestaria mensual',
  '📉 Deuda Presupuestaria acumulada',
  '🎁 Regalos',
  '💼 Vacaciones',
  'Fondo de reserva cargado',
  '📈 Inversiones',
]

const tabs = [
  { id: 'resumen', label: 'Resumen' },
  { id: 'presupuesto', label: 'Presupuesto' },
  { id: 'gastos', label: 'Gastos' },
  { id: 'ahorro', label: 'Ahorro' },
  { id: 'inversiones', label: 'Inversiones' },
  { id: 'reservas', label: 'Reservas' },
  { id: 'tipologias', label: 'Tipologías' },
  { id: 'objetivos', label: 'Objetivos' },
  { id: 'datos', label: 'Datos' },
]

const typologyCompareFields = [
  { field: '💸 Presupuesto Mes', label: 'Presupuesto mensual', color: '#2e4057' },
  { field: '💳 Gasto del mes', label: 'Gasto del mes', color: '#7a2e2e' },
]

const typologyCards = [
  { field: '💰 Ahorros', label: 'Ahorros', color: '#1f4d3d' },
  { field: '💼 Vacaciones', label: 'Vacaciones', color: '#b8863b' },
  { field: '📈 Inversiones', label: 'Reserva inversión', color: '#1f5c6b' },
  { field: 'Fondo de reserva cargado', label: 'Fondo reserva', color: '#6e7a3a' },
  { field: '📉 Deuda Presupuestaria acumulada', label: 'Deuda acumulada', color: '#b15a2e' },
  { field: '🎁 Regalos', label: 'Regalos', color: '#8a6a2b' },
]

const typologyFields = [typologyCards[0], ...typologyCompareFields, ...typologyCards.slice(1)]

const typologyPeriods = [
  { value: '6', label: '6 meses' },
  { value: '12', label: '1 año' },
  { value: '24', label: '2 años' },
  { value: '36', label: '3 años' },
  { value: 'all', label: 'Todo' },
]

const expensePeriods = [
  { value: '1', label: 'Mes actual' },
  { value: '3', label: '3 meses' },
  { value: '6', label: '6 meses' },
  { value: '12', label: '1 año' },
  { value: '36', label: '3 años' },
  { value: 'all', label: 'Todo' },
]

const expenseCategoryFields = [
  { field: 'alimentacion', label: 'Alimentación', color: '#6a9ab5' },
  { field: 'transporte', label: 'Transporte', color: '#1d3557' },
  { field: 'hosteleria', label: 'Hostelería', color: '#7fa06f' },
  { field: 'entretenimiento', label: 'Entretenimiento', color: '#7a2e2e' },
  { field: 'otros', label: 'Otros', color: '#6e7a6c' },
]

const historyFields = [
  { label: 'Mes', field: 'Mes', type: 'text' },
  { label: 'Total', field: 'total', type: 'money' },
  { label: 'Ahorros', field: '💰 Ahorros', type: 'money' },
  { label: 'Regalos', field: '🎁 Regalos', type: 'money' },
  { label: 'Vacaciones', field: '💼 Vacaciones', type: 'money' },
  { label: 'Fondo reserva', field: 'Fondo de reserva cargado', type: 'money' },
  { label: 'Gasto', field: '💳 Gasto del mes', type: 'money' },
  { label: 'Presupuesto', field: '💸 Presupuesto Mes', type: 'money' },
  { label: 'Disponible', field: '🧾 Presupuesto Disponible', type: 'money' },
  { label: 'Inversiones', field: '📈 Inversiones', type: 'money' },
  { label: 'Invertido', field: 'Dinero Invertido', type: 'money' },
  { label: 'Exceso mes', field: '📉 Deuda Presupuestaria mensual', type: 'money' },
  { label: 'Exceso acum.', field: '📉 Deuda Presupuestaria acumulada', type: 'money' },
]

function asNumber(value) {
  return Number(value) || 0
}

function getBudget(selectedRow) {
  const monthBudget = asNumber(selectedRow['💸 Presupuesto Mes'])
  const availableBudget = asNumber(selectedRow['🧾 Presupuesto Disponible']) || monthBudget
  const spent = asNumber(selectedRow['💳 Gasto del mes'])
  const budgetConsumption = Math.max(0, spent)
  const budgetRefund = Math.max(0, -spent)
  const monthlyDebt = asNumber(selectedRow['📉 Deuda Presupuestaria mensual'])
  const accumulatedDebt = asNumber(selectedRow['📉 Deuda Presupuestaria acumulada'])
  const debtReserve = accumulatedDebt > 0 ? Math.min(monthBudget * 0.1, accumulatedDebt) : 0
  const usableBudget = Math.max(0, monthBudget - debtReserve)
  const remaining = Math.max(0, usableBudget - spent)
  const overrun = Math.max(monthlyDebt, spent - usableBudget, 0)
  const totalForBar = Math.max(monthBudget, debtReserve + budgetConsumption + remaining, 1)
  const execution = usableBudget > 0 ? (Math.min(budgetConsumption, usableBudget) / usableBudget) * 100 : 0

  return {
    accumulatedDebt,
    availableBudget,
    budgetConsumption,
    budgetRefund,
    debtReserve,
    execution,
    monthBudget,
    monthlyDebt,
    overrun,
    remaining,
    spent,
    totalForBar,
    usableBudget,
  }
}

function getSegmentWidth(value, total) {
  if (value <= 0 || total <= 0) {
    return '0%'
  }
  return `${Math.max(6, (value / total) * 100)}%`
}

function getExpenseAnalysis(result) {
  return result?.analisis?.gastos ?? {
    categorias: [],
    mensual: [],
    totales_categoria: [],
    ultimo_mes: null,
  }
}

function getSavingsAnalysis(result) {
  return result?.analisis?.ahorro ?? {
    mensual: [],
    ultimo_mes: null,
  }
}

function formatDateEs(value) {
  if (!value) return 's/d'
  const date = new Date(`${value}T00:00:00`)
  if (Number.isNaN(date.getTime())) return 's/d'
  return date.toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })
}

function getIncomeAnalysis(result) {
  return result?.analisis?.ingresos ?? {
    categorias: [],
    totales_categoria: [],
    ultimo_mes: null,
  }
}

function movingAverage(values, windowSize = 3) {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1)
    const chunk = values.slice(start, index + 1)
    return chunk.reduce((total, value) => total + value, 0) / chunk.length
  })
}

function getSvgCoordinates(values, min, max, options = {}) {
  const width = options.width ?? 640
  const height = options.height ?? 220
  const padding = options.padding ?? 18
  const paddingLeft = options.paddingLeft ?? padding
  const paddingRight = options.paddingRight ?? padding
  const paddingTop = options.paddingTop ?? padding
  const paddingBottom = options.paddingBottom ?? padding
  const range = max - min || 1
  return values.map((value, index) => ({
    x: paddingLeft + (index / Math.max(1, values.length - 1)) * (width - paddingLeft - paddingRight),
    y: height - paddingBottom - ((value - min) / range) * (height - paddingTop - paddingBottom),
  }))
}

function formatMonthLabel(value) {
  const [year, month] = String(value ?? '').split('-')
  if (!year || !month) return value
  return `${month}/${year.slice(-2)}`
}

function formatMoneyCompact(value) {
  return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(value)
}

function formatMonthLong(value) {
  const [year, month] = String(value ?? '').split('-').map(Number)
  if (!year || !month) return value
  const formatted = new Intl.DateTimeFormat('es-ES', { month: 'long', year: 'numeric' }).format(new Date(year, month - 1, 1))
  return formatted.charAt(0).toUpperCase() + formatted.slice(1)
}

function predictTypology(rows, field, months = 6) {
  const values = rows.map((row) => asNumber(row[field]))
  const averaged = movingAverage(values)
  const first = averaged[0] ?? 0
  const last = averaged[averaged.length - 1] ?? 0
  const slope = averaged.length > 1 ? (last - first) / (averaged.length - 1) : 0

  return {
    base: Math.max(0, last + slope * months),
    conservative: Math.max(0, last + slope * months * 0.75),
    optimistic: Math.max(0, last + slope * months * 1.25),
  }
}

function getPeriodRows(rows, period) {
  if (period === 'all') {
    return rows
  }
  return rows.slice(-Number(period))
}

function getExpenseRowsForPeriod(monthlyRows, selectedMonth, period) {
  const scopedRows = monthlyRows.filter((row) => !selectedMonth || String(row.Mes) <= String(selectedMonth))
  if (period === 'all') return scopedRows
  return scopedRows.slice(-Number(period))
}

function getExpenseCategoryRowsForPeriod(categoryRows, selectedMonth, period) {
  const scopedRows = getExpenseRowsForPeriod(categoryRows, selectedMonth, period)
  const totals = scopedRows.reduce((acc, row) => {
    Object.entries(row).forEach(([key, value]) => {
      if (key === 'Mes') return
      acc[key] = (acc[key] ?? 0) + asNumber(value)
    })
    return acc
  }, {})
  return Object.entries(totals)
    .map(([categoria, total]) => ({ categoria, total }))
    .filter((row) => row.total > 0)
    .sort((left, right) => right.total - left.total)
}

function bucketExpenseCategoryRows(rows, bucketSize, maxBuckets, categoryFields) {
  const chunks = []
  let end = rows.length
  while (end > 0 && chunks.length < maxBuckets) {
    const start = Math.max(0, end - bucketSize)
    chunks.unshift(rows.slice(start, end))
    end = start
  }
  return chunks
    .filter((chunk) => chunk.length > 0)
    .map((chunk) => {
      const bucket = { Mes: chunk[0].Mes, MesFin: chunk.at(-1).Mes }
      categoryFields.forEach((field) => {
        bucket[field.field] = chunk.reduce((sum, row) => sum + asNumber(row[field.field]), 0)
      })
      return bucket
    })
}

function formatExpenseBucketLabel(bucket) {
  if (!bucket.MesFin || bucket.MesFin === bucket.Mes) return formatMonthLabel(bucket.Mes)
  return `${formatMonthLabel(bucket.Mes)}–${formatMonthLabel(bucket.MesFin)}`
}

function bucketSingleValueRows(rows, valueField, bucketSize, maxBuckets) {
  const chunks = []
  let end = rows.length
  while (end > 0 && chunks.length < maxBuckets) {
    const start = Math.max(0, end - bucketSize)
    chunks.unshift(rows.slice(start, end))
    end = start
  }
  return chunks
    .filter((chunk) => chunk.length > 0)
    .map((chunk) => ({
      Mes: chunk[0].Mes,
      MesFin: chunk.at(-1).Mes,
      value: chunk.reduce((sum, row) => sum + asNumber(row[valueField]), 0),
    }))
}

function getSummaryGroups(row) {
  return {
    patrimony: [
      { label: 'Total', value: row.total },
      { label: 'Ahorros', value: row['💰 Ahorros'] },
      { label: 'Dinero invertido', value: row['Dinero Invertido'] },
      { label: 'Fondo emergencia', value: row['Fondo de reserva cargado'] },
      { label: 'Vacaciones', value: row['💼 Vacaciones'] },
      { label: 'Regalos', value: row['🎁 Regalos'] },
    ],
    budget: [
      { label: 'Presupuesto mes', value: row['💸 Presupuesto Mes'] },
      { label: 'Presupuesto disponible', value: row['🧾 Presupuesto Disponible'] },
      { label: 'Gasto del mes', value: row['💳 Gasto del mes'] },
      { label: 'Deuda presupuestaria', value: row['📉 Deuda Presupuestaria mensual'] },
      { label: 'Deuda acumulada', value: row['📉 Deuda Presupuestaria acumulada'] },
    ],
  }
}

function normalizeLabel(value) {
  return String(value ?? '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
}

function getCategoryAmount(expenseAnalysis, month, categories) {
  const normalizedCategories = (Array.isArray(categories) ? categories : [categories]).map(normalizeLabel)
  const monthlyRow = expenseAnalysis.categorias.find((row) => row.Mes === month)
  if (!monthlyRow) return 0
  const key = Object.keys(monthlyRow).find((field) => normalizedCategories.includes(normalizeLabel(field)))
  return asNumber(key ? monthlyRow[key] : 0)
}

function getCategoryTotalUntilMonth(analysis, month, categories) {
  const normalizedCategories = (Array.isArray(categories) ? categories : [categories]).map(normalizeLabel)
  return analysis.categorias
    .filter((row) => !month || String(row.Mes) <= String(month))
    .reduce((total, row) => {
      const key = Object.keys(row).find((field) => normalizedCategories.includes(normalizeLabel(field)))
      return total + asNumber(key ? row[key] : 0)
    }, 0)
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return '0,0%'
  return `${value.toLocaleString('es-ES', { maximumFractionDigits: 1, minimumFractionDigits: 1 })}%`
}

function axisPaddingLeft(min, max, formatCompact = formatMoneyCompact) {
  const chars = Math.max(formatCompact(min).length, formatCompact(max).length)
  return Math.min(70, Math.max(34, chars * 6 + 8))
}

// Mide el ancho real del contenedor: el viewBox debe coincidir con el tamaño
// renderizado o el SVG se estira de forma desigual (eje X mucho más que el Y)
// y la línea se ve aplastada, como una foto redimensionada sin mantener proporción.
function useElementWidth(fallback) {
  const ref = useRef(null)
  const [width, setWidth] = useState(fallback)

  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const measure = (entry) => setWidth(Math.round(entry?.contentRect.width || element.getBoundingClientRect().width || fallback))
    measure()
    const observer = new ResizeObserver(([entry]) => measure(entry))
    observer.observe(element)
    return () => observer.disconnect()
  }, [fallback])

  return [ref, width]
}

// Convierte los puntos en una curva suave (Catmull-Rom a Bézier) que pasa
// exactamente por cada valor real, sin inventar datos entre meses.
function smoothLinePath(points) {
  if (points.length < 2) return ''
  if (points.length === 2) {
    return `M ${points[0].x.toFixed(1)},${points[0].y.toFixed(1)} L ${points[1].x.toFixed(1)},${points[1].y.toFixed(1)}`
  }
  let path = `M ${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}`
  for (let index = 0; index < points.length - 1; index += 1) {
    const p0 = points[index - 1] ?? points[index]
    const p1 = points[index]
    const p2 = points[index + 1]
    const p3 = points[index + 2] ?? p2
    const c1x = p1.x + (p2.x - p0.x) / 6
    const c1y = p1.y + (p2.y - p0.y) / 6
    const c2x = p2.x - (p3.x - p1.x) / 6
    const c2y = p2.y - (p3.y - p1.y) / 6
    path += ` C ${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`
  }
  return path
}

function smoothAreaPath(points, baselineY) {
  const line = smoothLinePath(points)
  if (!line) return ''
  const first = points[0]
  const last = points.at(-1)
  return `${line} L ${last.x.toFixed(1)},${baselineY.toFixed(1)} L ${first.x.toFixed(1)},${baselineY.toFixed(1)} Z`
}

function TypologyMiniChart({
  color,
  formatCompact = formatMoneyCompact,
  formatValue = formatMoney,
  height = 64,
  label,
  months,
  values,
}) {
  const [tooltip, setTooltip] = useState(null)
  const gradientId = useId()
  const [containerRef, width] = useElementWidth(240)
  const min = Math.min(...values, 0)
  const max = Math.max(...values, 1)
  const paddingLeft = axisPaddingLeft(min, max, formatCompact)
  const paddingRight = 10
  const paddingTop = 12
  const paddingBottom = 18
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinates = getSvgCoordinates(values, min, max, chartOptions)
  const areaPath = smoothAreaPath(coordinates, height - paddingBottom)
  const linePath = smoothLinePath(coordinates)
  const last = coordinates.at(-1)
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const plotBottom = height - paddingBottom
  const trend = values.at(-1) >= values[0] ? 'tendencia ascendente' : 'tendencia descendente'

  return (
    <div className="mini-trend" onMouseLeave={() => setTooltip(null)} ref={containerRef}>
      <svg
        aria-label={`${label}: de ${formatValue(values[0])} a ${formatValue(values.at(-1))} en ${values.length} meses, ${trend}.`}
        className="mini-trend-svg"
        height={height}
        role="img"
        viewBox={`0 0 ${width} ${height}`}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.28" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            className="mini-grid-line"
            key={fraction}
            x1={paddingLeft}
            x2={width - paddingRight}
            y1={paddingTop + (plotBottom - paddingTop) * fraction}
            y2={paddingTop + (plotBottom - paddingTop) * fraction}
          />
        ))}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        <path d={areaPath} fill={`url(#${gradientId})`} stroke="none" />
        <path d={linePath} fill="none" stroke={color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        {coordinates.map((point, index) => (
          <circle
            cx={point.x}
            cy={point.y}
            fill="transparent"
            key={months[index] ?? index}
            onFocus={() => setTooltip({ month: months[index], value: values[index], x: point.x, y: point.y })}
            onMouseEnter={() => setTooltip({ month: months[index], value: values[index], x: point.x, y: point.y })}
            r="9"
            tabIndex="0"
          />
        ))}
        <circle cx={last.x} cy={last.y} fill={color} pointerEvents="none" r="3.6" />
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={paddingTop + 3}>
          {formatCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={plotBottom}>
          {formatCompact(min)}
        </text>
        <text className="mini-axis-month" textAnchor="start" x={paddingLeft} y={height - 4}>
          {formatMonthLabel(months[0])}
        </text>
        <text className="mini-axis-month" textAnchor="end" x={width - paddingRight} y={height - 4}>
          {formatMonthLabel(months.at(-1))}
        </text>
      </svg>
      {tooltip && (
        <div className="mini-trend-tooltip" style={{ left: `${(tooltip.x / width) * 100}%` }}>
          <span>{formatMonthLabel(tooltip.month)}</span>
          <strong>{formatValue(tooltip.value)}</strong>
        </div>
      )}
    </div>
  )
}

function TypologyFeatureButton({ label, onFeature }) {
  return (
    <button aria-label={`Mostrar ${label} en grande`} className="typology-expand-button" onClick={onFeature} type="button">
      <Maximize2 aria-hidden="true" size={13} />
    </button>
  )
}

function TypologyCompareChart({
  formatCompact = formatMoneyCompact,
  formatValue = formatMoney,
  height = 100,
  months,
  seriesA,
  seriesB,
}) {
  const [tooltip, setTooltip] = useState(null)
  const [containerRef, width] = useElementWidth(560)
  const combined = [...seriesA.values, ...seriesB.values]
  const min = Math.min(...combined, 0)
  const max = Math.max(...combined, 1)
  const paddingLeft = axisPaddingLeft(min, max, formatCompact)
  const paddingRight = 10
  const paddingTop = 12
  const paddingBottom = 18
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinatesA = getSvgCoordinates(seriesA.values, min, max, chartOptions)
  const coordinatesB = getSvgCoordinates(seriesB.values, min, max, chartOptions)
  const pathA = smoothLinePath(coordinatesA)
  const pathB = smoothLinePath(coordinatesB)
  const lastA = coordinatesA.at(-1)
  const lastB = coordinatesB.at(-1)
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const plotBottom = height - paddingBottom

  return (
    <div className="mini-trend compare-trend" onMouseLeave={() => setTooltip(null)} ref={containerRef}>
      <svg
        aria-label={`${seriesA.label} y ${seriesB.label} comparados mes a mes en el mismo eje. Último mes: ${seriesA.label} ${formatValue(seriesA.values.at(-1))}, ${seriesB.label} ${formatValue(seriesB.values.at(-1))}.`}
        className="mini-trend-svg"
        height={height}
        role="img"
        viewBox={`0 0 ${width} ${height}`}
      >
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            className="mini-grid-line"
            key={fraction}
            x1={paddingLeft}
            x2={width - paddingRight}
            y1={paddingTop + (plotBottom - paddingTop) * fraction}
            y2={paddingTop + (plotBottom - paddingTop) * fraction}
          />
        ))}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        <path d={pathA} fill="none" stroke={seriesA.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        <path
          d={pathB}
          fill="none"
          stroke={seriesB.color}
          strokeDasharray="6 4"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
        />
        {coordinatesA.map((point, index) => (
          <circle
            cx={point.x}
            cy={point.y}
            fill="transparent"
            key={months[index] ?? index}
            onFocus={() => setTooltip({ a: seriesA.values[index], b: seriesB.values[index], month: months[index], x: point.x })}
            onMouseEnter={() => setTooltip({ a: seriesA.values[index], b: seriesB.values[index], month: months[index], x: point.x })}
            r="10"
            tabIndex="0"
          />
        ))}
        <circle cx={lastA.x} cy={lastA.y} fill={seriesA.color} pointerEvents="none" r="3.6" />
        <circle cx={lastB.x} cy={lastB.y} fill={seriesB.color} pointerEvents="none" r="3.6" />
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={paddingTop + 3}>
          {formatCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={plotBottom}>
          {formatCompact(min)}
        </text>
        <text className="mini-axis-month" textAnchor="start" x={paddingLeft} y={height - 4}>
          {formatMonthLabel(months[0])}
        </text>
        <text className="mini-axis-month" textAnchor="end" x={width - paddingRight} y={height - 4}>
          {formatMonthLabel(months.at(-1))}
        </text>
      </svg>
      {tooltip && (
        <div className="mini-trend-tooltip compare-tooltip" style={{ left: `${(tooltip.x / width) * 100}%` }}>
          <span>{formatMonthLabel(tooltip.month)}</span>
          <span className="mini-tooltip-row">
            <i style={{ background: seriesA.color }} />
            {seriesA.label} {formatValue(tooltip.a)}
          </span>
          <span className="mini-tooltip-row">
            <i style={{ background: seriesB.color }} />
            {seriesB.label} {formatValue(tooltip.b)}
          </span>
        </div>
      )}
    </div>
  )
}

function ExpenseCategoryLegend({ activeFields, categoryFields, onToggle }) {
  return (
    <div className="expense-chart-legend">
      {categoryFields.map((field) => {
        const active = activeFields.includes(field.field)
        return (
          <button
            aria-pressed={active}
            className={active ? 'expense-legend-toggle' : 'expense-legend-toggle inactive'}
            key={field.field}
            onClick={() => onToggle(field.field)}
            type="button"
          >
            <i style={{ background: field.color }} />
            {field.label}
          </button>
        )
      })}
      <span className="expense-avg-legend">
        <i />
        Media móvil
      </span>
    </div>
  )
}

function ExpenseCategoryChart({
  ariaLabel = 'Evolución de gastos por categoría, con media móvil de 3 periodos',
  categoryFields,
  getLabel = (row) => formatMonthLabel(row.Mes),
  maxLabels = 6,
  rows,
}) {
  const [containerRef, width] = useElementWidth(720)
  const [tooltip, setTooltip] = useState(null)
  const height = 240
  const paddingRight = 12
  const paddingTop = 16
  const paddingBottom = 26

  const positiveTotals = rows.map((row) => categoryFields.reduce((sum, field) => sum + Math.max(0, asNumber(row[field.field])), 0))
  const negativeTotals = rows.map((row) => categoryFields.reduce((sum, field) => sum + Math.min(0, asNumber(row[field.field])), 0))
  const netTotals = rows.map((row) => categoryFields.reduce((sum, field) => sum + asNumber(row[field.field]), 0))
  const max = Math.max(...positiveTotals, 1)
  const min = Math.min(...negativeTotals, 0)
  const paddingLeft = axisPaddingLeft(min, max)
  const plotWidth = Math.max(0, width - paddingLeft - paddingRight)
  const plotHeight = height - paddingTop - paddingBottom
  const slot = rows.length > 0 ? plotWidth / rows.length : 0
  const barWidth = Math.max(6, slot - 10)
  const scaleY = (value) => height - paddingBottom - ((value - min) / (max - min || 1)) * plotHeight
  const zeroY = scaleY(0)

  const bars = rows.map((row, index) => {
    const slotX = paddingLeft + index * slot
    const barX = slotX + (slot - barWidth) / 2
    let posCursor = 0
    let negCursor = 0
    const segments = []
    categoryFields.forEach((field) => {
      const value = asNumber(row[field.field])
      if (value > 0) {
        const y0 = scaleY(posCursor)
        const y1 = scaleY(posCursor + value)
        segments.push({ color: field.color, height: Math.max(0, y0 - y1), label: field.label, value, y: y1 })
        posCursor += value
      } else if (value < 0) {
        const y0 = scaleY(negCursor)
        const y1 = scaleY(negCursor + value)
        segments.push({ color: field.color, height: Math.max(0, y1 - y0), label: field.label, value, y: y0 })
        negCursor += value
      }
    })
    return { barX, hitWidth: slot, hitX: slotX, key: `${row.Mes}-${row.MesFin ?? ''}`, label: getLabel(row), segments }
  })

  const averageValues = movingAverage(netTotals)
  const averagePoints = bars.map((bar, index) => ({ x: bar.barX + barWidth / 2, y: scaleY(averageValues[index]) }))
  const averagePath = smoothLinePath(averagePoints)
  const labelStep = Math.max(1, Math.ceil(rows.length / maxLabels))

  return (
    <div className="expense-chart" onMouseLeave={() => setTooltip(null)} ref={containerRef}>
      <svg aria-label={ariaLabel} className="mini-trend-svg" height={height} role="img" viewBox={`0 0 ${width} ${height}`}>
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            className="mini-grid-line"
            key={fraction}
            x1={paddingLeft}
            x2={width - paddingRight}
            y1={paddingTop + plotHeight * fraction}
            y2={paddingTop + plotHeight * fraction}
          />
        ))}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        {bars.map((bar) => (
          <g key={bar.key}>
            {bar.segments.map((segment) => (
              <rect fill={segment.color} height={segment.height} key={segment.label} rx="1.5" width={barWidth} x={bar.barX} y={segment.y} />
            ))}
          </g>
        ))}
        <path d={averagePath} fill="none" stroke="#1c2a22" strokeDasharray="5 4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        {averagePoints.map((point, index) => (
          <circle cx={point.x} cy={point.y} fill="#1c2a22" key={bars[index]?.key ?? index} pointerEvents="none" r="2.6" />
        ))}
        {bars.map((bar) => (
          <rect
            aria-label={`${bar.label}: ${formatMoney(bar.segments.reduce((sum, segment) => sum + segment.value, 0))} en total`}
            fill="transparent"
            height={plotHeight}
            key={`hit-${bar.key}`}
            onFocus={() => setTooltip(bar)}
            onMouseEnter={() => setTooltip(bar)}
            role="img"
            tabIndex="0"
            width={bar.hitWidth}
            x={bar.hitX}
            y={paddingTop}
          />
        ))}
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={paddingTop + 3}>
          {formatMoneyCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={height - paddingBottom}>
          {formatMoneyCompact(min)}
        </text>
        {bars.map((bar, index) =>
          index % labelStep === 0 ? (
            <text className="mini-axis-month" key={bar.key} textAnchor="middle" x={bar.barX + barWidth / 2} y={height - 8}>
              {bar.label}
            </text>
          ) : null,
        )}
      </svg>
      {tooltip && (
        <div className="mini-trend-tooltip expense-chart-tooltip" style={{ left: `${((tooltip.barX + barWidth / 2) / width) * 100}%` }}>
          <span>{tooltip.label}</span>
          {tooltip.segments.map((segment) => (
            <span className="mini-tooltip-row" key={segment.label}>
              <i style={{ background: segment.color }} />
              {segment.label} {formatMoney(segment.value)}
            </span>
          ))}
          <strong>Total {formatMoney(tooltip.segments.reduce((sum, segment) => sum + segment.value, 0))}</strong>
        </div>
      )}
    </div>
  )
}

function SavingsAmountChart({ color = '#1f4d3d', getLabel = (row) => formatMonthLabel(row.Mes), maxLabels = 8, negativeColor = '#7a2e2e', rows }) {
  const [containerRef, width] = useElementWidth(720)
  const [tooltip, setTooltip] = useState(null)
  const height = 220
  const paddingRight = 12
  const paddingTop = 16
  const paddingBottom = 26

  const values = rows.map((row) => asNumber(row.value))
  const min = Math.min(...values, 0)
  const max = Math.max(...values, 1)
  const paddingLeft = axisPaddingLeft(min, max)
  const plotWidth = Math.max(0, width - paddingLeft - paddingRight)
  const plotHeight = height - paddingTop - paddingBottom
  const slot = rows.length > 0 ? plotWidth / rows.length : 0
  const barWidth = Math.max(6, slot - 10)
  const scaleY = (value) => height - paddingBottom - ((value - min) / (max - min || 1)) * plotHeight
  const zeroY = scaleY(0)

  const bars = rows.map((row, index) => {
    const slotX = paddingLeft + index * slot
    const barX = slotX + (slot - barWidth) / 2
    const value = asNumber(row.value)
    const barY = value >= 0 ? scaleY(value) : zeroY
    return { barHeight: Math.abs(scaleY(value) - zeroY), barX, barY, hitWidth: slot, hitX: slotX, key: `${row.Mes}-${row.MesFin ?? ''}`, label: getLabel(row), value }
  })

  const averageValues = movingAverage(values)
  const averagePoints = bars.map((bar, index) => ({ x: bar.barX + barWidth / 2, y: scaleY(averageValues[index]) }))
  const averagePath = smoothLinePath(averagePoints)
  const labelStep = Math.max(1, Math.ceil(rows.length / maxLabels))

  return (
    <div className="expense-chart" onMouseLeave={() => setTooltip(null)} ref={containerRef}>
      <svg
        aria-label="Cantidad ahorrada por bloques de meses anteriores, con media móvil"
        className="mini-trend-svg"
        height={height}
        role="img"
        viewBox={`0 0 ${width} ${height}`}
      >
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            className="mini-grid-line"
            key={fraction}
            x1={paddingLeft}
            x2={width - paddingRight}
            y1={paddingTop + plotHeight * fraction}
            y2={paddingTop + plotHeight * fraction}
          />
        ))}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        {bars.map((bar) => (
          <rect fill={bar.value >= 0 ? color : negativeColor} height={bar.barHeight} key={bar.key} rx="1.5" width={barWidth} x={bar.barX} y={bar.barY} />
        ))}
        <path d={averagePath} fill="none" stroke="#1c2a22" strokeDasharray="5 4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        {averagePoints.map((point, index) => (
          <circle cx={point.x} cy={point.y} fill="#1c2a22" key={bars[index]?.key ?? index} pointerEvents="none" r="2.6" />
        ))}
        {bars.map((bar) => (
          <rect
            aria-label={`${bar.label}: ${formatMoney(bar.value)}`}
            fill="transparent"
            height={plotHeight}
            key={`hit-${bar.key}`}
            onFocus={() => setTooltip(bar)}
            onMouseEnter={() => setTooltip(bar)}
            role="img"
            tabIndex="0"
            width={bar.hitWidth}
            x={bar.hitX}
            y={paddingTop}
          />
        ))}
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={paddingTop + 3}>
          {formatMoneyCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={height - paddingBottom}>
          {formatMoneyCompact(min)}
        </text>
        {bars.map((bar, index) =>
          index % labelStep === 0 ? (
            <text className="mini-axis-month" key={bar.key} textAnchor="middle" x={bar.barX + barWidth / 2} y={height - 8}>
              {bar.label}
            </text>
          ) : null,
        )}
      </svg>
      {tooltip && (
        <div className="mini-trend-tooltip expense-chart-tooltip" style={{ left: `${((tooltip.barX + barWidth / 2) / width) * 100}%` }}>
          <span>{tooltip.label}</span>
          <strong>{formatMoney(tooltip.value)}</strong>
        </div>
      )}
    </div>
  )
}

export function ResultsPanel({
  isProcessing,
  rows,
  selectedRow,
  selectedObjective,
  previousRow,
  recentRows,
  trendMax,
  result,
  objectiveRows,
  objectiveNames,
  filteredObjectiveRows,
  objectiveTotals,
  onMonthChange,
  onObjectiveFilterChange,
  onExportResult,
  onExportHistoryCsv,
  onExportObjectivesCsv,
}) {
  const [activeTab, setActiveTab] = useState('resumen')
  const [typologyPeriod, setTypologyPeriod] = useState('12')
  const [featuredTypology, setFeaturedTypology] = useState(typologyCards[0].field)
  const [expensePeriod, setExpensePeriod] = useState('6')
  const [activeExpenseCategories, setActiveExpenseCategories] = useState(() => expenseCategoryFields.map((field) => field.field))
  const [savingsPeriod, setSavingsPeriod] = useState('12')
  const [savingsAmountPeriod, setSavingsAmountPeriod] = useState('6')
  const [budgetTooltip, setBudgetTooltip] = useState(null)
  const budget = selectedRow ? getBudget(selectedRow) : null
  const expenseAnalysis = getExpenseAnalysis(result)
  const incomeAnalysis = getIncomeAnalysis(result)
  const expenseCategoryRows = getExpenseCategoryRowsForPeriod(expenseAnalysis.categorias, selectedRow?.Mes, expensePeriod)
  const expenseCategoryTimeline = getExpenseRowsForPeriod(expenseAnalysis.categorias, selectedRow?.Mes, expensePeriod)
  const expensePieTotal = expenseCategoryRows.reduce((total, row) => total + asNumber(row.total), 0)
  let expensePieCursor = 0
  const expensePieSegments = expenseCategoryRows.map((row) => {
    const field = expenseCategoryFields.find((item) => item.field === row.categoria)
    const pct = expensePieTotal > 0 ? (asNumber(row.total) / expensePieTotal) * 100 : 0
    const segment = {
      color: field?.color ?? '#8a8a78',
      end: expensePieCursor + pct,
      label: field?.label ?? row.categoria,
      pct,
      start: expensePieCursor,
      total: asNumber(row.total),
    }
    expensePieCursor += pct
    return segment
  })
  const expensePieGradient =
    expensePieSegments.length > 0
      ? `conic-gradient(${expensePieSegments.map((segment) => `${segment.color} ${segment.start}% ${segment.end}%`).join(', ')})`
      : null
  const expenseFullTimeline = getExpenseRowsForPeriod(expenseAnalysis.categorias, selectedRow?.Mes, 'all')
  const expenseBucketSize = expensePeriod === 'all' ? Math.max(1, Math.ceil(expenseFullTimeline.length / 8)) : Number(expensePeriod)
  const expenseBucketedRows = bucketExpenseCategoryRows(expenseFullTimeline, expenseBucketSize, 8, expenseCategoryFields)
  const toggleExpenseCategory = (field) =>
    setActiveExpenseCategories((current) => {
      if (current.includes(field)) {
        const next = current.filter((item) => item !== field)
        return next.length > 0 ? next : current
      }
      return [...current, field]
    })
  const visibleExpenseCategoryFields = expenseCategoryFields.filter((field) => activeExpenseCategories.includes(field.field))
  const savingsAnalysis = getSavingsAnalysis(result)
  const scopedSavingsRows = savingsAnalysis.mensual.filter((row) => !selectedRow?.Mes || String(row.Mes) <= String(selectedRow.Mes))
  const selectedSavingsRow = scopedSavingsRows.at(-1) ?? savingsAnalysis.ultimo_mes
  const savingsPercentRows = getPeriodRows(scopedSavingsRows, savingsPeriod)
  const savingsMonths = savingsPercentRows.map((row) => row.Mes)
  const savingsPercentValues = savingsPercentRows.map((row) => asNumber(row.porcentaje_ahorro))
  const savingsAverageValues = movingAverage(savingsPercentValues)
  const savingsBucketSize =
    savingsAmountPeriod === 'all' ? Math.max(1, Math.ceil(scopedSavingsRows.length / 8)) : Number(savingsAmountPeriod)
  const savingsBucketedRows = bucketSingleValueRows(scopedSavingsRows, 'balance', savingsBucketSize, 8)
  const investmentRows = rows.slice(-12)
  const investmentMonths = investmentRows.map((row) => row.Mes)
  const investmentReserveValues = investmentRows.map((row) => asNumber(row['📈 Inversiones']))
  const investmentInvestedValues = investmentRows.map((row) => asNumber(row['Dinero Invertido']))
  const reserveMax = Math.max(
    ...rows.map((row) =>
      Math.max(
        Math.abs(asNumber(row['🎁 Regalos'])),
        Math.abs(asNumber(row['💼 Vacaciones'])),
        Math.abs(asNumber(row['Fondo de reserva cargado'])),
      ),
    ),
    1,
  )
  const typologyRows = getPeriodRows(rows, typologyPeriod)
  const typologyMonths = typologyRows.map((row) => row.Mes)
  const featuredCard = typologyCards.find((card) => card.field === featuredTypology) ?? typologyCards[0]
  const featuredValues = typologyRows.map((row) => asNumber(row[featuredCard.field]))
  const otherTypologyCards = typologyCards.filter((card) => card.field !== featuredCard.field)
  const summaryGroups = selectedRow ? getSummaryGroups(selectedRow) : null
  const totalMoney = asNumber(selectedRow?.total)
  const investedMoney = asNumber(selectedRow?.['Dinero Invertido'])
  const investedPct = totalMoney > 0 ? (investedMoney / totalMoney) * 100 : 0
  const interestAmount = getCategoryTotalUntilMonth(incomeAnalysis, selectedRow?.Mes, ['interes', 'intereses'])
  const interestPct = investedMoney > 0 ? (Math.abs(interestAmount) / investedMoney) * 100 : 0
  const investmentDividends = asNumber(selectedRow?.Dividendos)
  const investmentInterest = Math.max(0, interestAmount - investmentDividends)
  const dividendCompanyBreakdown = aggregateDividendsByCompany(getDividendPayments(result), { untilMonth: selectedRow?.Mes })
  const dividendCompanyMax = Math.max(1, ...dividendCompanyBreakdown.map((row) => asNumber(row.total)))

  return (
    <section className="panel result-panel" aria-busy={isProcessing}>
      <div className="panel-header dashboard-header">
        <div>
          <span className="section-kicker">Vista mensual</span>
          <h2>Finanzas</h2>
          <p>{selectedRow ? `Lectura consolidada de ${formatMonthLong(selectedRow.Mes)}` : 'Procesa un Excel para consultar tus datos'}</p>
        </div>
        <div className="panel-actions">
          {rows.length > 0 ? (
            <label className="month-selector">
              <CalendarDays aria-hidden="true" size={17} />
              <span>Periodo</span>
              <select value={selectedRow?.Mes ?? ''} onChange={(event) => onMonthChange(event.target.value)}>
                {rows.map((row) => (
                  <option key={row.Mes} value={row.Mes}>
                    {row.Mes}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span>Pendiente</span>
          )}
          {result && (
            <details className="export-menu">
              <summary>
                <Download aria-hidden="true" size={17} />
                Exportar
              </summary>
              <div className="export-actions">
                <button className="text-button" type="button" onClick={onExportResult}>
                  <FileJson aria-hidden="true" size={16} />
                  Resultado JSON
                </button>
                <button className="text-button" type="button" onClick={onExportHistoryCsv}>
                  <Table2 aria-hidden="true" size={16} />
                  Historial CSV
                </button>
                {objectiveRows.length > 0 && (
                  <button className="text-button" type="button" onClick={onExportObjectivesCsv}>
                    <Table2 aria-hidden="true" size={16} />
                    Objetivos CSV
                  </button>
                )}
              </div>
            </details>
          )}
        </div>
      </div>

      {isProcessing ? (
        <div className="empty-state processing-state" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <strong>Procesando Excel</strong>
          <span>El resumen se actualizará cuando termine el cálculo.</span>
        </div>
      ) : selectedRow ? (
        <>
          <div className="tabs-nav" role="tablist" aria-label="Análisis disponibles">
            {tabs.map((tab) => (
              <button
                aria-selected={activeTab === tab.id}
                className={activeTab === tab.id ? 'tab-button active' : 'tab-button'}
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                type="button"
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="tab-content" role="tabpanel">
            {activeTab === 'resumen' && (
              <>
                <section className="summary-dashboard">
                  <div className="summary-groups">
                    <div className="summary-group">
                      <h3>Patrimonio</h3>
                      <div className="summary-metrics">
                        {summaryGroups.patrimony.map((item) => (
                          <article className={item.label === 'Total' ? 'metric total-metric' : 'metric'} key={item.label}>
                            <span>{item.label}</span>
                            <strong>{formatMoney(item.value)}</strong>
                          </article>
                        ))}
                      </div>
                    </div>

                    <div className="summary-group">
                      <h3>Presupuesto</h3>
                      <div className="summary-metrics budget-metrics">
                        {summaryGroups.budget.map((item) => (
                          <article className="metric" key={item.label}>
                            <span>{item.label}</span>
                            <strong>{formatMoney(item.value)}</strong>
                          </article>
                        ))}
                      </div>
                    </div>
                  </div>

                  <aside className="summary-ratios">
                    <article>
                      <span>Dinero invertido sobre total</span>
                      <strong>{formatPercent(investedPct)}</strong>
                      <div className="ratio-bar">
                        <span style={{ width: `${Math.max(0, Math.min(100, investedPct))}%` }} />
                      </div>
                      <small>
                        {formatMoney(investedMoney)} de {formatMoney(totalMoney)}
                      </small>
                    </article>
                    <article className="interest-ratio">
                      <div
                        className="mini-donut"
                        style={{
                          '--donut-value': `${Math.max(0, Math.min(100, interestPct))}%`,
                        }}
                      >
                        <strong>{formatPercent(interestPct)}</strong>
                      </div>
                      <div>
                        <span>Intereses sobre dinero invertido</span>
                        <strong>{formatMoney(Math.abs(interestAmount))}</strong>
                      </div>
                    </article>
                  </aside>
                </section>

                {previousRow && (
                  <div className="comparison-grid">
                    {comparisonRows.map(({ label, field }) => {
                      const delta = asNumber(selectedRow[field]) - asNumber(previousRow[field])
                      return (
                        <article className={delta >= 0 ? 'comparison-item positive' : 'comparison-item negative'} key={field}>
                          <span>
                            {label} vs {previousRow.Mes}
                          </span>
                          <strong>{formatDelta(delta)}</strong>
                        </article>
                      )
                    })}
                  </div>
                )}

                <div className="movement-strip">
                  <span>{result.movimientos.gastos} gastos</span>
                  <span>{result.movimientos.ingresos} ingresos</span>
                  <span>{result.movimientos.transferencias} transferencias</span>
                  <span>{result.movimientos.cuentas} cuentas</span>
                  <span>{result.historial.objetivos.length} objetivos</span>
                </div>

                <section className="trend-panel">
                  <h3 className="table-title">Tendencia</h3>
                  <div className="trend-list">
                    {recentRows.map((row) => (
                      <article className="trend-row" key={row.Mes}>
                        <span className="trend-month">{row.Mes}</span>
                        <div className="trend-bars">
                          <div className="trend-bar total">
                            <span style={{ width: `${Math.max(3, (Math.abs(asNumber(row.total)) / trendMax) * 100)}%` }} />
                          </div>
                          <div className="trend-bar gasto">
                            <span style={{ width: `${Math.max(3, (Math.abs(asNumber(row['💳 Gasto del mes'])) / trendMax) * 100)}%` }} />
                          </div>
                          <div className="trend-bar presupuesto">
                            <span style={{ width: `${Math.max(3, (Math.abs(asNumber(row['💸 Presupuesto Mes'])) / trendMax) * 100)}%` }} />
                          </div>
                        </div>
                        <strong>{formatMoney(row.total)}</strong>
                      </article>
                    ))}
                  </div>
                  <div className="trend-legend">
                    <span>Total</span>
                    <span>Gasto</span>
                    <span>Presupuesto</span>
                  </div>
                </section>
              </>
            )}

            {activeTab === 'presupuesto' && budget && (
              <section className="budget-layout">
                <div className="budget-card budget-visual">
                  <div>
                    <h3 className="table-title">Presupuesto de {selectedRow.Mes}</h3>
                    <p className="muted-text">
                      {budget.overrun > 0
                        ? `Exceso de ${formatMoney(budget.overrun)} sobre el presupuesto disponible.`
                        : `Quedan ${formatMoney(budget.remaining)} disponibles este mes.`}
                    </p>
                  </div>
                  <div className="budget-bar-wrap">
                    <div className="budget-bar" aria-label="Distribución del presupuesto mensual">
                      {budget.debtReserve > 0 && (
                        <span
                          aria-label={`Reserva deuda: ${formatMoney(budget.debtReserve)}`}
                          className="budget-segment reserved"
                          onBlur={() => setBudgetTooltip(null)}
                          onFocus={() => setBudgetTooltip({ label: 'Reserva deuda', value: budget.debtReserve })}
                          onMouseEnter={() => setBudgetTooltip({ label: 'Reserva deuda', value: budget.debtReserve })}
                          onMouseLeave={() => setBudgetTooltip(null)}
                          role="img"
                          style={{ width: getSegmentWidth(budget.debtReserve, budget.totalForBar) }}
                          tabIndex="0"
                        />
                      )}
                      {budget.budgetConsumption > 0 && (
                        <span
                          aria-label={`Gasto: ${formatMoney(budget.spent)}`}
                          className="budget-segment spent"
                          onBlur={() => setBudgetTooltip(null)}
                          onFocus={() => setBudgetTooltip({ label: 'Gasto', value: budget.spent })}
                          onMouseEnter={() => setBudgetTooltip({ label: 'Gasto', value: budget.spent })}
                          onMouseLeave={() => setBudgetTooltip(null)}
                          role="img"
                          style={{ width: getSegmentWidth(budget.budgetConsumption, budget.totalForBar) }}
                          tabIndex="0"
                        />
                      )}
                      {budget.remaining > 0 && (
                        <span
                          aria-label={`Restante: ${formatMoney(budget.remaining)}`}
                          className="budget-segment remaining"
                          onBlur={() => setBudgetTooltip(null)}
                          onFocus={() => setBudgetTooltip({ label: 'Restante', value: budget.remaining })}
                          onMouseEnter={() => setBudgetTooltip({ label: 'Restante', value: budget.remaining })}
                          onMouseLeave={() => setBudgetTooltip(null)}
                          role="img"
                          style={{ width: getSegmentWidth(budget.remaining, budget.totalForBar) }}
                          tabIndex="0"
                        />
                      )}
                    </div>
                    {budgetTooltip && (
                      <div className="mini-trend-tooltip" style={{ left: '50%' }}>
                        <span>{budgetTooltip.label}</span>
                        <strong>{formatMoney(budgetTooltip.value)}</strong>
                      </div>
                    )}
                  </div>
                  <div className="budget-bar-values">
                    {budget.debtReserve > 0 ? <span>Reserva deuda {formatMoney(budget.debtReserve)}</span> : null}
                    <span>Gasto neto {formatMoney(budget.spent)}</span>
                    {budget.budgetRefund > 0 ? <span>Ajuste a favor {formatMoney(budget.budgetRefund)}</span> : null}
                    <span>Disponible {formatMoney(budget.remaining)}</span>
                  </div>
                  <div className="budget-legend">
                    <span>Reserva deuda</span>
                    <span>Gasto</span>
                    <span>Disponible</span>
                  </div>
                </div>

                <div className="budget-summary">
                  <p className="budget-summary-heading">Presupuesto</p>
                  <article>
                    <span>Presupuesto mes</span>
                    <strong>{formatMoney(budget.monthBudget)}</strong>
                  </article>
                  <article>
                    <span>Gasto neto</span>
                    <strong>{formatMoney(budget.spent)}</strong>
                  </article>
                  <article>
                    <span>Disponible visual</span>
                    <strong>{formatMoney(budget.remaining)}</strong>
                  </article>
                  <article>
                    <span>Ejecutado</span>
                    <strong>{budget.execution.toFixed(1)}%</strong>
                    <div className="ratio-bar">
                      <span style={{ width: `${Math.max(0, Math.min(100, budget.execution))}%` }} />
                    </div>
                  </article>
                  <p className="budget-summary-heading">Deuda</p>
                  <article>
                    <span>Reserva deuda</span>
                    <strong>{formatMoney(budget.debtReserve)}</strong>
                  </article>
                  <article>
                    <span>Exceso mes</span>
                    <strong>{formatMoney(budget.monthlyDebt)}</strong>
                  </article>
                  <article>
                    <span>Exceso acumulado</span>
                    <strong>{formatMoney(budget.accumulatedDebt)}</strong>
                  </article>
                </div>
              </section>
            )}

            {activeTab === 'gastos' && (
              expenseCategoryTimeline.length > 0 ? (
                <section className="expenses-layout">
                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Gastos por categoría</h3>
                      <label className="period-selector">
                        <span>Periodo</span>
                        <select value={expensePeriod} onChange={(event) => setExpensePeriod(event.target.value)}>
                          {expensePeriods.map((period) => (
                            <option key={period.value} value={period.value}>
                              {period.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    {expensePieGradient ? (
                      <div className="expense-pie-layout">
                        <div className="expense-pie" style={{ background: expensePieGradient }}>
                          <div className="expense-pie-hole">
                            <strong>{formatMoney(expensePieTotal)}</strong>
                            <span>Total del periodo</span>
                          </div>
                        </div>
                        <ul className="expense-pie-legend">
                          {expensePieSegments.map((segment) => (
                            <li key={segment.label}>
                              <span className="expense-pie-legend-label">
                                <i style={{ background: segment.color }} />
                                {segment.label}
                              </span>
                              <span className="expense-pie-legend-value">{formatMoney(segment.total)}</span>
                              <span className="expense-pie-legend-pct">{formatPercent(segment.pct)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <div className="empty-state compact-empty">
                        <strong>Sin gasto en el periodo</strong>
                        <span>Elige un periodo con gastos registrados para ver el reparto.</span>
                      </div>
                    )}
                  </div>

                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Evolución por categoría</h3>
                      <span className="muted-text">Media móvil de 3 meses</span>
                    </div>
                    <ExpenseCategoryChart categoryFields={visibleExpenseCategoryFields} rows={expenseCategoryTimeline} />
                    <ExpenseCategoryLegend
                      activeFields={activeExpenseCategories}
                      categoryFields={expenseCategoryFields}
                      onToggle={toggleExpenseCategory}
                    />
                  </div>

                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Comparativa por bloques</h3>
                      <span className="muted-text">
                        Bloques de {expenseBucketSize} {expenseBucketSize === 1 ? 'mes' : 'meses'}, últimos {expenseBucketedRows.length}
                      </span>
                    </div>
                    <ExpenseCategoryChart
                      ariaLabel="Gastos por categoría agrupados en bloques de meses anteriores, con media móvil"
                      categoryFields={visibleExpenseCategoryFields}
                      getLabel={formatExpenseBucketLabel}
                      maxLabels={8}
                      rows={expenseBucketedRows}
                    />
                    <ExpenseCategoryLegend
                      activeFields={activeExpenseCategories}
                      categoryFields={expenseCategoryFields}
                      onToggle={toggleExpenseCategory}
                    />
                  </div>
                </section>
              ) : (
                <div className="empty-state compact-empty">
                  <strong>Sin análisis de gastos</strong>
                  <span>El Excel procesado no contiene datos suficientes para agrupar gastos por categoría.</span>
                </div>
              )
            )}

            {activeTab === 'ahorro' && (
              savingsAnalysis.mensual.length > 0 ? (
                <section className="savings-layout">
                  <div className="savings-summary executive-kpis">
                    <article className="tone-income">
                      <span>Ingresos</span>
                      <strong>{formatMoney(selectedSavingsRow?.ingresos)}</strong>
                    </article>
                    <article className="tone-expense">
                      <span>Gastos</span>
                      <strong>{formatMoney(selectedSavingsRow?.gastos)}</strong>
                    </article>
                    <article className="tone-balance">
                      <span>Balance</span>
                      <strong>{formatMoney(selectedSavingsRow?.balance)}</strong>
                    </article>
                    <article className="tone-savings">
                      <span>Ahorro</span>
                      <strong>{asNumber(selectedSavingsRow?.porcentaje_ahorro).toFixed(1)}%</strong>
                      <small>
                        {asNumber(selectedSavingsRow?.gastos) === 0 && asNumber(selectedSavingsRow?.ingresos) > 0
                          ? 'Sin gastos registrados'
                          : 'Sobre los ingresos del mes'}
                      </small>
                    </article>
                  </div>

                  <section className="typology-chart-panel">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Porcentaje de ahorro</h3>
                        <span className="muted-text">Comparado con la media móvil de 3 meses</span>
                      </div>
                      <label className="period-selector">
                        <span>Periodo</span>
                        <select value={savingsPeriod} onChange={(event) => setSavingsPeriod(event.target.value)}>
                          {expensePeriods.map((period) => (
                            <option key={period.value} value={period.value}>
                              {period.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className="typology-compare">
                      <div className="typology-compare-legend">
                        <span>
                          <i style={{ background: '#1f4d3d' }} />
                          Ahorro mensual
                        </span>
                        <span>
                          <i className="dashed" style={{ borderColor: '#1c2a22' }} />
                          Media móvil
                        </span>
                      </div>
                      <TypologyCompareChart
                        formatCompact={formatPercent}
                        formatValue={formatPercent}
                        height={160}
                        months={savingsMonths}
                        seriesA={{ color: '#1f4d3d', label: 'Ahorro mensual', values: savingsPercentValues }}
                        seriesB={{ color: '#1c2a22', label: 'Media móvil', values: savingsAverageValues }}
                      />
                    </div>
                  </section>

                  <section className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Cantidad ahorrada</h3>
                        <span className="muted-text">
                          Bloques de {savingsBucketSize} {savingsBucketSize === 1 ? 'mes' : 'meses'}, últimos {savingsBucketedRows.length}
                        </span>
                      </div>
                      <label className="period-selector">
                        <span>Periodo</span>
                        <select value={savingsAmountPeriod} onChange={(event) => setSavingsAmountPeriod(event.target.value)}>
                          {expensePeriods.map((period) => (
                            <option key={period.value} value={period.value}>
                              {period.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <SavingsAmountChart getLabel={formatExpenseBucketLabel} rows={savingsBucketedRows} />
                    <div className="expense-chart-legend">
                      <span>
                        <i style={{ background: '#1f4d3d' }} />
                        Ahorro positivo
                      </span>
                      <span>
                        <i style={{ background: '#7a2e2e' }} />
                        Ahorro negativo
                      </span>
                      <span className="expense-avg-legend">
                        <i />
                        Media móvil
                      </span>
                    </div>
                  </section>
                </section>
              ) : (
                <div className="empty-state compact-empty">
                  <strong>Sin análisis de ahorro</strong>
                  <span>El Excel procesado no contiene ingresos suficientes para calcular ahorro mensual.</span>
                </div>
              )
            )}

            {activeTab === 'inversiones' && (
              <section className="investment-layout">
                <div className="investment-summary">
                  <article>
                    <span>Bolsa inversiones</span>
                    <strong>{formatMoney(selectedRow['📈 Inversiones'])}</strong>
                  </article>
                  <article>
                    <span>Dinero invertido</span>
                    <strong>{formatMoney(selectedRow['Dinero Invertido'])}</strong>
                  </article>
                  <article>
                    <span>Intereses</span>
                    <strong>{formatMoney(investmentInterest)}</strong>
                  </article>
                  <article>
                    <span>Dividendos</span>
                    <strong>{formatMoney(investmentDividends)}</strong>
                  </article>
                </div>

                <section className="typology-chart-panel">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Evolución inversiones</h3>
                      <span className="muted-text">Bolsa disponible frente a dinero invertido</span>
                    </div>
                  </div>
                  <div className="typology-compare">
                    <div className="typology-compare-legend">
                      <span>
                        <i style={{ background: '#1f5c6b' }} />
                        Bolsa
                      </span>
                      <span>
                        <i className="dashed" style={{ borderColor: '#1f4d3d' }} />
                        Invertido
                      </span>
                    </div>
                    <TypologyCompareChart
                      height={200}
                      months={investmentMonths}
                      seriesA={{ color: '#1f5c6b', label: 'Bolsa', values: investmentReserveValues }}
                      seriesB={{ color: '#1f4d3d', label: 'Invertido', values: investmentInvestedValues }}
                    />
                  </div>
                </section>

                <section className="typology-chart-panel">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Dividendos por empresa</h3>
                      <span className="muted-text">Acumulado hasta {selectedRow?.Mes ?? 'el periodo seleccionado'}</span>
                    </div>
                  </div>
                  {dividendCompanyBreakdown.length > 0 ? (
                    <ul className="dividend-company-list">
                      {dividendCompanyBreakdown.map((row) => (
                        <li key={row.codigo}>
                          <div className="dividend-company-row-head">
                            <span className="dividend-company-name">{row.empresa}</span>
                            <span className="dividend-company-amount">{formatMoney(row.total)}</span>
                          </div>
                          <div className="dividend-company-bar-track">
                            <div
                              className="dividend-company-bar-fill"
                              style={{ width: `${Math.max(4, (asNumber(row.total) / dividendCompanyMax) * 100)}%` }}
                            />
                          </div>
                          <span className="dividend-company-meta">
                            {row.pagos} {row.pagos === 1 ? 'pago' : 'pagos'} · último el {formatDateEs(row.ultimo_pago)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="empty-state compact-empty">
                      <strong>Sin dividendos etiquetados</strong>
                      <span>Etiqueta un ingreso de categoría Interés con "Dividendos" y el nombre de la empresa para verlo aquí.</span>
                    </div>
                  )}
                </section>
              </section>
            )}

            {activeTab === 'reservas' && (
              <section className="reserve-layout">
                <div className="reserve-summary">
                  <article>
                    <span>Regalos</span>
                    <strong>{formatMoney(selectedRow['🎁 Regalos'])}</strong>
                  </article>
                  <article>
                    <span>Vacaciones</span>
                    <strong>{formatMoney(selectedRow['💼 Vacaciones'])}</strong>
                  </article>
                  <article>
                    <span>Fondo reserva</span>
                    <strong>{formatMoney(selectedRow['Fondo de reserva cargado'])}</strong>
                  </article>
                </div>

                <section className="trend-panel compact-trend">
                  <h3 className="table-title">Evolución reservas</h3>
                  <div className="trend-list">
                    {rows.slice(-12).map((row) => (
                      <article className="reserve-row" key={row.Mes}>
                        <span className="trend-month">{row.Mes}</span>
                        <div className="reserve-bars">
                          <div className="reserve-bar gifts">
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row['🎁 Regalos'])) / reserveMax) * 100)}%` }} />
                          </div>
                          <div className="reserve-bar holidays">
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row['💼 Vacaciones'])) / reserveMax) * 100)}%` }} />
                          </div>
                          <div className="reserve-bar emergency">
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row['Fondo de reserva cargado'])) / reserveMax) * 100)}%` }} />
                          </div>
                        </div>
                        <strong>{formatMoney(asNumber(row['🎁 Regalos']) + asNumber(row['💼 Vacaciones']) + asNumber(row['Fondo de reserva cargado']))}</strong>
                      </article>
                    ))}
                  </div>
                  <div className="reserve-legend">
                    <span>Regalos</span>
                    <span>Vacaciones</span>
                    <span>Fondo reserva</span>
                  </div>
                </section>
              </section>
            )}

            {activeTab === 'tipologias' && (
              <section className="typology-layout">
                <div className="typology-chart-panel">
                  <div className="table-toolbar compact-toolbar">
                    <h3 className="table-title">Evolución por tipología</h3>
                    <div className="typology-toolbar-controls">
                      <label className="period-selector">
                        <span>Mostrar en grande</span>
                        <select value={featuredTypology} onChange={(event) => setFeaturedTypology(event.target.value)}>
                          {typologyCards.map((card) => (
                            <option key={card.field} value={card.field}>
                              {card.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="period-selector">
                        <span>Periodo</span>
                        <select value={typologyPeriod} onChange={(event) => setTypologyPeriod(event.target.value)}>
                          {typologyPeriods.map((period) => (
                            <option key={period.value} value={period.value}>
                              {period.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                  {typologyRows.length > 0 ? (
                    <div className="typology-board">
                      <article className="typology-hero">
                        <div className="typology-hero-head">
                          <span className="typology-hero-label">{featuredCard.label}</span>
                          <strong className="typology-hero-value">{formatMoney(featuredValues.at(-1))}</strong>
                          <span className="typology-hero-delta">
                            {formatDelta(featuredValues.at(-1) - featuredValues[0])} en {typologyRows.length} meses
                          </span>
                        </div>
                        <TypologyMiniChart
                          color={featuredCard.color}
                          height={200}
                          label={featuredCard.label}
                          months={typologyMonths}
                          values={featuredValues}
                        />
                      </article>

                      <article className="typology-compare">
                        <div className="typology-compare-legend">
                          <span>
                            <i style={{ background: typologyCompareFields[0].color }} />
                            {typologyCompareFields[0].label}
                          </span>
                          <span>
                            <i className="dashed" style={{ borderColor: typologyCompareFields[1].color }} />
                            {typologyCompareFields[1].label}
                          </span>
                        </div>
                        <TypologyCompareChart
                          months={typologyMonths}
                          seriesA={{
                            ...typologyCompareFields[0],
                            values: typologyRows.map((row) => asNumber(row[typologyCompareFields[0].field])),
                          }}
                          seriesB={{
                            ...typologyCompareFields[1],
                            values: typologyRows.map((row) => asNumber(row[typologyCompareFields[1].field])),
                          }}
                        />
                      </article>

                      <div className="typology-grid">
                        {otherTypologyCards.map(({ field, label, color }) => {
                          const values = typologyRows.map((row) => asNumber(row[field]))
                          return (
                            <article className="typology-mini" key={field}>
                              <div className="typology-mini-head">
                                <div className="typology-mini-head-info">
                                  <span>{label}</span>
                                  <strong>{formatMoney(values.at(-1))}</strong>
                                </div>
                                <TypologyFeatureButton label={label} onFeature={() => setFeaturedTypology(field)} />
                              </div>
                              <TypologyMiniChart color={color} height={72} label={label} months={typologyMonths} values={values} />
                            </article>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="empty-state compact-empty">
                      <strong>Sin histórico suficiente</strong>
                      <span>Procesa al menos un mes para ver la evolución.</span>
                    </div>
                  )}
                </div>

                <div className="typology-predictions">
                  <h3 className="table-title">Predicción a 6 meses</h3>
                  <div className="prediction-grid">
                    {typologyFields.map(({ field, label, color }) => {
                      const prediction = predictTypology(typologyRows, field)
                      return (
                        <article className="prediction-card" key={field} style={{ '--prediction-color': color }}>
                          <strong>{label}</strong>
                          <span>Base {formatMoney(prediction.base)}</span>
                          <span>Conservadora {formatMoney(prediction.conservative)}</span>
                          <span>Optimista {formatMoney(prediction.optimistic)}</span>
                        </article>
                      )
                    })}
                  </div>
                </div>
              </section>
            )}

            {activeTab === 'objetivos' && (
              objectiveRows.length > 0 ? (
                <div className="table-wrap compact-table-wrap">
                  <div className="table-toolbar">
                    <h3 className="table-title">Objetivos</h3>
                    <label>
                      <span>Filtro</span>
                      <select value={selectedObjective} onChange={(event) => onObjectiveFilterChange(event.target.value)}>
                        <option value="all">Todos</option>
                        {objectiveNames.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="objective-summary">
                    <article>
                      <span>Aporte</span>
                      <strong>{formatMoney(objectiveTotals.aporte)}</strong>
                    </article>
                    <article>
                      <span>Gasto etiquetado</span>
                      <strong>{formatMoney(objectiveTotals.gasto)}</strong>
                    </article>
                    <article>
                      <span>Liquidación</span>
                      <strong>{formatMoney(objectiveTotals.liquidacion)}</strong>
                    </article>
                  </div>
                  <table>
                    <thead>
                      <tr>
                        <th>Mes</th>
                        <th>Objetivo</th>
                        <th>Aporte</th>
                        <th>Saldo</th>
                        <th>Liquidación</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredObjectiveRows.slice(-12).map((row, index) => (
                        <tr key={`${row.Mes}-${row.Objetivo}-${index}`}>
                          <td>{row.Mes}</td>
                          <td>{row.Objetivo}</td>
                          <td>{formatMoney(row.aporte_mes)}</td>
                          <td>{formatMoney(row.saldo_fin_mes)}</td>
                          <td>{formatMoney(row.liquidacion)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state compact-empty">
                  <strong>Sin objetivos calculados</strong>
                  <span>Añade objetivos en la entrada y procesa el Excel para ver su evolución.</span>
                </div>
              )
            )}

            {activeTab === 'datos' && (
              <>
                <div className="movement-strip">
                  <span>{rows.length} meses</span>
                  <span>{result.movimientos.gastos} gastos</span>
                  <span>{result.movimientos.ingresos} ingresos</span>
                  <span>{objectiveRows.length} filas de objetivos</span>
                </div>
                <div className="table-wrap compact-table-wrap">
                  <h3 className="table-title">Historial</h3>
                  <table>
                    <thead>
                      <tr>
                        {historyFields.map((field) => (
                          <th key={field.field}>{field.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.slice(-24).map((row) => (
                        <tr key={row.Mes}>
                          {historyFields.map((field) => (
                            <td key={field.field}>
                              {field.type === 'money' ? formatMoney(row[field.field]) : row[field.field]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </>
      ) : (
        <div className="empty-state">
          <strong>Sin cálculo cargado</strong>
          <span>Cuando proceses un Excel aparecerán aquí el último mes y el historial reciente.</span>
        </div>
      )}
    </section>
  )
}
