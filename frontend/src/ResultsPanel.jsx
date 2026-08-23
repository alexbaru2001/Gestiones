import { useId, useState } from 'react'
import { CalendarDays, Download, FileJson, Maximize2, Table2 } from 'lucide-react'
import { formatDelta, formatMoney } from './formatters'
import { comparisonRows } from './resultSelectors'

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
  { value: '12', label: '12 meses' },
  { value: 'all', label: 'Todo' },
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

function getSvgPoints(values, min, max, options) {
  return getSvgCoordinates(values, min, max, options)
    .map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')
}

function getSvgAreaPath(values, min, max, options = {}) {
  const coordinates = getSvgCoordinates(values, min, max, options)
  if (!coordinates.length) return ''
  const height = options.height ?? 220
  const paddingBottom = options.paddingBottom ?? options.padding ?? 18
  const baseline = height - paddingBottom
  const line = coordinates.map(({ x, y }) => `L ${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')
  return `M ${coordinates[0].x.toFixed(1)} ${baseline.toFixed(1)} ${line} L ${coordinates.at(-1).x.toFixed(1)} ${baseline.toFixed(1)} Z`
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

function getAxisTicks(min, max, count = 5) {
  const range = max - min || 1
  return Array.from({ length: count }, (_, index) => min + (range / Math.max(1, count - 1)) * index)
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
      acc[key] = (acc[key] ?? 0) + Math.abs(asNumber(value))
    })
    return acc
  }, {})
  return Object.entries(totals)
    .map(([categoria, total]) => ({ categoria, total }))
    .filter((row) => row.total > 0)
    .sort((left, right) => right.total - left.total)
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

function axisPaddingLeft(min, max) {
  const chars = Math.max(formatMoneyCompact(min).length, formatMoneyCompact(max).length)
  return Math.min(70, Math.max(34, chars * 6 + 8))
}

function TypologyMiniChart({ values, months, color, label, height = 64 }) {
  const [tooltip, setTooltip] = useState(null)
  const gradientId = useId()
  const width = 240
  const min = Math.min(...values, 0)
  const max = Math.max(...values, 1)
  const paddingLeft = axisPaddingLeft(min, max)
  const paddingRight = 10
  const paddingTop = 12
  const paddingBottom = 18
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinates = getSvgCoordinates(values, min, max, chartOptions)
  const areaPath = getSvgAreaPath(values, min, max, chartOptions)
  const linePoints = getSvgPoints(values, min, max, chartOptions)
  const last = coordinates.at(-1)
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const plotBottom = height - paddingBottom
  const trend = values.at(-1) >= values[0] ? 'tendencia ascendente' : 'tendencia descendente'

  return (
    <div className="mini-trend" onMouseLeave={() => setTooltip(null)}>
      <svg
        aria-label={`${label}: de ${formatMoney(values[0])} a ${formatMoney(values.at(-1))} en ${values.length} meses, ${trend}.`}
        className="mini-trend-svg"
        height={height}
        preserveAspectRatio="none"
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
        <polyline fill="none" points={linePoints} stroke={color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
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
          {formatMoneyCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={plotBottom}>
          {formatMoneyCompact(min)}
        </text>
        <text className="mini-axis-month" textAnchor="start" x={paddingLeft} y={height - 4}>
          {formatMonthLabel(months[0])}
        </text>
        <text className="mini-axis-month" textAnchor="end" x={width - paddingRight} y={height - 4}>
          {formatMonthLabel(months.at(-1))}
        </text>
      </svg>
      {tooltip && (
        <div
          className="mini-trend-tooltip"
          style={{ left: `${(tooltip.x / width) * 100}%`, top: `${(tooltip.y / height) * 100}%` }}
        >
          <span>{formatMonthLabel(tooltip.month)}</span>
          <strong>{formatMoney(tooltip.value)}</strong>
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

function TypologyCompareChart({ height = 100, months, seriesA, seriesB }) {
  const [tooltip, setTooltip] = useState(null)
  const width = 560
  const combined = [...seriesA.values, ...seriesB.values]
  const min = Math.min(...combined, 0)
  const max = Math.max(...combined, 1)
  const paddingLeft = axisPaddingLeft(min, max)
  const paddingRight = 10
  const paddingTop = 12
  const paddingBottom = 18
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinatesA = getSvgCoordinates(seriesA.values, min, max, chartOptions)
  const coordinatesB = getSvgCoordinates(seriesB.values, min, max, chartOptions)
  const pointsA = getSvgPoints(seriesA.values, min, max, chartOptions)
  const pointsB = getSvgPoints(seriesB.values, min, max, chartOptions)
  const lastA = coordinatesA.at(-1)
  const lastB = coordinatesB.at(-1)
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const plotBottom = height - paddingBottom

  return (
    <div className="mini-trend compare-trend" onMouseLeave={() => setTooltip(null)}>
      <svg
        aria-label={`${seriesA.label} y ${seriesB.label} comparados mes a mes en el mismo eje. Último mes: ${seriesA.label} ${formatMoney(seriesA.values.at(-1))}, ${seriesB.label} ${formatMoney(seriesB.values.at(-1))}.`}
        className="mini-trend-svg"
        height={height}
        preserveAspectRatio="none"
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
        <polyline fill="none" points={pointsA} stroke={seriesA.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        <polyline
          fill="none"
          points={pointsB}
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
          {formatMoneyCompact(max)}
        </text>
        <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 5} y={plotBottom}>
          {formatMoneyCompact(min)}
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
          <strong style={{ color: seriesA.color }}>
            {seriesA.label} {formatMoney(tooltip.a)}
          </strong>
          <strong style={{ color: seriesB.color }}>
            {seriesB.label} {formatMoney(tooltip.b)}
          </strong>
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
  const [savingsPeriod, setSavingsPeriod] = useState('12')
  const [savingsTooltip, setSavingsTooltip] = useState(null)
  const [investmentTooltip, setInvestmentTooltip] = useState(null)
  const budget = selectedRow ? getBudget(selectedRow) : null
  const expenseAnalysis = getExpenseAnalysis(result)
  const incomeAnalysis = getIncomeAnalysis(result)
  const expenseRows = getExpenseRowsForPeriod(expenseAnalysis.mensual, selectedRow?.Mes, expensePeriod)
  const expenseCategoryRows = getExpenseCategoryRowsForPeriod(expenseAnalysis.categorias, selectedRow?.Mes, expensePeriod)
  const expenseMax = Math.max(...expenseCategoryRows.map((row) => asNumber(row.total)), 1)
  const expenseMonthlyMax = Math.max(...expenseRows.map((row) => Math.abs(asNumber(row.gastos))), 1)
  const savingsAnalysis = getSavingsAnalysis(result)
  const scopedSavingsRows = savingsAnalysis.mensual.filter((row) => !selectedRow?.Mes || String(row.Mes) <= String(selectedRow.Mes))
  const selectedSavingsRow = scopedSavingsRows.at(-1) ?? savingsAnalysis.ultimo_mes
  const savingsPercentRows = getPeriodRows(scopedSavingsRows, savingsPeriod)
  const savingsMax = Math.max(...savingsPercentRows.map((row) => Math.abs(asNumber(row.balance))), 1)
  const savingsPercentValues = savingsPercentRows.map((row) => asNumber(row.porcentaje_ahorro))
  const savingsAverageValues = movingAverage(savingsPercentValues)
  const savingsPercentBounds = {
    max: Math.max(...savingsPercentValues, ...savingsAverageValues, 100),
    min: Math.min(...savingsPercentValues, ...savingsAverageValues, 0),
  }
  const savingsChart = {
    width: 640,
    height: 260,
    paddingLeft: 54,
    paddingRight: 22,
    paddingTop: 24,
    paddingBottom: 42,
  }
  const savingsCoordinates = getSvgCoordinates(
    savingsPercentValues,
    savingsPercentBounds.min,
    savingsPercentBounds.max,
    savingsChart,
  )
  const savingsAverageCoordinates = getSvgCoordinates(
    savingsAverageValues,
    savingsPercentBounds.min,
    savingsPercentBounds.max,
    savingsChart,
  )
  const savingsYAxisTicks = getAxisTicks(savingsPercentBounds.min, savingsPercentBounds.max, 5)
  const savingsXLabelStep = Math.max(1, Math.ceil(savingsPercentRows.length / 6))
  const investmentRows = rows.slice(-12)
  const investmentChart = {
    width: 640,
    height: 260,
    paddingLeft: 54,
    paddingRight: 22,
    paddingTop: 24,
    paddingBottom: 42,
  }
  const investmentReserveValues = investmentRows.map((row) => asNumber(row['📈 Inversiones']))
  const investmentInvestedValues = investmentRows.map((row) => asNumber(row['Dinero Invertido']))
  const investmentBounds = {
    max: Math.max(...investmentReserveValues, ...investmentInvestedValues, 1),
    min: Math.min(...investmentReserveValues, ...investmentInvestedValues, 0),
  }
  const investmentReserveCoordinates = getSvgCoordinates(
    investmentReserveValues,
    investmentBounds.min,
    investmentBounds.max,
    investmentChart,
  )
  const investmentInvestedCoordinates = getSvgCoordinates(
    investmentInvestedValues,
    investmentBounds.min,
    investmentBounds.max,
    investmentChart,
  )
  const investmentYAxisTicks = getAxisTicks(investmentBounds.min, investmentBounds.max, 5)
  const investmentXLabelStep = Math.max(1, Math.ceil(investmentRows.length / 6))
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
                  <div className="budget-bar" aria-label="Distribución del presupuesto mensual">
                    {budget.debtReserve > 0 && (
                      <span
                        className="budget-segment reserved"
                        style={{ width: getSegmentWidth(budget.debtReserve, budget.totalForBar) }}
                        title={`Reserva deuda: ${formatMoney(budget.debtReserve)}`}
                      />
                    )}
                    {budget.budgetConsumption > 0 && (
                      <span
                        className="budget-segment spent"
                        style={{ width: getSegmentWidth(budget.budgetConsumption, budget.totalForBar) }}
                        title={`Gasto: ${formatMoney(budget.spent)}`}
                      />
                    )}
                    {budget.remaining > 0 && (
                      <span
                        className="budget-segment remaining"
                        style={{ width: getSegmentWidth(budget.remaining, budget.totalForBar) }}
                        title={`Restante: ${formatMoney(budget.remaining)}`}
                      />
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
                  <article>
                    <span>Presupuesto mes</span>
                    <strong>{formatMoney(budget.monthBudget)}</strong>
                  </article>
                  <article>
                    <span>Disponible visual</span>
                    <strong>{formatMoney(budget.remaining)}</strong>
                  </article>
                  <article>
                    <span>Reserva deuda</span>
                    <strong>{formatMoney(budget.debtReserve)}</strong>
                  </article>
                  <article>
                    <span>Gasto neto</span>
                    <strong>{formatMoney(budget.spent)}</strong>
                  </article>
                  <article>
                    <span>Ejecutado</span>
                    <strong>{budget.execution.toFixed(1)}%</strong>
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
              expenseRows.length > 0 ? (
                <section className="expenses-layout">
                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Gasto mes a mes</h3>
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
                    <div className="expense-month-bars">
                      {expenseRows.map((row) => (
                        <article key={row.Mes}>
                          <span>{formatMonthLabel(row.Mes)}</span>
                          <div className="expense-month-bar" title={`Gasto ${row.Mes}: ${formatMoney(row.gastos)}`}>
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row.gastos)) / expenseMonthlyMax) * 100)}%` }} />
                          </div>
                          <strong>{formatMoney(row.gastos)}</strong>
                        </article>
                      ))}
                    </div>
                  </div>

                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Categorias del periodo</h3>
                      <span>{expenseRows[0]?.Mes} - {expenseRows.at(-1)?.Mes}</span>
                    </div>
                    <div className="category-list">
                      {expenseCategoryRows.map((row) => (
                        <article className="category-row" key={row.categoria}>
                          <div>
                            <strong>{row.categoria}</strong>
                            <span>{formatMoney(row.total)}</span>
                          </div>
                          <div className="category-bar">
                            <span style={{ width: `${Math.max(4, (asNumber(row.total) / expenseMax) * 100)}%` }} />
                          </div>
                        </article>
                      ))}
                    </div>
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

                  <section className="trend-panel compact-trend savings-percent-panel">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Porcentaje de ahorro</h3>
                        <span>Comparado con la media móvil de 3 meses</span>
                      </div>
                      <label className="period-selector">
                        <span>Periodo</span>
                        <select value={savingsPeriod} onChange={(event) => setSavingsPeriod(event.target.value)}>
                          <option value="6">6 meses</option>
                          <option value="12">12 meses</option>
                          <option value="all">Todo</option>
                        </select>
                      </label>
                    </div>
                    <div className="savings-chart-wrap">
                      <svg
                        className="savings-percent-chart"
                        onMouseLeave={() => setSavingsTooltip(null)}
                        viewBox="0 0 640 260"
                        role="img"
                        aria-label="Porcentaje de ahorro mensual"
                      >
                        <defs>
                          <linearGradient id="savingsArea" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stopColor="#157a5c" stopOpacity="0.22" />
                            <stop offset="100%" stopColor="#157a5c" stopOpacity="0.01" />
                          </linearGradient>
                        </defs>
                        {savingsYAxisTicks.map((tick) => {
                          const y =
                            savingsChart.height -
                            savingsChart.paddingBottom -
                            ((tick - savingsPercentBounds.min) / (savingsPercentBounds.max - savingsPercentBounds.min || 1)) *
                              (savingsChart.height - savingsChart.paddingTop - savingsChart.paddingBottom)
                          return (
                            <g key={tick.toFixed(2)}>
                              <line className="chart-grid-line" x1="54" x2="618" y1={y} y2={y} />
                              <text className="chart-axis-label" x="44" y={y + 4} textAnchor="end">
                                {formatPercent(tick)}
                              </text>
                            </g>
                          )
                        })}
                        <line className="chart-axis" x1="54" x2="618" y1="218" y2="218" />
                        {savingsTooltip ? (
                          <line className="chart-crosshair" x1={savingsTooltip.x} x2={savingsTooltip.x} y1="24" y2="218" />
                        ) : null}
                        {savingsPercentRows.map((row, index) =>
                          index % savingsXLabelStep === 0 || index === savingsPercentRows.length - 1 ? (
                            <g key={row.Mes}>
                              <line className="chart-axis-tick" x1={savingsCoordinates[index]?.x} x2={savingsCoordinates[index]?.x} y1="218" y2="224" />
                              <text className="chart-axis-label" x={savingsCoordinates[index]?.x} y="244" textAnchor="middle">
                                {formatMonthLabel(row.Mes)}
                              </text>
                            </g>
                          ) : null,
                        )}
                        <path
                          className="chart-area-fill savings-area"
                          d={getSvgAreaPath(savingsPercentValues, savingsPercentBounds.min, savingsPercentBounds.max, savingsChart)}
                        />
                        <polyline
                          fill="none"
                          points={getSvgPoints(savingsPercentValues, savingsPercentBounds.min, savingsPercentBounds.max, savingsChart)}
                          stroke="#157a5c"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="3.4"
                        />
                        <polyline
                          className="chart-moving-line"
                          fill="none"
                          points={getSvgPoints(savingsAverageValues, savingsPercentBounds.min, savingsPercentBounds.max, savingsChart)}
                          stroke="#b85a4b"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="3"
                        />
                        {savingsAverageCoordinates.map(({ x, y }, index) => (
                          <circle className="chart-average-point" cx={x} cy={y} fill="#b85a4b" key={`avg-${savingsPercentRows[index]?.Mes}`} r="3" />
                        ))}
                        {savingsCoordinates.map(({ x, y }, index) => (
                          <g key={savingsPercentRows[index]?.Mes}>
                            {index === savingsCoordinates.length - 1 || savingsTooltip?.month === savingsPercentRows[index]?.Mes ? (
                              <text className="chart-value-label" x={x} y={y < 42 ? y + 18 : y - 10} textAnchor="middle">
                                {formatPercent(savingsPercentValues[index])}
                              </text>
                            ) : null}
                            <circle
                              className="chart-point"
                              cx={x}
                              cy={y}
                              fill="#286b57"
                              onFocus={() =>
                                setSavingsTooltip({
                                  average: savingsAverageValues[index],
                                  month: savingsPercentRows[index]?.Mes,
                                  balance: savingsPercentRows[index]?.balance,
                                  expenses: savingsPercentRows[index]?.gastos,
                                  income: savingsPercentRows[index]?.ingresos,
                                  value: savingsPercentValues[index],
                                  x,
                                  y,
                                })
                              }
                              onMouseEnter={() =>
                                setSavingsTooltip({
                                  average: savingsAverageValues[index],
                                  month: savingsPercentRows[index]?.Mes,
                                  balance: savingsPercentRows[index]?.balance,
                                  expenses: savingsPercentRows[index]?.gastos,
                                  income: savingsPercentRows[index]?.ingresos,
                                  value: savingsPercentValues[index],
                                  x,
                                  y,
                                })
                              }
                              r={savingsTooltip?.month === savingsPercentRows[index]?.Mes ? '5.5' : '3.8'}
                              tabIndex="0"
                            />
                          </g>
                        ))}
                      </svg>
                      {savingsTooltip && (
                        <div
                          className="chart-tooltip savings-tooltip"
                          style={{
                            '--tooltip-color': '#286b57',
                            left: `${(savingsTooltip.x / savingsChart.width) * 100}%`,
                            top: `${(savingsTooltip.y / savingsChart.height) * 100}%`,
                          }}
                        >
                          <strong>{savingsTooltip.month}</strong>
                          <span>Ahorro: {formatPercent(savingsTooltip.value)}</span>
                          <span>Ingresos: {formatMoney(savingsTooltip.income)}</span>
                          <span>Gastos: {formatMoney(savingsTooltip.expenses)}</span>
                          <span>Balance: {formatMoney(savingsTooltip.balance)}</span>
                          <span>Media móvil: {formatPercent(savingsTooltip.average)}</span>
                        </div>
                      )}
                    </div>
                    <div className="savings-percent-legend">
                      <span>Ahorro mensual</span>
                      <span>Media móvil</span>
                    </div>
                  </section>

                  <section className="trend-panel compact-trend">
                    <h3 className="table-title">Balance mensual</h3>
                    <div className="trend-list">
                      {savingsPercentRows.map((row) => (
                        <article className="savings-row" key={row.Mes}>
                          <span className="trend-month">{row.Mes}</span>
                          <div className={asNumber(row.balance) >= 0 ? 'savings-bar positive' : 'savings-bar negative'}>
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row.balance)) / savingsMax) * 100)}%` }} />
                          </div>
                          <strong>{asNumber(row.porcentaje_ahorro).toFixed(1)}%</strong>
                        </article>
                      ))}
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
                    <span>Patrimonio inversión</span>
                    <strong>{formatMoney(asNumber(selectedRow['📈 Inversiones']) + asNumber(selectedRow['Dinero Invertido']))}</strong>
                  </article>
                </div>

                <section className="trend-panel compact-trend investor-chart-panel">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Evolución inversiones</h3>
                      <span>Bolsa disponible frente a dinero invertido</span>
                    </div>
                  </div>
                  <div className="savings-chart-wrap">
                    <svg
                      className="savings-percent-chart investment-line-chart"
                      onMouseLeave={() => setInvestmentTooltip(null)}
                      viewBox="0 0 640 260"
                      role="img"
                      aria-label="Evolución de inversiones"
                    >
                      <defs>
                        <linearGradient id="investmentArea" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#436a92" stopOpacity="0.2" />
                          <stop offset="100%" stopColor="#436a92" stopOpacity="0.01" />
                        </linearGradient>
                      </defs>
                      {investmentYAxisTicks.map((tick) => {
                        const y =
                          investmentChart.height -
                          investmentChart.paddingBottom -
                          ((tick - investmentBounds.min) / (investmentBounds.max - investmentBounds.min || 1)) *
                            (investmentChart.height - investmentChart.paddingTop - investmentChart.paddingBottom)
                        return (
                          <g key={tick.toFixed(2)}>
                            <line className="chart-grid-line" x1="54" x2="618" y1={y} y2={y} />
                            <text className="chart-axis-label" x="44" y={y + 4} textAnchor="end">
                              {formatMoney(tick).replace(',00', '')}
                            </text>
                          </g>
                        )
                      })}
                      <line className="chart-axis" x1="54" x2="618" y1="218" y2="218" />
                      {investmentTooltip ? (
                        <line className="chart-crosshair" x1={investmentTooltip.x} x2={investmentTooltip.x} y1="24" y2="218" />
                      ) : null}
                      {investmentRows.map((row, index) =>
                        index % investmentXLabelStep === 0 || index === investmentRows.length - 1 ? (
                          <g key={row.Mes}>
                            <line className="chart-axis-tick" x1={investmentInvestedCoordinates[index]?.x} x2={investmentInvestedCoordinates[index]?.x} y1="218" y2="224" />
                            <text className="chart-axis-label" x={investmentInvestedCoordinates[index]?.x} y="244" textAnchor="middle">
                              {formatMonthLabel(row.Mes)}
                            </text>
                          </g>
                        ) : null,
                      )}
                      <path
                        className="chart-area-fill investment-area"
                        d={getSvgAreaPath(investmentInvestedValues, investmentBounds.min, investmentBounds.max, investmentChart)}
                      />
                      <polyline
                        className="investment-reserve-line"
                        fill="none"
                        points={getSvgPoints(investmentReserveValues, investmentBounds.min, investmentBounds.max, investmentChart)}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <polyline
                        className="investment-main-line"
                        fill="none"
                        points={getSvgPoints(investmentInvestedValues, investmentBounds.min, investmentBounds.max, investmentChart)}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      {investmentInvestedCoordinates.map(({ x, y }, index) => (
                        <circle
                          className="chart-hit-point"
                          cx={x}
                          cy={y}
                          key={investmentRows[index]?.Mes}
                          onFocus={() =>
                            setInvestmentTooltip({
                              month: investmentRows[index]?.Mes,
                              reserve: investmentReserveValues[index],
                              invested: investmentInvestedValues[index],
                              x,
                              y,
                            })
                          }
                          onMouseEnter={() =>
                            setInvestmentTooltip({
                              month: investmentRows[index]?.Mes,
                              reserve: investmentReserveValues[index],
                              invested: investmentInvestedValues[index],
                              x,
                              y,
                            })
                          }
                          r="11"
                          tabIndex="0"
                        />
                      ))}
                    </svg>
                    {investmentTooltip && (
                      <div
                        className="chart-tooltip investment-finance-tooltip"
                        style={{
                          '--tooltip-color': '#436a92',
                          left: `${(investmentTooltip.x / investmentChart.width) * 100}%`,
                          top: `${(investmentTooltip.y / investmentChart.height) * 100}%`,
                        }}
                      >
                        <strong>{investmentTooltip.month}</strong>
                        <span>Invertido: {formatMoney(investmentTooltip.invested)}</span>
                        <span>Bolsa: {formatMoney(investmentTooltip.reserve)}</span>
                      </div>
                    )}
                  </div>
                  <div className="investment-legend investor-legend">
                    <span>Bolsa</span>
                    <span>Invertido</span>
                  </div>
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
