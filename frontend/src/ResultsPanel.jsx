import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { CalendarDays, Download, FileJson, Maximize2, Table2 } from 'lucide-react'
import { ChartCrosshair, ChartReadout, PeriodTabs } from './ChartParts'
import {
  CHART_ASPECT,
  getChartHeight,
  getNiceTicks,
  getSvgCoordinates,
  getValueDomain,
  linePath,
  pickLabelIndices,
} from './chartGeometry'
import { useChartCrosshair } from './useChartCrosshair'
import { formatDelta, formatMoney } from './formatters'
import { aggregateDividendsByCompany, comparisonRows, getAdjustedSavings, getBudget, getDividendPayments } from './resultSelectors'

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
  { field: '💸 Presupuesto Mes', label: 'Presupuesto mensual', color: '#1d3f9a' },
  { field: '💳 Gasto del mes', label: 'Gasto del mes', color: '#b3261e' },
]

const typologyCards = [
  { field: '💰 Ahorros', label: 'Ahorros', color: '#0b6b63' },
  { field: '💼 Vacaciones', label: 'Vacaciones', color: '#b45309' },
  { field: '📈 Inversiones', label: 'Reserva inversión', color: '#0b6b63' },
  { field: 'Fondo de reserva cargado', label: 'Fondo reserva', color: '#0b6b63' },
  { field: '📉 Deuda Presupuestaria acumulada', label: 'Deuda acumulada', color: '#b45309' },
  { field: '🎁 Regalos', label: 'Regalos', color: '#b45309' },
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
  { field: 'alimentacion', label: 'Alimentación', color: '#7a9cc6' },
  { field: 'transporte', label: 'Transporte', color: '#1d3f9a' },
  { field: 'hosteleria', label: 'Hostelería', color: '#0b6b63' },
  { field: 'entretenimiento', label: 'Entretenimiento', color: '#b45309' },
  { field: 'otros', label: 'Otros', color: '#8a93a3' },
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
  { label: 'Sobrante del mes', field: '🧾 Presupuesto Disponible', type: 'money' },
  { label: 'Inversiones', field: '📈 Inversiones', type: 'money' },
  { label: 'Invertido', field: 'Dinero Invertido', type: 'money' },
  { label: 'Exceso mes', field: '📉 Deuda Presupuestaria mensual', type: 'money' },
  { label: 'Exceso acum.', field: '📉 Deuda Presupuestaria acumulada', type: 'money' },
]

function asNumber(value) {
  return Number(value) || 0
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

/**
 * Situación del fondo de reserva tal y como la calcula el backend: objetivo (seis meses de gasto
 * medio), lo ya cargado y el porcentaje cubierto. Sin objetivo no hay progreso que enseñar, así
 * que el valor por defecto deja la barra fuera en vez de inventarse una meta.
 */
function getReserveFund(result) {
  const fondo = result?.analisis?.fondo_reserva
  return {
    cargado: asNumber(fondo?.cargado),
    objetivo: asNumber(fondo?.objetivo),
    pendiente: asNumber(fondo?.pendiente),
    porcentaje: asNumber(fondo?.porcentaje),
    mesesDeCobertura: asNumber(fondo?.meses_de_cobertura) || 6,
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

function getInvestmentContributions(result) {
  return result?.analisis?.aporte_inversion ?? []
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

function formatMonthLabel(value) {
  const [year, month] = String(value ?? '').split('-')
  if (!year || !month) return value
  return `${month}/${year.slice(-2)}`
}

function formatMoneyCompact(value) {
  return new Intl.NumberFormat('es-ES', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
    useGrouping: 'always',
  })
    .format(value)
    .replace('-', '\u2212')
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

/** Porcentaje del objetivo cubierto por el fondo del mes indicado, topado al 100 %. */
function reserveCoverage(row, reserveFund) {
  if (!(reserveFund?.objetivo > 0)) return 0
  return Math.min(100, (asNumber(row['Fondo de reserva cargado']) / reserveFund.objetivo) * 100)
}

function getSummaryGroups(row, reserveFund) {
  return {
    patrimony: [
      { label: 'Total', value: row.total },
      {
        label: 'Ahorros',
        value: getAdjustedSavings(row),
        // El sobre de ahorros no se descuenta cuando se invierte por encima de la bolsa, así que
        // la cifra bruta sale inflada. Con el ajuste, las tarjetas suman exactamente el total.
        note: `Ahorro ${formatMoney(row['💰 Ahorros'])} ${asNumber(row['📈 Inversiones']) < 0 ? '−' : '+'} bolsa de inversión ${formatMoney(Math.abs(asNumber(row['📈 Inversiones'])))}`,
      },
      { label: 'Dinero invertido', value: row['Dinero Invertido'] },
      {
        label: 'Fondo emergencia',
        value: row['Fondo de reserva cargado'],
        // El objetivo son seis meses de gasto medio; sin histórico suficiente no hay meta que medir.
        // El porcentaje se calcula con el mes que estás mirando, no con el último del histórico.
        ...(reserveFund?.objetivo > 0
          ? {
              note: `${formatPercent(reserveCoverage(row, reserveFund))} de ${formatMoney(reserveFund.objetivo)}, ${reserveFund.mesesDeCobertura} meses de gasto`,
              progress: reserveCoverage(row, reserveFund),
            }
          : {}),
      },
      { label: 'Vacaciones', value: row['💼 Vacaciones'] },
      { label: 'Regalos', value: row['🎁 Regalos'] },
    ],
    budget: [
      { label: 'Presupuesto mes', value: row['💸 Presupuesto Mes'] },
      { label: 'Sobrante del mes', value: row['🧾 Presupuesto Disponible'] },
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

function formatPoints(value) {
  if (!Number.isFinite(value)) return '0,0 pp'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toLocaleString('es-ES', { maximumFractionDigits: 1, minimumFractionDigits: 1 })} pp`.replace(
    '-',
    '\u2212',
  )
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return '0,0 %'
  return `${value.toLocaleString('es-ES', { maximumFractionDigits: 1, minimumFractionDigits: 1 })} %`.replace('-', '\u2212')
}

// En una tarjeta de 180 px, "2.750 €" se come la mitad del ancho y la curva se queda sin sitio.
function formatMoneyTiny(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return ''
  const text =
    Math.abs(number) >= 1000
      ? `${new Intl.NumberFormat('es-ES', { maximumFractionDigits: 1 }).format(number / 1000)}k`
      : new Intl.NumberFormat('es-ES', { maximumFractionDigits: 0 }).format(number)
  return text.replace('-', '\u2212')
}

function axisPaddingLeft(min, max, formatCompact = formatMoneyCompact, minPadding = 42) {
  // Se mide por caracteres porque el eje usa tipografía monoespaciada; a 12px cada uno ocupa ~7,5px.
  const chars = Math.max(formatCompact(min).length, formatCompact(max).length)
  return Math.min(86, Math.max(minPadding, chars * 7.5 + 10))
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

// El tooltip se ancla al punto, pero en los extremos se saldría del panel (y llegaba a tapar la
// fila de periodos): se limita para que su centro no pase de los bordes útiles.
function tooltipLeft(x, width) {
  if (!width) return '50%'
  return `${Math.min(92, Math.max(8, (x / width) * 100))}%`
}

function TypologyMiniChart({
  aspect = CHART_ASPECT,
  color,
  formatCompact = formatMoneyCompact,
  formatValue = formatMoney,
  label,
  maxHeight = 520,
  minHeight = 200,
  months,
  onActiveIndexChange,
  values,
}) {
  const [containerRef, width] = useElementWidth(420)
  const height = getChartHeight(width, aspect, minHeight, maxHeight)
  const { min, max } = getValueDomain(values)
  const isNarrow = width < 360
  const axisFormat = isNarrow ? formatMoneyTiny : formatCompact
  const paddingLeft = axisPaddingLeft(min, max, axisFormat, isNarrow ? 24 : 42)
  const paddingRight = isNarrow ? 8 : 10
  const paddingTop = 12
  const paddingBottom = 26
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinates = getSvgCoordinates(values, min, max, chartOptions)
  const last = coordinates.at(-1)
  const showsZero = min <= 0 && max >= 0
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const baselineY = coordinates[0]?.y ?? null
  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const monthLabelIndices = pickLabelIndices(months.length, width > 520 ? 5 : width > 320 ? 3 : 2)
  const plotBottom = height - paddingBottom
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: plotBottom }
  const trend = values.at(-1) >= values[0] ? 'tendencia ascendente' : 'tendencia descendente'

  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(
    coordinates.map((point) => point.x),
    width,
  )
  // El valor grande de la tarjeta lo pinta el contenedor, así que se le avisa del punto señalado.
  useEffect(() => {
    onActiveIndexChange?.(isHovering ? activeIndex : null)
  }, [activeIndex, isHovering, onActiveIndexChange])

  return (
    <div className="mini-trend" ref={containerRef}>
      <svg
        aria-label={`${label}: de ${formatValue(values[0])} a ${formatValue(values.at(-1))} en ${values.length} meses, ${trend}.`}
        className="mini-trend-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = getSvgCoordinates([tick], min, max, chartOptions)[0].y
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {axisFormat(tick)}
              </text>
            </g>
          )
        })}
        {showsZero ? (
          <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        ) : null}
        <path d={linePath(coordinates)} fill="none" stroke={color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.6" />
        {monthLabelIndices.map((index) => (
          <text
            className="mini-axis-month"
            key={months[index] ?? index}
            textAnchor={index === 0 ? 'start' : index === months.length - 1 ? 'end' : 'middle'}
            x={coordinates[index]?.x}
            y={height - 7}
          >
            {formatMonthLabel(months[index])}
          </text>
        ))}
        <rect className="chart-frame" height={plotBottom - paddingTop} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={baselineY}
          handlers={handlers}
          isHovering={isHovering}
          label={`Recorrer ${label}`}
          plot={plot}
          pointColor={color}
          points={coordinates}
        />
      </svg>
      {isHovering && coordinates[activeIndex] ? (
        <div className="mini-trend-tooltip" style={{ left: tooltipLeft(coordinates[activeIndex].x, width) }}>
          <span>{formatMonthLabel(months[activeIndex])}</span>
          <strong>{formatValue(values[activeIndex])}</strong>
        </div>
      ) : null}
    </div>
  )
}

function TypologyMiniCard({ color, label, months, onFeature, values }) {
  // Cada mini tarjeta lleva su cifra: sin estado propio, señalar una cambiaría el valor de todas.
  const [activeIndex, setActiveIndex] = useState(null)
  const shownIndex = activeIndex ?? values.length - 1

  return (
    <article className="chart-card typology-mini">
      <div className="typology-mini-head">
        <div className="typology-mini-head-info">
          <span>{label}</span>
          <strong>{formatMoney(values[shownIndex])}</strong>
        </div>
        <TypologyFeatureButton label={label} onFeature={onFeature} />
      </div>
      <TypologyMiniChart
        color={color}
        height={72}
        label={label}
        months={months}
        onActiveIndexChange={setActiveIndex}
        values={values}
      />
    </article>
  )
}

// Composición de lo que entra en la bolsa de inversión cada mes. La barra de aportación va
// apilada por origen (sueldo, intereses, dividendos y el sobrante del presupuesto) y al lado,
// sin apilar, lo que ese mes salió de la bolsa para invertirse de verdad.
const contributionParts = [
  { key: 'sueldo', label: 'Sueldo', color: '#1d3f9a' },
  { key: 'intereses', label: 'Intereses', color: '#7a9cc6' },
  { key: 'dividendos', label: 'Dividendos', color: '#b45309' },
  { key: 'extra', label: 'Extra por ahorrar', color: '#8a93a3' },
]

const investedColor = '#0b6b63'

function InvestmentContributionChart({ aspect = CHART_ASPECT, maxLabels = 8, rows }) {
  const [containerRef, width] = useElementWidth(640)
  const height = getChartHeight(width, aspect)
  const paddingRight = 12
  const paddingTop = 16
  const paddingBottom = 32

  const contributions = rows.map((row) => contributionParts.reduce((sum, part) => sum + asNumber(row[part.key]), 0))
  const invested = rows.map((row) => asNumber(row.invertido))
  // Las barras siempre arrancan en 0: su longitud representa el importe y recortarla engañaría.
  const { min, max } = getValueDomain([...contributions, ...invested], { includeZero: true })
  const paddingLeft = axisPaddingLeft(min, max)
  const plotWidth = Math.max(0, width - paddingLeft - paddingRight)
  const plotHeight = height - paddingTop - paddingBottom
  const slot = rows.length > 0 ? plotWidth / rows.length : 0
  const groupWidth = Math.max(6, slot - 10)
  const barWidth = Math.max(3, groupWidth / 2 - 2)
  const scaleY = (value) => height - paddingBottom - ((value - min) / (max - min || 1)) * plotHeight
  const zeroY = scaleY(0)

  const groups = rows.map((row, index) => {
    const slotX = paddingLeft + index * slot
    const groupX = slotX + (slot - groupWidth) / 2
    let stackTop = 0
    const segments = contributionParts
      .map((part) => {
        const value = asNumber(row[part.key])
        if (value <= 0) return null
        const y = scaleY(stackTop + value)
        const segment = { ...part, height: Math.abs(scaleY(stackTop) - y), value, y }
        stackTop += value
        return segment
      })
      .filter(Boolean)
    return {
      contribution: contributions[index],
      contributionX: groupX,
      invested: invested[index],
      investedHeight: Math.abs(scaleY(invested[index]) - zeroY),
      investedX: groupX + barWidth + 4,
      investedY: scaleY(invested[index]),
      key: row.Mes,
      label: formatMonthLabel(row.Mes),
      segments,
      x: groupX + groupWidth / 2,
    }
  })

  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const labelStep = Math.max(1, Math.ceil(rows.length / maxLabels))
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: paddingTop + plotHeight }
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(groups.map((group) => group.x), width)
  const active = isHovering ? groups[activeIndex] : null

  return (
    <div className="expense-chart" ref={containerRef}>
      <svg
        aria-label="Aportación mensual a la bolsa de inversión por origen, frente a lo invertido"
        className="mini-trend-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = scaleY(tick)
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {formatMoneyCompact(tick)}
              </text>
            </g>
          )
        })}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        {groups.map((group) => (
          <g key={group.key}>
            {group.segments.map((segment) => (
              <rect fill={segment.color} height={segment.height} key={segment.key} rx="1.5" width={barWidth} x={group.contributionX} y={segment.y} />
            ))}
            <rect fill={investedColor} height={group.investedHeight} rx="1.5" width={barWidth} x={group.investedX} y={group.investedY} />
          </g>
        ))}
        {groups.map((group, index) =>
          index % labelStep === 0 ? (
            <text className="mini-axis-month" key={group.key} textAnchor="middle" x={group.x} y={height - 8}>
              {group.label}
            </text>
          ) : null,
        )}
        <rect className="chart-frame" height={plotHeight} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={null}
          handlers={handlers}
          isHovering={isHovering}
          label="Recorrer la aportación a la bolsa de inversión"
          plot={plot}
          pointColor={investedColor}
          points={groups.map((group) => ({ x: group.x, y: zeroY }))}
        />
      </svg>
      {active && (
        <div className="mini-trend-tooltip expense-chart-tooltip" style={{ left: tooltipLeft(active.x, width) }}>
          <span>{active.label}</span>
          {contributionParts.map((part) => (
            <span className="mini-tooltip-row" key={part.key}>
              <i style={{ background: part.color }} />
              {part.label} {formatMoney(asNumber(rows[activeIndex]?.[part.key]))}
            </span>
          ))}
          <strong>Aportado {formatMoney(active.contribution)}</strong>
          <span className="mini-tooltip-row">
            <i style={{ background: investedColor }} />
            Invertido {formatMoney(active.invested)}
          </span>
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
  aspect = CHART_ASPECT,
  formatCompact = formatMoneyCompact,
  formatValue = formatMoney,
  maxHeight = 520,
  minHeight = 200,
  months,
  onActiveIndexChange,
  seriesA,
  seriesB,
}) {
  const [containerRef, width] = useElementWidth(560)
  const height = getChartHeight(width, aspect, minHeight, maxHeight)
  const combined = [...seriesA.values, ...seriesB.values]
  const { min, max } = getValueDomain(combined)
  const paddingLeft = axisPaddingLeft(min, max, formatCompact)
  const paddingRight = 10
  const paddingTop = 12
  const paddingBottom = 26
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const coordinatesA = getSvgCoordinates(seriesA.values, min, max, chartOptions)
  const coordinatesB = getSvgCoordinates(seriesB.values, min, max, chartOptions)
  const pathA = linePath(coordinatesA)
  const pathB = linePath(coordinatesB)
  const lastA = coordinatesA.at(-1)
  const lastB = coordinatesB.at(-1)
  const showsZero = min <= 0 && max >= 0
  const zeroY = getSvgCoordinates([0], min, max, chartOptions)[0].y
  const plotBottom = height - paddingBottom
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: plotBottom }
  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const monthLabelIndices = pickLabelIndices(months.length, width > 520 ? 5 : 4)
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(
    coordinatesA.map((point) => point.x),
    width,
  )
  // La cifra de cabecera la pinta la tarjeta, igual que en las gráficas de tipología.
  useEffect(() => {
    onActiveIndexChange?.(isHovering ? activeIndex : null)
  }, [activeIndex, isHovering, onActiveIndexChange])

  return (
    <div className="mini-trend compare-trend" ref={containerRef}>
      <svg
        aria-label={`${seriesA.label} y ${seriesB.label} comparados mes a mes en el mismo eje. Último mes: ${seriesA.label} ${formatValue(seriesA.values.at(-1))}, ${seriesB.label} ${formatValue(seriesB.values.at(-1))}.`}
        className="mini-trend-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = getSvgCoordinates([tick], min, max, chartOptions)[0].y
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {formatCompact(tick)}
              </text>
            </g>
          )
        })}
        {showsZero ? (
          <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        ) : null}
        <path d={pathA} fill="none" stroke={seriesA.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.6" />
        <path
          d={pathB}
          fill="none"
          stroke={seriesB.color}
          strokeDasharray="6 4"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2.6"
        />
        <circle cx={lastA.x} cy={lastA.y} fill={seriesA.color} pointerEvents="none" r="3.6" />
        <circle cx={lastB.x} cy={lastB.y} fill={seriesB.color} pointerEvents="none" r="3.6" />
        {isHovering && coordinatesB[activeIndex] ? (
          <circle className="chart-active-point" cx={coordinatesB[activeIndex].x} cy={coordinatesB[activeIndex].y} fill={seriesB.color} r="5" />
        ) : null}
        {monthLabelIndices.map((index) => (
          <text
            className="mini-axis-month"
            key={months[index] ?? index}
            textAnchor={index === 0 ? 'start' : index === months.length - 1 ? 'end' : 'middle'}
            x={coordinatesA[index]?.x}
            y={height - 7}
          >
            {formatMonthLabel(months[index])}
          </text>
        ))}
        <rect className="chart-frame" height={plotBottom - paddingTop} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={coordinatesA[0]?.y ?? null}
          handlers={handlers}
          isHovering={isHovering}
          label={`Recorrer ${seriesA.label} y ${seriesB.label}`}
          plot={plot}
          pointColor={seriesA.color}
          points={coordinatesA}
        />
      </svg>
      {isHovering && coordinatesA[activeIndex] && (
        <div className="mini-trend-tooltip compare-tooltip" style={{ left: tooltipLeft(coordinatesA[activeIndex].x, width) }}>
          <span>{formatMonthLabel(months[activeIndex])}</span>
          <span className="mini-tooltip-row">
            <i style={{ background: seriesA.color }} />
            {seriesA.label} {formatValue(seriesA.values[activeIndex])}
          </span>
          <span className="mini-tooltip-row">
            <i style={{ background: seriesB.color }} />
            {seriesB.label} {formatValue(seriesB.values[activeIndex])}
          </span>
        </div>
      )}
    </div>
  )
}

/**
 * Reservas en una sola gráfica: una línea por reserva y la cifra final etiquetada junto a su
 * línea. Sustituye a las doce filas de tres barras, que obligaban a comparar de memoria.
 */
function ReserveLinesChart({ aspect = 2.6, maxHeight = 420, minHeight = 240, months, series }) {
  const [containerRef, width] = useElementWidth(720)
  const height = getChartHeight(width, aspect, minHeight, maxHeight)
  const combined = series.flatMap((item) => item.values)
  const { min, max } = getValueDomain(combined)
  const paddingLeft = axisPaddingLeft(min, max)
  const paddingRight = 104
  const paddingTop = 14
  const paddingBottom = 28
  const chartOptions = { height, paddingBottom, paddingLeft, paddingRight, paddingTop, width }
  const drawn = series.map((item) => ({ ...item, points: getSvgCoordinates(item.values, min, max, chartOptions) }))
  const plotBottom = height - paddingBottom
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: plotBottom }
  const yTicks = getNiceTicks(min, max, height > 300 ? 5 : 4)
  const monthLabelIndices = pickLabelIndices(months.length, width > 620 ? 6 : 4)
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(
    (drawn[0]?.points ?? []).map((point) => point.x),
    width,
  )

  // Dos reservas con cifras parecidas acaban a la misma altura: las etiquetas se separan lo justo
  // para que ninguna tape a la otra.
  const endLabels = (() => {
    const items = drawn
      .map((item) => {
        const last = item.points.at(-1)
        return last ? { color: item.color, label: item.label, value: item.values.at(-1), x: last.x, y: last.y } : null
      })
      .filter(Boolean)
      .sort((a, b) => a.y - b.y)
    let previous = -Infinity
    return items.map((item) => {
      const labelY = Math.max(item.y + 4, previous + 16)
      previous = labelY
      return { ...item, labelY }
    })
  })()

  return (
    <div className="mini-trend reserve-trend" ref={containerRef}>
      <svg
        aria-label={`Evolución de las reservas mes a mes. ${series
          .map((item) => `${item.label} ${formatMoney(item.values.at(-1))}`)
          .join('. ')}.`}
        className="mini-trend-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = getSvgCoordinates([tick], min, max, chartOptions)[0].y
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {formatMoneyCompact(tick)}
              </text>
            </g>
          )
        })}
        {drawn.map((item) => (
          <path
            d={linePath(item.points)}
            fill="none"
            key={item.label}
            stroke={item.color}
            strokeDasharray={item.dashed ? '6 4' : undefined}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2.5"
          />
        ))}
        {endLabels.map((item) => (
          <g key={`${item.label}-fin`}>
            <circle cx={item.x} cy={item.y} fill={item.color} pointerEvents="none" r="3.6" />
            <text className="reserve-end-label" fill={item.color} x={item.x + 8} y={item.labelY}>
              {formatMoney(item.value)}
            </text>
          </g>
        ))}
        {monthLabelIndices.map((index) => (
          <text
            className="mini-axis-month"
            key={months[index] ?? index}
            textAnchor={index === 0 ? 'start' : index === months.length - 1 ? 'end' : 'middle'}
            x={drawn[0]?.points[index]?.x}
            y={height - 7}
          >
            {formatMonthLabel(months[index])}
          </text>
        ))}
        <rect className="chart-frame" height={plotBottom - paddingTop} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          handlers={handlers}
          isHovering={isHovering}
          label="Recorrer la evolución de las reservas"
          plot={plot}
          pointColor={drawn[0]?.color}
          points={drawn[0]?.points ?? []}
        />
      </svg>
      {isHovering && drawn[0]?.points[activeIndex] && (
        <div className="mini-trend-tooltip" style={{ left: tooltipLeft(drawn[0].points[activeIndex].x, width) }}>
          <span>{formatMonthLabel(months[activeIndex])}</span>
          {drawn.map((item) => (
            <span className="mini-tooltip-row" key={item.label}>
              <i style={{ background: item.color }} />
              {item.label} {formatMoney(item.values[activeIndex])}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ExpenseCategoryLegend({ activeFields, categoryFields, onToggle, showAverage = true }) {
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
      {showAverage ? (
        <span className="expense-avg-legend">
          <i />
          Media móvil
        </span>
      ) : null}
    </div>
  )
}

/**
 * Comparativa por bloques en barras horizontales apiladas.
 *
 * Cada fila reparte el 100 % del gasto de su bloque entre las categorías, así que lo que se
 * compara de un bloque a otro es la composición. La magnitud absoluta no cabe en el ancho sin
 * aplastar los bloques pequeños, de modo que va aparte, como cifra a la derecha de la barra.
 */
function ExpenseBlocksChart({ ariaLabel, categoryFields, getLabel, rows }) {
  const [active, setActive] = useState(null)

  const blocks = rows.map((row, index) => {
    const segments = categoryFields
      .map((field) => ({ color: field.color, label: field.label, value: Math.max(0, asNumber(row[field.field])) }))
      .filter((segment) => segment.value > 0)
    const total = segments.reduce((sum, segment) => sum + segment.value, 0)
    let cursor = 0
    const placed = segments.map((segment) => {
      const share = total > 0 ? (segment.value / total) * 100 : 0
      const mid = cursor + share / 2
      cursor += share
      return { ...segment, mid, share }
    })
    return { index, key: `${row.Mes}-${row.MesFin ?? ''}`, label: getLabel(row), segments: placed, total }
  })

  return (
    <div aria-label={ariaLabel} className="block-compare" role="group">
      {blocks.map((block) => (
        <div
          className={[
            'block-compare-row',
            block.index === 0 ? 'is-first' : '',
            block.index === blocks.length - 1 ? 'is-current' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          key={block.key}
        >
          <span className="block-compare-label">{block.label}</span>
          <div className="block-compare-track">
            <div className="block-compare-bar">
              {block.segments.map((segment) => (
                <button
                  aria-label={`${segment.label} en ${block.label}: ${formatMoney(segment.value)}, ${formatPercent(segment.share)} del bloque`}
                  key={segment.label}
                  onBlur={() => setActive(null)}
                  onFocus={() => setActive({ ...segment, block: block.label, key: block.key })}
                  onMouseEnter={() => setActive({ ...segment, block: block.label, key: block.key })}
                  onMouseLeave={() => setActive(null)}
                  style={{ background: segment.color, width: `${segment.share}%` }}
                  type="button"
                />
              ))}
              {block.total === 0 ? <span className="block-compare-rest" /> : null}
            </div>
            {active && active.key === block.key ? (
              <div className="mini-trend-tooltip block-compare-tip" style={{ left: `${active.mid}%` }}>
                <span>{active.block}</span>
                <span className="mini-tooltip-row">
                  <i style={{ background: active.color }} />
                  {active.label} {formatMoney(active.value)}
                </span>
                <strong>{formatPercent(active.share)} del bloque</strong>
              </div>
            ) : null}
          </div>
          <strong className="block-compare-total">{formatMoney(block.total)}</strong>
        </div>
      ))}
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
  const height = 240
  const paddingRight = 12
  const paddingTop = 16
  const paddingBottom = 32

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
  const averagePath = linePath(averagePoints)
  const labelStep = Math.max(1, Math.ceil(rows.length / maxLabels))
  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: paddingTop + plotHeight }
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(
    bars.map((bar) => bar.barX + barWidth / 2),
    width,
  )
  const activeBar = isHovering ? bars[activeIndex] : null

  return (
    <div className="expense-chart" ref={containerRef}>
      <svg aria-label={ariaLabel} className="mini-trend-svg" height={height} ref={svgRef} viewBox={`0 0 ${width} ${height}`}>
        {yTicks.map((tick) => {
          const y = scaleY(tick)
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {formatMoneyCompact(tick)}
              </text>
            </g>
          )
        })}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        {bars.map((bar) => (
          <g key={bar.key}>
            {bar.segments.map((segment) => (
              <rect fill={segment.color} height={segment.height} key={segment.label} rx="1.5" width={barWidth} x={bar.barX} y={segment.y} />
            ))}
          </g>
        ))}
        <path d={averagePath} fill="none" stroke="#14213d" strokeDasharray="5 4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
        {averagePoints.map((point, index) => (
          <circle cx={point.x} cy={point.y} fill="#14213d" key={bars[index]?.key ?? index} pointerEvents="none" r="2.6" />
        ))}
        {bars.map((bar, index) =>
          index % labelStep === 0 ? (
            <text className="mini-axis-month" key={bar.key} textAnchor="middle" x={bar.barX + barWidth / 2} y={height - 8}>
              {bar.label}
            </text>
          ) : null,
        )}
        <rect className="chart-frame" height={plotHeight} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={null}
          handlers={handlers}
          isHovering={isHovering}
          label="Recorrer los gastos por categoría"
          plot={plot}
          pointColor="#14213d"
          points={averagePoints}
        />
      </svg>
      {activeBar && (
        <div className="mini-trend-tooltip expense-chart-tooltip" style={{ left: tooltipLeft(activeBar.barX + barWidth / 2, width) }}>
          <span>{activeBar.label}</span>
          {activeBar.segments.map((segment) => (
            <span className="mini-tooltip-row" key={segment.label}>
              <i style={{ background: segment.color }} />
              {segment.label} {formatMoney(segment.value)}
            </span>
          ))}
          <strong>Total {formatMoney(activeBar.segments.reduce((sum, segment) => sum + segment.value, 0))}</strong>
        </div>
      )}
    </div>
  )
}

/**
 * Velas mensuales del ahorro acumulado.
 *
 * El cuerpo va del saldo con el que se abre el mes al saldo con el que se cierra, y se pinta en
 * verde o rojo según suba o baje. Las mechas son las **cotas reales del recorrido del mes**: si
 * todos los ingresos hubieran entrado antes de gastar nada, el saldo habría tocado
 * `apertura + ingresos`; si se hubiera gastado todo primero, `apertura − gastos`. No son máximos
 * y mínimos observados día a día —el detalle diario no llega a esta pantalla—, sino los límites
 * entre los que el saldo se movió con seguridad. Una vela de cuerpo corto y mechas largas es un
 * mes de mucho movimiento que acabó casi donde empezó.
 */
function buildSavingsCandles(rows) {
  let running = 0
  return rows.map((row) => {
    const open = running
    const income = asNumber(row.ingresos)
    const spend = asNumber(row.gastos)
    const close = open + asNumber(row.balance)
    running = close
    return {
      Mes: row.Mes,
      close,
      high: Math.max(open, close, open + income),
      income,
      low: Math.min(open, close, open - spend),
      open,
      spend,
    }
  })
}

function SavingsCandleChart({ candles }) {
  const [containerRef, width] = useElementWidth(640)
  const height = getChartHeight(width, CHART_ASPECT, 240, 560)
  const paddingRight = 14
  const paddingTop = 18
  const paddingBottom = 34

  const lows = candles.map((candle) => candle.low)
  const highs = candles.map((candle) => candle.high)
  const { min, max } = getValueDomain([...lows, ...highs])
  const paddingLeft = axisPaddingLeft(min, max)
  const plotWidth = Math.max(0, width - paddingLeft - paddingRight)
  const plotHeight = height - paddingTop - paddingBottom
  const slot = candles.length > 0 ? plotWidth / candles.length : 0
  const bodyWidth = Math.max(3, Math.min(26, slot - 8))
  const scaleY = (value) => height - paddingBottom - ((value - min) / (max - min || 1)) * plotHeight

  const drawn = candles.map((candle, index) => {
    const center = paddingLeft + index * slot + slot / 2
    const top = scaleY(Math.max(candle.open, candle.close))
    const bottom = scaleY(Math.min(candle.open, candle.close))
    return {
      ...candle,
      bodyHeight: Math.max(1.5, bottom - top),
      bodyTop: top,
      center,
      // Una vela de un mes sin cambio sería invisible: se le deja el grosor mínimo de una línea.
      isUp: candle.close >= candle.open,
      wickBottom: scaleY(candle.low),
      wickTop: scaleY(candle.high),
    }
  })

  const yTicks = getNiceTicks(min, max, height > 300 ? 6 : 4)
  const labelIndices = pickLabelIndices(candles.length, width > 640 ? 8 : 5)
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: paddingTop + plotHeight }
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(drawn.map((candle) => candle.center), width)
  const active = drawn[isHovering ? activeIndex : drawn.length - 1] ?? null

  return (
    <div className="expense-chart" ref={containerRef}>
      {active ? (
        <div className="chart-readout">
          <span className="chart-readout-label">Ahorro acumulado</span>
          <strong className="chart-readout-value">{formatMoney(active.close)}</strong>
          <span className={`chart-readout-change ${active.isUp ? 'is-up' : 'is-down'}`}>
            {`${active.close - active.open >= 0 ? '+' : ''}${formatMoney(active.close - active.open)}`}
          </span>
          <span className="chart-readout-stamp">{formatMonthLabel(active.Mes)}</span>
        </div>
      ) : null}
      <svg
        aria-label="Velas mensuales del ahorro acumulado"
        className="mini-trend-svg candle-chart-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = scaleY(tick)
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 8} y={y + 4}>
                {formatMoneyCompact(tick)}
              </text>
            </g>
          )
        })}
        {drawn.map((candle) => (
          <g className={candle.isUp ? 'candle is-up' : 'candle is-down'} key={candle.Mes}>
            <line x1={candle.center} x2={candle.center} y1={candle.wickTop} y2={candle.wickBottom} />
            <rect
              height={candle.bodyHeight}
              rx="1.5"
              width={bodyWidth}
              x={candle.center - bodyWidth / 2}
              y={candle.bodyTop}
            />
          </g>
        ))}
        {labelIndices.map((index) => (
          <text className="mini-axis-month" key={drawn[index]?.Mes ?? index} textAnchor="middle" x={drawn[index]?.center} y={height - 10}>
            {formatMonthLabel(drawn[index]?.Mes)}
          </text>
        ))}
        <rect className="chart-frame" height={plotHeight} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        {isHovering && active ? (
          <line className="chart-crosshair" x1={active.center} x2={active.center} y1={plot.top} y2={plot.bottom} />
        ) : null}
        <rect
          aria-label="Recorrer las velas del ahorro"
          className="chart-capture"
          height={Math.max(0, plot.bottom - plot.top)}
          role="application"
          tabIndex={0}
          width={Math.max(0, plot.right - plot.left)}
          x={plot.left}
          y={plot.top}
          {...handlers}
        />
      </svg>
      {active ? (
        <div className="candle-detail">
          <span>Apertura <strong>{formatMoney(active.open)}</strong></span>
          <span>Cierre <strong>{formatMoney(active.close)}</strong></span>
          <span>Ingresos <strong>{formatMoney(active.income)}</strong></span>
          <span>Gastos <strong>{formatMoney(active.spend)}</strong></span>
        </div>
      ) : null}
    </div>
  )
}

function SavingsAmountChart({ aspect = CHART_ASPECT, color = '#0b6b63', getLabel = (row) => formatMonthLabel(row.Mes), maxLabels = 8, negativeColor = '#b3261e', rows }) {
  const [containerRef, width] = useElementWidth(640)
  const height = getChartHeight(width, aspect)
  const paddingRight = 12
  const paddingTop = 16
  const paddingBottom = 32

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
  const averagePath = linePath(averagePoints)
  const labelStep = Math.max(1, Math.ceil(rows.length / maxLabels))
  const yTicks = getNiceTicks(min, max, height > 260 ? 5 : 4)
  const plot = { left: paddingLeft, right: width - paddingRight, top: paddingTop, bottom: paddingTop + plotHeight }
  const { activeIndex, handlers, isHovering, svgRef } = useChartCrosshair(
    bars.map((bar) => bar.barX + barWidth / 2),
    width,
  )
  const activeBar = isHovering ? bars[activeIndex] : null

  return (
    <div className="expense-chart" ref={containerRef}>
      <svg
        aria-label="Cantidad ahorrada por bloques de meses anteriores, con media móvil"
        className="mini-trend-svg"
        height={height}
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
      >
        {yTicks.map((tick) => {
          const y = scaleY(tick)
          return (
            <g key={tick}>
              <line className="mini-grid-line" x1={paddingLeft} x2={width - paddingRight} y1={y} y2={y} />
              <text className="mini-axis-label" textAnchor="end" x={paddingLeft - 6} y={y + 3}>
                {formatMoneyCompact(tick)}
              </text>
            </g>
          )
        })}
        <line className="mini-zero-line" x1={paddingLeft} x2={width - paddingRight} y1={zeroY} y2={zeroY} />
        {bars.map((bar) => (
          <rect fill={bar.value >= 0 ? color : negativeColor} height={bar.barHeight} key={bar.key} rx="1.5" width={barWidth} x={bar.barX} y={bar.barY} />
        ))}
        <path d={averagePath} fill="none" stroke="#14213d" strokeDasharray="5 4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.6" />
        {averagePoints.map((point, index) => (
          <circle cx={point.x} cy={point.y} fill="#14213d" key={bars[index]?.key ?? index} pointerEvents="none" r="2.6" />
        ))}
        {bars.map((bar, index) =>
          index % labelStep === 0 ? (
            <text className="mini-axis-month" key={bar.key} textAnchor="middle" x={bar.barX + barWidth / 2} y={height - 8}>
              {bar.label}
            </text>
          ) : null,
        )}
        <rect className="chart-frame" height={plotHeight} width={width - paddingLeft - paddingRight} x={paddingLeft} y={paddingTop} />
        <ChartCrosshair
          activeIndex={activeIndex}
          baselineY={null}
          handlers={handlers}
          isHovering={isHovering}
          label="Recorrer las cantidades ahorradas"
          plot={plot}
          pointColor="#14213d"
          points={averagePoints}
        />
      </svg>
      {activeBar && (
        <div className="mini-trend-tooltip expense-chart-tooltip" style={{ left: tooltipLeft(activeBar.barX + barWidth / 2, width) }}>
          <span>{activeBar.label}</span>
          <strong>{formatMoney(activeBar.value)}</strong>
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
  onOpenInput,
  params,
  onObjectiveFilterChange,
  onExportResult,
  onExportHistoryCsv,
  onExportObjectivesCsv,
}) {
  const [activeTab, setActiveTab] = useState('resumen')
  const [typologyPeriod, setTypologyPeriod] = useState('12')
  const [featuredTypology, setFeaturedTypology] = useState(typologyCards[0].field)
  // Índice que el ratón está señalando en la gráfica grande: gobierna la cifra de cabecera.
  const [featuredActiveIndex, setFeaturedActiveIndex] = useState(null)
  const [expensePeriod, setExpensePeriod] = useState('6')
  const [activeExpenseCategories, setActiveExpenseCategories] = useState(() => expenseCategoryFields.map((field) => field.field))
  const [savingsPeriod, setSavingsPeriod] = useState('12')
  const [savingsAmountPeriod, setSavingsAmountPeriod] = useState('6')
  const [savingsCandlePeriod, setSavingsCandlePeriod] = useState('12')
  const [savingsPercentActiveIndex, setSavingsPercentActiveIndex] = useState(null)
  const [typologyCompareActiveIndex, setTypologyCompareActiveIndex] = useState(null)
  const [yieldActiveIndex, setYieldActiveIndex] = useState(null)
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
      color: field?.color ?? '#9aa3b2',
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
  // Las velas se construyen sobre TODO el histórico y se recorta después: si se recortara antes,
  // la primera vela abriría en cero y el acumulado saldría mal.
  const savingsCandles = getPeriodRows(buildSavingsCandles(scopedSavingsRows), savingsCandlePeriod)
  const investmentContributionRows = getInvestmentContributions(result).slice(-12)
  const investmentRows = rows.slice(-12)
  const investmentMonths = investmentRows.map((row) => row.Mes)
  const investmentReserveValues = investmentRows.map((row) => asNumber(row['📈 Inversiones']))
  const investmentInvestedValues = investmentRows.map((row) => asNumber(row['Dinero Invertido']))
  // Rendimiento acumulado sobre lo invertido: qué parte del dinero puesto han devuelto ya los
  // intereses y los dividendos. Se mide sobre el invertido de cada mes, así que baja cuando se
  // aporta dinero nuevo aunque el cobro siga creciendo.
  const investmentYieldRows = investmentRows.map((row) => {
    const invested = asNumber(row['Dinero Invertido'])
    const dividends = asNumber(row.Dividendos)
    const interestTotal = Math.abs(getCategoryTotalUntilMonth(incomeAnalysis, row.Mes, ['interes', 'intereses']))
    const interestOnly = Math.max(0, interestTotal - dividends)
    return {
      dividends: invested > 0 ? (dividends / invested) * 100 : 0,
      interests: invested > 0 ? (interestOnly / invested) * 100 : 0,
    }
  })
  const investmentInterestPctValues = investmentYieldRows.map((row) => row.interests)
  const investmentDividendPctValues = investmentYieldRows.map((row) => row.dividends)
  const reserveRows = rows.slice(-12)
  const reserveMonths = reserveRows.map((row) => row.Mes)
  const reserveSeries = [
    { color: '#0b6b63', label: 'Fondo de reserva', values: reserveRows.map((row) => asNumber(row['Fondo de reserva cargado'])) },
    { color: '#b45309', label: 'Vacaciones', values: reserveRows.map((row) => asNumber(row['💼 Vacaciones'])) },
    { color: '#1d3f9a', dashed: true, label: 'Regalos', values: reserveRows.map((row) => asNumber(row['🎁 Regalos'])) },
  ]
  const typologyRows = getPeriodRows(rows, typologyPeriod)
  const typologyMonths = typologyRows.map((row) => row.Mes)
  const featuredCard = typologyCards.find((card) => card.field === featuredTypology) ?? typologyCards[0]
  const featuredValues = typologyRows.map((row) => asNumber(row[featuredCard.field]))
  const otherTypologyCards = typologyCards.filter((card) => card.field !== featuredCard.field)
  const reserveFund = getReserveFund(result)
  const summaryGroups = selectedRow ? getSummaryGroups(selectedRow, reserveFund) : null
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
                            {typeof item.progress === 'number' ? (
                              <div className="ratio-bar metric-progress">
                                <span style={{ width: `${Math.max(0, Math.min(100, item.progress))}%` }} />
                              </div>
                            ) : null}
                            {item.note ? <small className="metric-note">{item.note}</small> : null}
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
                    <div className="data-list month-data">
                      <h3 className="table-title">Datos del mes</h3>
                      <dl>
                        <div>
                          <dt>Gastos</dt>
                          <dd>{result.movimientos.gastos}</dd>
                        </div>
                        <div>
                          <dt>Ingresos</dt>
                          <dd>{result.movimientos.ingresos}</dd>
                        </div>
                        <div>
                          <dt>Transferencias</dt>
                          <dd>{result.movimientos.transferencias}</dd>
                        </div>
                        <div>
                          <dt>Cuentas</dt>
                          <dd>{result.movimientos.cuentas}</dd>
                        </div>
                        <div>
                          <dt>Objetivos</dt>
                          <dd>{result.historial.objetivos.length}</dd>
                        </div>
                      </dl>
                    </div>
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

                <section className="trend-panel">
                  <h3 className="table-title">Tendencia del patrimonio</h3>
                  <div className="trend-list">
                    {recentRows.map((row, index) => (
                      <article
                        className={index === recentRows.length - 1 ? 'trend-row is-current' : 'trend-row'}
                        key={row.Mes}
                      >
                        <span className="trend-month">{row.Mes}</span>
                        <div className="trend-bar total">
                          <span style={{ width: `${Math.max(3, (Math.abs(asNumber(row.total)) / trendMax) * 100)}%` }} />
                        </div>
                        <strong>{formatMoney(row.total)}</strong>
                      </article>
                    ))}
                  </div>
                </section>
              </>
            )}

            {activeTab === 'presupuesto' && budget && (
              <section className="budget-layout">
                <div className="budget-card budget-visual">
                  <div className="budget-hero">
                    <h3 className="table-title">Presupuesto de {selectedRow.Mes}</h3>
                    <p className="budget-hero-value">
                      {budget.overrun > 0 ? `Te has pasado ${formatMoney(budget.overrun)}` : `Quedan ${formatMoney(budget.remaining)}`}
                    </p>
                    <p className="budget-hero-note">
                      {budget.overrun > 0
                        ? `sobre el presupuesto disponible de ${formatMoney(budget.availableBudget)}`
                        : budget.availableBudget === 0 && budget.accumulatedDebt > 0
                          ? `la deuda acumulada se lleva todo el presupuesto de ${formatMoney(budget.monthBudget)}`
                          : `disponibles este mes, de un presupuesto de ${formatMoney(budget.monthBudget)}`}
                    </p>
                  </div>
                  <div className="budget-bar-wrap">
                    <div className="budget-bar" aria-label="Distribución del presupuesto mensual">
                      {budget.debtCharge > 0 && (
                        <span
                          aria-label={`Deuda cubierta: ${formatMoney(budget.debtCharge)}`}
                          className="budget-segment reserved"
                          onBlur={() => setBudgetTooltip(null)}
                          onFocus={() => setBudgetTooltip({ label: 'Deuda cubierta', value: budget.debtCharge })}
                          onMouseEnter={() => setBudgetTooltip({ label: 'Deuda cubierta', value: budget.debtCharge })}
                          onMouseLeave={() => setBudgetTooltip(null)}
                          role="img"
                          style={{ width: getSegmentWidth(budget.debtCharge, budget.totalForBar) }}
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
                  <div className="budget-legend">
                    <span className="legend-spent">
                      <i aria-hidden="true" />
                      Gasto neto
                      <strong>{formatMoney(budget.spent)}</strong>
                    </span>
                    <span className="legend-remaining">
                      <i aria-hidden="true" />
                      Disponible
                      <strong>{formatMoney(budget.remaining)}</strong>
                    </span>
                    <span className="legend-debt">
                      <i aria-hidden="true" />
                      Deuda cubierta
                      <strong>{formatMoney(budget.debtCharge)}</strong>
                    </span>
                    {budget.budgetRefund > 0 ? (
                      <span className="legend-refund">
                        <i aria-hidden="true" />
                        Ajuste a favor
                        <strong>{formatMoney(budget.budgetRefund)}</strong>
                      </span>
                    ) : null}
                  </div>
                  <div className="budget-execution">
                    <div className="budget-execution-head">
                      <span>Ejecutado</span>
                      <strong>{formatPercent(budget.execution)}</strong>
                    </div>
                    <div className="ratio-bar">
                      <span style={{ width: `${Math.max(0, Math.min(100, budget.execution))}%` }} />
                    </div>
                  </div>
                </div>

                <div className="budget-summary">
                  <div className="data-list">
                    <h3 className="table-title">Presupuesto</h3>
                    <dl>
                      <div>
                        <dt>Presupuesto mes</dt>
                        <dd>{formatMoney(budget.monthBudget)}</dd>
                      </div>
                      <div>
                        <dt>Gasto neto</dt>
                        <dd>{formatMoney(budget.spent)}</dd>
                      </div>
                      <div>
                        <dt>Disponible tras la deuda</dt>
                        <dd>{formatMoney(budget.remaining)}</dd>
                      </div>
                    </dl>
                  </div>
                  <div className="data-list">
                    <h3 className="table-title">Deuda y exceso</h3>
                    <dl>
                      <div>
                        <dt>Deuda cubierta</dt>
                        <dd>{formatMoney(budget.debtCharge)}</dd>
                      </div>
                      <div>
                        <dt>Deuda que sigue pendiente</dt>
                        <dd>{formatMoney(budget.debtCarried)}</dd>
                      </div>
                      <div>
                        <dt>Exceso del mes</dt>
                        <dd>{formatMoney(budget.monthlyDebt)}</dd>
                      </div>
                      <div>
                        <dt>Exceso acumulado</dt>
                        <dd>{formatMoney(budget.accumulatedDebt)}</dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </section>
            )}

            {activeTab === 'gastos' && (
              expenseCategoryTimeline.length > 0 ? (
                <section className="expenses-layout">
                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Gastos por categoría</h3>
                      <PeriodTabs
                        ariaLabel="Periodo de gastos"
                        onChange={setExpensePeriod}
                        options={expensePeriods}
                        value={expensePeriod}
                      />
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
                    <ExpenseBlocksChart
                      ariaLabel="Reparto del gasto por categoría en cada bloque de meses"
                      categoryFields={visibleExpenseCategoryFields}
                      getLabel={formatExpenseBucketLabel}
                      rows={expenseBucketedRows}
                    />
                    <ExpenseCategoryLegend
                      activeFields={activeExpenseCategories}
                      categoryFields={expenseCategoryFields}
                      onToggle={toggleExpenseCategory}
                      showAverage={false}
                    />
                    <p className="muted-text">
                      Cada barra reparte el 100 % del gasto de su bloque. La cifra de la derecha es el total del bloque.
                    </p>
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
                      <strong>{formatPercent(asNumber(selectedSavingsRow?.porcentaje_ahorro))}</strong>
                      <small>
                        {asNumber(selectedSavingsRow?.gastos) === 0 && asNumber(selectedSavingsRow?.ingresos) > 0
                          ? 'Sin gastos registrados'
                          : 'Sobre los ingresos del mes'}
                      </small>
                    </article>
                  </div>

                  <div className="chart-grid">
                  <section className="typology-chart-panel chart-card chart-card-lead">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Porcentaje de ahorro</h3>
                        <span className="muted-text">Comparado con la media móvil de 3 meses</span>
                      </div>
                      <PeriodTabs
                        ariaLabel="Periodo de ahorro"
                        onChange={setSavingsPeriod}
                        options={expensePeriods}
                        value={savingsPeriod}
                      />
                    </div>
                    <ChartReadout
                      activeIndex={savingsPercentActiveIndex}
                      formatDelta={formatPoints}
                      formatValue={formatPercent}
                      label="Ahorro del mes"
                      labels={savingsMonths.map((month) => formatMonthLabel(month))}
                      showRelative={false}
                      values={savingsPercentValues}
                    />
                    <div className="typology-compare">
                      <div className="typology-compare-legend">
                        <span>
                          <i style={{ background: '#0b6b63' }} />
                          Ahorro mensual
                        </span>
                        <span>
                          <i className="dashed" style={{ borderColor: '#14213d' }} />
                          Media móvil
                        </span>
                      </div>
                      <TypologyCompareChart
                        formatCompact={formatPercent}
                        formatValue={formatPercent}
                        months={savingsMonths}
                        onActiveIndexChange={setSavingsPercentActiveIndex}
                        seriesA={{ color: '#0b6b63', label: 'Ahorro mensual', values: savingsPercentValues }}
                        seriesB={{ color: '#14213d', label: 'Media móvil', values: savingsAverageValues }}
                      />
                    </div>
                  </section>

                  <section className="expenses-panel chart-card chart-card-lead">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Ahorro acumulado, mes a mes</h3>
                        <span className="muted-text">Cada vela va de la apertura al cierre del mes</span>
                      </div>
                      <PeriodTabs
                        ariaLabel="Periodo de las velas de ahorro"
                        onChange={setSavingsCandlePeriod}
                        options={expensePeriods}
                        value={savingsCandlePeriod}
                      />
                    </div>
                    <SavingsCandleChart candles={savingsCandles} />
                    <div className="expense-chart-legend">
                      <span>
                        <i style={{ background: '#0b6b63' }} />
                        Mes al alza
                      </span>
                      <span>
                        <i style={{ background: '#b3261e' }} />
                        Mes a la baja
                      </span>
                      <span className="expense-avg-legend">
                        Mecha arriba: apertura + ingresos · Mecha abajo: apertura − gastos
                      </span>
                    </div>
                  </section>

                  <section className="expenses-panel chart-card chart-card-lead">
                    <div className="table-toolbar compact-toolbar">
                      <div>
                        <h3 className="table-title">Cantidad ahorrada</h3>
                        <span className="muted-text">
                          Bloques de {savingsBucketSize} {savingsBucketSize === 1 ? 'mes' : 'meses'}, últimos {savingsBucketedRows.length}
                        </span>
                      </div>
                      <PeriodTabs
                        ariaLabel="Periodo de importes ahorrados"
                        onChange={setSavingsAmountPeriod}
                        options={expensePeriods}
                        value={savingsAmountPeriod}
                      />
                    </div>
                    <SavingsAmountChart getLabel={formatExpenseBucketLabel} rows={savingsBucketedRows} />
                    <div className="expense-chart-legend">
                      <span>
                        <i style={{ background: '#0b6b63' }} />
                        Ahorro positivo
                      </span>
                      <span>
                        <i style={{ background: '#b3261e' }} />
                        Ahorro negativo
                      </span>
                      <span className="expense-avg-legend">
                        <i />
                        Media móvil
                      </span>
                    </div>
                  </section>
                  </div>
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

                <div className="chart-grid">
                <section className="typology-chart-panel chart-card chart-card-lead">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Evolución inversiones</h3>
                      <span className="muted-text">Bolsa disponible frente a dinero invertido</span>
                    </div>
                  </div>
                  <div className="typology-compare">
                    <div className="typology-compare-legend">
                      <span>
                        <i style={{ background: '#0b6b63' }} />
                        Bolsa
                      </span>
                      <span>
                        <i className="dashed" style={{ borderColor: '#0b6b63' }} />
                        Invertido
                      </span>
                    </div>
                    <TypologyCompareChart
                      months={investmentMonths}
                      seriesA={{ color: '#0b6b63', label: 'Bolsa', values: investmentReserveValues }}
                      seriesB={{ color: '#0b6b63', label: 'Invertido', values: investmentInvestedValues }}
                    />
                  </div>
                </section>

                <section className="typology-chart-panel chart-card chart-card-lead">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Aportación a la bolsa</h3>
                      <span className="muted-text">De dónde sale cada mes, frente a lo que se invirtió</span>
                    </div>
                  </div>
                  {investmentContributionRows.length > 0 ? (
                    <>
                      <div className="expense-chart-legend">
                        {contributionParts.map((part) => (
                          <span key={part.key}>
                            <i style={{ background: part.color }} />
                            {part.label}
                          </span>
                        ))}
                        <span>
                          <i style={{ background: investedColor }} />
                          Invertido
                        </span>
                      </div>
                      <InvestmentContributionChart rows={investmentContributionRows} />
                    </>
                  ) : (
                    <div className="empty-state compact-empty">
                      <strong>Sin aportaciones registradas</strong>
                      <span>Procesa un Excel con ingresos para ver de dónde sale el dinero invertido.</span>
                    </div>
                  )}
                </section>

                <section className="typology-chart-panel chart-card chart-card-lead">
                  <div className="table-toolbar compact-toolbar">
                    <div>
                      <h3 className="table-title">Rendimiento sobre lo invertido</h3>
                      <span className="muted-text">Qué porcentaje del dinero invertido han devuelto ya intereses y dividendos</span>
                    </div>
                  </div>
                  <ChartReadout
                    activeIndex={yieldActiveIndex}
                    formatDelta={formatPoints}
                    formatValue={formatPercent}
                    label="Intereses sobre lo invertido"
                    labels={investmentMonths.map((month) => formatMonthLabel(month))}
                    showRelative={false}
                    values={investmentInterestPctValues}
                  />
                  <div className="typology-compare">
                    <div className="typology-compare-legend">
                      <span>
                        <i style={{ background: '#0b6b63' }} />
                        Intereses
                      </span>
                      <span>
                        <i className="dashed" style={{ borderColor: '#b45309' }} />
                        Dividendos
                      </span>
                    </div>
                    <TypologyCompareChart
                      formatCompact={formatPercent}
                      formatValue={formatPercent}
                      months={investmentMonths}
                      onActiveIndexChange={setYieldActiveIndex}
                      seriesA={{ color: '#0b6b63', label: 'Intereses', values: investmentInterestPctValues }}
                      seriesB={{ color: '#b45309', label: 'Dividendos', values: investmentDividendPctValues }}
                    />
                  </div>
                </section>

                <section className="typology-chart-panel chart-card chart-card-lead">
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
                </div>
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

                <section className="trend-panel reserve-chart-panel">
                  <div className="chart-header">
                    <h3 className="table-title">Evolución de las reservas</h3>
                    <span>{reserveMonths.length > 0 ? `De ${formatMonthLong(reserveMonths[0])} a ${formatMonthLong(reserveMonths.at(-1))}` : ''}</span>
                  </div>
                  <ReserveLinesChart months={reserveMonths} series={reserveSeries} />
                  <div className="chart-legend-row">
                    {reserveSeries.map((item) => (
                      <span key={item.label}>
                        <i aria-hidden="true" style={{ background: item.color }} />
                        {item.label}
                      </span>
                    ))}
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
                      <PeriodTabs
                        ariaLabel="Periodo de tipologías"
                        onChange={setTypologyPeriod}
                        options={typologyPeriods}
                        value={typologyPeriod}
                      />
                    </div>
                  </div>
                  {typologyRows.length > 0 ? (
                    <div className="chart-grid">
                      <article className="chart-card chart-card-lead typology-hero">
                        <ChartReadout
                          activeIndex={featuredActiveIndex}
                          className="typology-hero-head"
                          formatDelta={formatDelta}
                          formatValue={formatMoney}
                          label={featuredCard.label}
                          labels={typologyMonths.map((month) => formatMonthLabel(month))}
                          periodLabel={`${typologyRows.length} meses`}
                          values={featuredValues}
                        />
                        <TypologyMiniChart
                          color={featuredCard.color}
                          label={featuredCard.label}
                          months={typologyMonths}
                          onActiveIndexChange={setFeaturedActiveIndex}
                          values={featuredValues}
                        />
                      </article>

                      <article className="chart-card chart-card-lead typology-compare">
                        <ChartReadout
                          activeIndex={typologyCompareActiveIndex}
                          className="typology-hero-head"
                          formatDelta={formatDelta}
                          formatValue={formatMoney}
                          label={typologyCompareFields[1].label}
                          labels={typologyMonths.map((month) => formatMonthLabel(month))}
                          periodLabel={`${typologyRows.length} meses`}
                          values={typologyRows.map((row) => asNumber(row[typologyCompareFields[1].field]))}
                        />
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
                          onActiveIndexChange={setTypologyCompareActiveIndex}
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

                      {otherTypologyCards.map(({ field, label, color }) => {
                        const values = typologyRows.map((row) => asNumber(row[field]))
                        return (
                          <TypologyMiniCard
                            color={color}
                            key={field}
                            label={label}
                            months={typologyMonths}
                            onFeature={() => setFeaturedTypology(field)}
                            values={values}
                          />
                        )
                      })}
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
                  <div className="prediction-table">
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Tipología</th>
                          <th scope="col">Conservadora</th>
                          <th scope="col">Base</th>
                          <th scope="col">Optimista</th>
                        </tr>
                      </thead>
                      <tbody>
                        {typologyFields.map(({ field, label }) => {
                          const prediction = predictTypology(typologyRows, field)
                          return (
                            <tr key={field}>
                              <th scope="row">{label}</th>
                              <td>{formatMoney(prediction.conservative)}</td>
                              <td className="is-base">{formatMoney(prediction.base)}</td>
                              <td>{formatMoney(prediction.optimistic)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
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
                <div className="empty-state objectives-empty">
                  <div className="empty-state-copy">
                    <strong>Aún no hay objetivos calculados</strong>
                    <span>Añade un objetivo en Datos Excel y procesa el Excel. Aquí verás cuánto llevas y cuándo lo alcanzarás.</span>
                    <button className="primary-button" onClick={onOpenInput} type="button">
                      Ir a Datos Excel
                    </button>
                  </div>
                  {reserveFund.objetivo > 0 ? (
                    <div className="objective-example">
                      <span>Tu fondo de reserva ya funciona así</span>
                      <strong>
                        Fondo de emergencia <em>{formatMoney(reserveFund.cargado)} de {formatMoney(reserveFund.objetivo)}</em>
                      </strong>
                      <div className="ratio-bar">
                        <span style={{ width: `${Math.max(0, Math.min(100, reserveFund.porcentaje * 100))}%` }} />
                      </div>
                      <span>
                        {reserveFund.pendiente > 0
                          ? `Te faltan ${formatMoney(reserveFund.pendiente)}.`
                          : 'Fondo completo: ya cubre el objetivo.'}{' '}
                        El objetivo son {reserveFund.mesesDeCobertura} meses de tu gasto medio, y se recalcula con cada
                        Excel.
                      </span>
                    </div>
                  ) : null}
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
                  <div className="table-toolbar">
                    <h3 className="table-title">Historial</h3>
                    <span className="muted-text">Del mes más reciente al más antiguo. Desliza para ver más columnas.</span>
                  </div>
                  <table>
                    <thead>
                      <tr>
                        {historyFields.map((field) => (
                          <th key={field.field}>{field.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows
                        .slice(-24)
                        .slice()
                        .reverse()
                        .map((row) => (
                          <tr className={row.Mes === selectedRow.Mes ? 'is-current' : ''} key={row.Mes}>
                            {historyFields.map((field) => {
                              const value = row[field.field]
                              const amount = asNumber(value)
                              const tone =
                                field.type !== 'money' ? '' : amount === 0 ? 'is-zero' : amount < 0 ? 'is-negative' : ''
                              return (
                                <td className={tone} key={field.field}>
                                  {field.type === 'money' ? formatMoney(value) : value}
                                </td>
                              )
                            })}
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
        <div className="landing-grid">
          <section className="panel split-preview">
            <div className="chart-header">
              <h3 className="table-title">Tu reparto de ingresos</h3>
              <span>Según los porcentajes de la izquierda</span>
            </div>
            <div className="split-bar" role="img" aria-label="Reparto configurado de los ingresos">
              <span className="split-seg split-gasto" style={{ width: `${Math.max(0, Number(params?.porcentaje_gasto ?? 0) * 100)}%` }} />
              <span className="split-seg split-inversion" style={{ width: `${Math.max(0, Number(params?.porcentaje_inversion ?? 0) * 100)}%` }} />
              <span className="split-seg split-vacaciones" style={{ width: `${Math.max(0, Number(params?.porcentaje_vacaciones ?? 0) * 100)}%` }} />
              <span className="split-seg split-free" />
            </div>
            <div className="split-legend">
              {[
                { key: 'gasto', label: 'Gasto', value: Number(params?.porcentaje_gasto ?? 0) },
                { key: 'inversion', label: 'Inversión', value: Number(params?.porcentaje_inversion ?? 0) },
                { key: 'vacaciones', label: 'Vacaciones', value: Number(params?.porcentaje_vacaciones ?? 0) },
                {
                  key: 'free',
                  label: 'Sin asignar',
                  value: Math.max(
                    0,
                    1 -
                      Number(params?.porcentaje_gasto ?? 0) -
                      Number(params?.porcentaje_inversion ?? 0) -
                      Number(params?.porcentaje_vacaciones ?? 0),
                  ),
                },
              ].map((item) => (
                <span key={item.key}>
                  <i className={`split-dot split-${item.key}`} aria-hidden="true" />
                  {item.label}
                  <strong>{formatPercent(item.value * 100)}</strong>
                </span>
              ))}
            </div>
          </section>

          <section className="panel landing-dropzone">
            <h3 className="table-title">Carga el Excel del mes</h3>
            <p>Al procesarlo verás el último mes, cómo se reparte frente a tus porcentajes y cuánto llevas cumplido de cada objetivo.</p>
            <ul className="landing-list">
              <li>Resumen del último mes</li>
              <li>Real frente a objetivo</li>
              <li>Historial de los últimos meses</li>
            </ul>
            <button className="primary-button" onClick={onOpenInput} type="button">
              Seleccionar Excel
            </button>
          </section>
        </div>
      )}
    </section>
  )
}
