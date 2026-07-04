import { useState } from 'react'
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

const typologyFields = [
  { field: '💰 Ahorros', label: 'Ahorros', color: '#286b57' },
  { field: '💸 Presupuesto Mes', label: 'Presupuesto mensual', color: '#3f7f8f' },
  { field: '💼 Vacaciones', label: 'Vacaciones', color: '#d2a53f' },
  { field: '📈 Inversiones', label: 'Reserva inversión', color: '#5c6f9e' },
  { field: 'Fondo de reserva cargado', label: 'Fondo reserva', color: '#6c7a72' },
  { field: '📉 Deuda Presupuestaria acumulada', label: 'Deuda acumulada', color: '#8f4638' },
  { field: '💳 Gasto del mes', label: 'Gasto mensual', color: '#b85a4b' },
  { field: '🎁 Regalos', label: 'Regalos', color: '#a66a4d' },
]

const defaultTypologyFields = [
  '💰 Ahorros',
  '💸 Presupuesto Mes',
  '💼 Vacaciones',
  '📈 Inversiones',
  'Fondo de reserva cargado',
  '📉 Deuda Presupuestaria acumulada',
  '💳 Gasto del mes',
]

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
  const spentMagnitude = Math.abs(spent)
  const monthlyDebt = asNumber(selectedRow['📉 Deuda Presupuestaria mensual'])
  const accumulatedDebt = asNumber(selectedRow['📉 Deuda Presupuestaria acumulada'])
  const debtReserve = accumulatedDebt > 0 ? Math.min(monthBudget * 0.1, accumulatedDebt) : 0
  const usableBudget = Math.max(0, monthBudget - debtReserve)
  const remaining = Math.max(0, usableBudget - spentMagnitude)
  const overrun = Math.max(monthlyDebt, spentMagnitude - usableBudget, 0)
  const totalForBar = Math.max(monthBudget, debtReserve + spentMagnitude + remaining, 1)
  const execution = usableBudget > 0 ? (Math.min(spentMagnitude, usableBudget) / usableBudget) * 100 : 0

  return {
    accumulatedDebt,
    availableBudget,
    debtReserve,
    execution,
    monthBudget,
    monthlyDebt,
    overrun,
    remaining,
    spent,
    spentMagnitude,
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

function getTypologyBounds(rows, fields) {
  const values = rows.flatMap((row) => fields.map(({ field }) => asNumber(row[field])))
  return {
    max: Math.max(...values, 1),
    min: Math.min(...values, 0),
  }
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
  const [activeTypologyFields, setActiveTypologyFields] = useState(defaultTypologyFields)
  const [typologyPeriod, setTypologyPeriod] = useState('12')
  const [expensePeriod, setExpensePeriod] = useState('6')
  const [typologyTooltip, setTypologyTooltip] = useState(null)
  const [savingsTooltip, setSavingsTooltip] = useState(null)
  const budget = selectedRow ? getBudget(selectedRow) : null
  const expenseAnalysis = getExpenseAnalysis(result)
  const incomeAnalysis = getIncomeAnalysis(result)
  const expenseRows = getExpenseRowsForPeriod(expenseAnalysis.mensual, selectedRow?.Mes, expensePeriod)
  const expenseCategoryRows = getExpenseCategoryRowsForPeriod(expenseAnalysis.categorias, selectedRow?.Mes, expensePeriod)
  const expenseMax = Math.max(...expenseCategoryRows.map((row) => asNumber(row.total)), 1)
  const expenseMonthlyMax = Math.max(...expenseRows.map((row) => Math.abs(asNumber(row.gastos))), 1)
  const savingsAnalysis = getSavingsAnalysis(result)
  const savingsMax = Math.max(...savingsAnalysis.mensual.map((row) => Math.abs(asNumber(row.balance))), 1)
  const savingsPercentRows = savingsAnalysis.mensual.slice(-12)
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
  const investmentMax = Math.max(
    ...rows.map((row) => Math.max(Math.abs(asNumber(row['📈 Inversiones'])), Math.abs(asNumber(row['Dinero Invertido'])))),
    1,
  )
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
  const selectedTypologyFields = typologyFields.filter(({ field }) => activeTypologyFields.includes(field))
  const typologyBounds = getTypologyBounds(typologyRows, selectedTypologyFields)
  const summaryGroups = selectedRow ? getSummaryGroups(selectedRow) : null
  const totalMoney = asNumber(selectedRow?.total)
  const investedMoney = asNumber(selectedRow?.['Dinero Invertido'])
  const investedPct = totalMoney > 0 ? (investedMoney / totalMoney) * 100 : 0
  const interestAmount = getCategoryTotalUntilMonth(incomeAnalysis, selectedRow?.Mes, ['interes', 'intereses'])
  const interestPct = investedMoney > 0 ? (Math.abs(interestAmount) / investedMoney) * 100 : 0
  const toggleTypologyField = (field) => {
    setTypologyTooltip(null)
    setActiveTypologyFields((current) =>
      current.includes(field) ? current.filter((item) => item !== field) : [...current, field],
    )
  }

  return (
    <section className="panel result-panel" aria-busy={isProcessing}>
      <div className="panel-header">
        <h2>Resultados</h2>
        <div className="panel-actions">
          {rows.length > 0 ? (
            <label className="month-selector">
              <span>Mes</span>
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
            <div className="export-actions">
              <button className="text-button" type="button" onClick={onExportResult}>
                JSON
              </button>
              <button className="text-button" type="button" onClick={onExportHistoryCsv}>
                Historial CSV
              </button>
              {objectiveRows.length > 0 && (
                <button className="text-button" type="button" onClick={onExportObjectivesCsv}>
                  Objetivos CSV
                </button>
              )}
            </div>
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
                    {budget.spentMagnitude > 0 && (
                      <span
                        className="budget-segment spent"
                        style={{ width: getSegmentWidth(budget.spentMagnitude, budget.totalForBar) }}
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
                    <span>Gastado {formatMoney(budget.spent)}</span>
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
                    <span>Gastado</span>
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
                  <div className="savings-summary">
                    <article>
                      <span>Ingresos</span>
                      <strong>{formatMoney(savingsAnalysis.ultimo_mes?.ingresos)}</strong>
                    </article>
                    <article>
                      <span>Gastos</span>
                      <strong>{formatMoney(savingsAnalysis.ultimo_mes?.gastos)}</strong>
                    </article>
                    <article>
                      <span>Balance</span>
                      <strong>{formatMoney(savingsAnalysis.ultimo_mes?.balance)}</strong>
                    </article>
                    <article>
                      <span>Ahorro</span>
                      <strong>{asNumber(savingsAnalysis.ultimo_mes?.porcentaje_ahorro).toFixed(1)}%</strong>
                    </article>
                  </div>

                  <section className="trend-panel compact-trend savings-percent-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Porcentaje de ahorro</h3>
                      <span>Media móvil 3 meses</span>
                    </div>
                    <div className="savings-chart-wrap">
                      <svg
                        className="savings-percent-chart"
                        onMouseLeave={() => setSavingsTooltip(null)}
                        viewBox="0 0 640 260"
                        role="img"
                        aria-label="Porcentaje de ahorro mensual"
                      >
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
                        <line className="chart-axis" x1="54" x2="54" y1="24" y2="218" />
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
                        <polyline
                          fill="none"
                          points={getSvgPoints(savingsPercentValues, savingsPercentBounds.min, savingsPercentBounds.max, savingsChart)}
                          stroke="#286b57"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="3"
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
                            <text className="chart-value-label" x={x} y={y < 42 ? y + 18 : y - 10} textAnchor="middle">
                              {formatPercent(savingsPercentValues[index])}
                            </text>
                            <circle
                              className="chart-point"
                              cx={x}
                              cy={y}
                              fill="#286b57"
                              onFocus={() =>
                                setSavingsTooltip({
                                  average: savingsAverageValues[index],
                                  month: savingsPercentRows[index]?.Mes,
                                  value: savingsPercentValues[index],
                                  x,
                                  y,
                                })
                              }
                              onMouseEnter={() =>
                                setSavingsTooltip({
                                  average: savingsAverageValues[index],
                                  month: savingsPercentRows[index]?.Mes,
                                  value: savingsPercentValues[index],
                                  x,
                                  y,
                                })
                              }
                              r="5"
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
                      {savingsAnalysis.mensual.slice(-12).map((row) => (
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

                <section className="trend-panel compact-trend">
                  <h3 className="table-title">Evolución inversiones</h3>
                  <div className="trend-list">
                    {rows.slice(-12).map((row) => (
                      <article className="investment-row" key={row.Mes}>
                        <span className="trend-month">{row.Mes}</span>
                        <div className="investment-bars">
                          <div className="investment-bar planned">
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row['📈 Inversiones'])) / investmentMax) * 100)}%` }} />
                          </div>
                          <div className="investment-bar invested">
                            <span style={{ width: `${Math.max(4, (Math.abs(asNumber(row['Dinero Invertido'])) / investmentMax) * 100)}%` }} />
                          </div>
                        </div>
                        <strong>{formatMoney(row['Dinero Invertido'])}</strong>
                      </article>
                    ))}
                  </div>
                  <div className="investment-legend">
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
                    <label className="period-selector">
                      <span>Periodo</span>
                      <select
                        value={typologyPeriod}
                        onChange={(event) => {
                          setTypologyTooltip(null)
                          setTypologyPeriod(event.target.value)
                        }}
                      >
                        {typologyPeriods.map((period) => (
                          <option key={period.value} value={period.value}>
                            {period.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="typology-controls" aria-label="Seleccionar tipologías">
                    {typologyFields.map(({ field, label, color }) => (
                      <label className="typology-toggle" key={field} style={{ '--legend-color': color }}>
                        <input
                          checked={activeTypologyFields.includes(field)}
                          onChange={() => toggleTypologyField(field)}
                          type="checkbox"
                        />
                        <span>{label}</span>
                      </label>
                    ))}
                  </div>
                  {selectedTypologyFields.length > 0 ? (
                    <>
                      <div className="chart-scale">
                        <span>{formatMoney(typologyBounds.max)}</span>
                        <span>{formatMoney(typologyBounds.min)}</span>
                      </div>
                      <div className="typology-chart-wrap">
                        <svg
                          className="typology-chart"
                          onMouseLeave={() => setTypologyTooltip(null)}
                          viewBox="0 0 640 220"
                          role="img"
                          aria-label="Evolución mensual por tipología"
                        >
                          <line className="chart-axis" x1="18" x2="622" y1="202" y2="202" />
                          <line className="chart-axis" x1="18" x2="18" y1="18" y2="202" />
                          {selectedTypologyFields.map(({ field, label, color }) => {
                            const values = typologyRows.map((row) => asNumber(row[field]))
                            const avgValues = movingAverage(values)
                            const coordinates = getSvgCoordinates(values, typologyBounds.min, typologyBounds.max)
                            return (
                              <g key={field}>
                                <polyline
                                  fill="none"
                                  points={getSvgPoints(values, typologyBounds.min, typologyBounds.max)}
                                  stroke={color}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth="3"
                                />
                                <polyline
                                  className="chart-moving-line"
                                  fill="none"
                                  points={getSvgPoints(avgValues, typologyBounds.min, typologyBounds.max)}
                                  stroke={color}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth="2"
                                />
                                {coordinates.map(({ x, y }, index) => (
                                  <circle
                                    className="chart-point"
                                    cx={x}
                                    cy={y}
                                    fill={color}
                                    key={`${field}-${typologyRows[index]?.Mes}`}
                                    onFocus={() =>
                                      setTypologyTooltip({
                                        color,
                                        label,
                                        month: typologyRows[index]?.Mes,
                                        value: values[index],
                                        x,
                                        y,
                                      })
                                    }
                                    onMouseEnter={() =>
                                      setTypologyTooltip({
                                        color,
                                        label,
                                        month: typologyRows[index]?.Mes,
                                        value: values[index],
                                        x,
                                        y,
                                      })
                                    }
                                    r="5"
                                    tabIndex="0"
                                  />
                                ))}
                              </g>
                            )
                          })}
                        </svg>
                        {typologyTooltip && (
                          <div
                            className="chart-tooltip"
                            style={{
                              '--tooltip-color': typologyTooltip.color,
                              left: `${(typologyTooltip.x / 640) * 100}%`,
                              top: `${(typologyTooltip.y / 220) * 100}%`,
                            }}
                          >
                            <strong>{typologyTooltip.label}</strong>
                            <span>{typologyTooltip.month}</span>
                            <span>{formatMoney(typologyTooltip.value)}</span>
                          </div>
                        )}
                      </div>
                      <div className="typology-latest">
                        {selectedTypologyFields.map(({ field, label, color }) => (
                          <article key={field} style={{ '--legend-color': color }}>
                            <span>{label}</span>
                            <strong>{formatMoney(typologyRows[typologyRows.length - 1]?.[field])}</strong>
                          </article>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty-state compact-empty">
                      <strong>Sin tipologías seleccionadas</strong>
                      <span>Activa al menos una serie para ver la gráfica.</span>
                    </div>
                  )}
                </div>

                <div className="typology-predictions">
                  <h3 className="table-title">Predicción a 6 meses</h3>
                  <div className="prediction-grid">
                    {selectedTypologyFields.map(({ field, label, color }) => {
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
