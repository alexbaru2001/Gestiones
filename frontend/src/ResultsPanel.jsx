import { useState } from 'react'
import { formatDelta, formatMoney } from './formatters'
import { comparisonRows } from './resultSelectors'

const moneyFields = [
  'total',
  '💰 Ahorros',
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
  { field: '💼 Vacaciones', label: 'Vacaciones', color: '#d2a53f' },
  { field: '📈 Inversiones', label: 'Reserva inversión', color: '#5c6f9e' },
  { field: 'Fondo de reserva cargado', label: 'Fondo reserva', color: '#6c7a72' },
  { field: '📉 Deuda Presupuestaria acumulada', label: 'Deuda acumulada', color: '#8f4638' },
  { field: '💳 Gasto del mes', label: 'Gasto mensual', color: '#b85a4b' },
  { field: '🎁 Regalos', label: 'Regalos', color: '#a66a4d' },
]

const defaultTypologyFields = [
  '💰 Ahorros',
  '💼 Vacaciones',
  '📈 Inversiones',
  'Fondo de reserva cargado',
  '📉 Deuda Presupuestaria acumulada',
  '💳 Gasto del mes',
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
  const monthlyDebt = asNumber(selectedRow['📉 Deuda Presupuestaria mensual'])
  const accumulatedDebt = asNumber(selectedRow['📉 Deuda Presupuestaria acumulada'])
  const committed = Math.max(0, monthBudget - availableBudget)
  const remaining = Math.max(0, availableBudget - spent)
  const overrun = monthlyDebt || Math.max(0, spent - availableBudget)
  const totalForBar = Math.max(monthBudget, committed + spent + remaining, 1)
  const execution = availableBudget > 0 ? (spent / availableBudget) * 100 : 0

  return {
    accumulatedDebt,
    availableBudget,
    committed,
    execution,
    monthBudget,
    monthlyDebt,
    overrun,
    remaining,
    spent,
    totalForBar,
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

function movingAverage(values, windowSize = 3) {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1)
    const chunk = values.slice(start, index + 1)
    return chunk.reduce((total, value) => total + value, 0) / chunk.length
  })
}

function getSvgPoints(values, min, max) {
  return getSvgCoordinates(values, min, max)
    .map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')
}

function getSvgCoordinates(values, min, max) {
  const width = 640
  const height = 220
  const padding = 18
  const range = max - min || 1
  return values.map((value, index) => ({
    x: padding + (index / Math.max(1, values.length - 1)) * (width - padding * 2),
    y: height - padding - ((value - min) / range) * (height - padding * 2),
  }))
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
  const [typologyTooltip, setTypologyTooltip] = useState(null)
  const budget = selectedRow ? getBudget(selectedRow) : null
  const expenseAnalysis = getExpenseAnalysis(result)
  const expenseMax = Math.max(...expenseAnalysis.totales_categoria.map((row) => asNumber(row.total)), 1)
  const savingsAnalysis = getSavingsAnalysis(result)
  const savingsMax = Math.max(...savingsAnalysis.mensual.map((row) => Math.abs(asNumber(row.balance))), 1)
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
  const typologyRows = rows.slice(-12)
  const selectedTypologyFields = typologyFields.filter(({ field }) => activeTypologyFields.includes(field))
  const typologyBounds = getTypologyBounds(typologyRows, selectedTypologyFields)
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
                <div className="metrics-grid">
                  {moneyFields.map((field) => (
                    <article className="metric" key={field}>
                      <span>{field}</span>
                      <strong>{formatMoney(selectedRow[field])}</strong>
                    </article>
                  ))}
                </div>

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
                    {budget.committed > 0 && (
                      <span
                        className="budget-segment committed"
                        style={{ width: getSegmentWidth(budget.committed, budget.totalForBar) }}
                        title={`Compromisos: ${formatMoney(budget.committed)}`}
                      />
                    )}
                    {budget.spent > 0 && (
                      <span
                        className="budget-segment spent"
                        style={{ width: getSegmentWidth(budget.spent, budget.totalForBar) }}
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
                  <div className="budget-legend">
                    <span>Compromisos</span>
                    <span>Gasto</span>
                    <span>Restante</span>
                  </div>
                </div>

                <div className="budget-summary">
                  <article>
                    <span>Presupuesto mes</span>
                    <strong>{formatMoney(budget.monthBudget)}</strong>
                  </article>
                  <article>
                    <span>Disponible</span>
                    <strong>{formatMoney(budget.availableBudget)}</strong>
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
              expenseAnalysis.totales_categoria.length > 0 ? (
                <section className="expenses-layout">
                  <div className="expenses-panel">
                    <div className="table-toolbar compact-toolbar">
                      <h3 className="table-title">Gastos por categoría</h3>
                      <span>{expenseAnalysis.ultimo_mes?.Mes ?? selectedRow.Mes}</span>
                    </div>
                    <div className="category-list">
                      {expenseAnalysis.totales_categoria.map((row) => (
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

                  <div className="table-wrap compact-table-wrap">
                    <h3 className="table-title">Evolución mensual</h3>
                    <table>
                      <thead>
                        <tr>
                          <th>Mes</th>
                          <th>Ingresos</th>
                          <th>Gastos</th>
                          <th>Balance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {expenseAnalysis.mensual.slice(-12).map((row) => (
                          <tr key={row.Mes}>
                            <td>{row.Mes}</td>
                            <td>{formatMoney(row.ingresos)}</td>
                            <td>{formatMoney(row.gastos)}</td>
                            <td>{formatMoney(row.balance)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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
                    <span>Últimos {typologyRows.length} meses</span>
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
                      const prediction = predictTypology(rows, field)
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
