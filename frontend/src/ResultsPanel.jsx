import { useState } from 'react'
import { formatDelta, formatMoney } from './formatters'
import { comparisonRows } from './resultSelectors'

const moneyFields = [
  'total',
  '💰 Ahorros',
  '💳 Gasto del mes',
  '💸 Presupuesto Mes',
  '🧾 Presupuesto Disponible',
  '📈 Inversiones',
]

const tabs = [
  { id: 'resumen', label: 'Resumen' },
  { id: 'presupuesto', label: 'Presupuesto' },
  { id: 'objetivos', label: 'Objetivos' },
  { id: 'datos', label: 'Datos' },
]

const historyFields = [
  { label: 'Mes', field: 'Mes', type: 'text' },
  { label: 'Total', field: 'total', type: 'money' },
  { label: 'Ahorros', field: '💰 Ahorros', type: 'money' },
  { label: 'Gasto', field: '💳 Gasto del mes', type: 'money' },
  { label: 'Presupuesto', field: '💸 Presupuesto Mes', type: 'money' },
  { label: 'Disponible', field: '🧾 Presupuesto Disponible', type: 'money' },
  { label: 'Inversiones', field: '📈 Inversiones', type: 'money' },
]

function asNumber(value) {
  return Number(value) || 0
}

function getBudget(selectedRow) {
  const monthBudget = asNumber(selectedRow['💸 Presupuesto Mes'])
  const availableBudget = asNumber(selectedRow['🧾 Presupuesto Disponible']) || monthBudget
  const spent = asNumber(selectedRow['💳 Gasto del mes'])
  const committed = Math.max(0, monthBudget - availableBudget)
  const remaining = Math.max(0, availableBudget - spent)
  const overrun = Math.max(0, spent - availableBudget)
  const totalForBar = Math.max(monthBudget, committed + spent + remaining, 1)
  const execution = availableBudget > 0 ? (spent / availableBudget) * 100 : 0

  return {
    availableBudget,
    committed,
    execution,
    monthBudget,
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
  const budget = selectedRow ? getBudget(selectedRow) : null

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
