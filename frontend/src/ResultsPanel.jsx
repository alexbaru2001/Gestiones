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
  '📈 Inversiones',
]

const tabs = [
  { id: 'resumen', label: 'Resumen' },
  { id: 'presupuesto', label: 'Presupuesto' },
  { id: 'gastos', label: 'Gastos' },
  { id: 'ahorro', label: 'Ahorro' },
  { id: 'inversiones', label: 'Inversiones' },
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
  const expenseAnalysis = getExpenseAnalysis(result)
  const expenseMax = Math.max(...expenseAnalysis.totales_categoria.map((row) => asNumber(row.total)), 1)
  const savingsAnalysis = getSavingsAnalysis(result)
  const savingsMax = Math.max(...savingsAnalysis.mensual.map((row) => Math.abs(asNumber(row.balance))), 1)
  const investmentMax = Math.max(
    ...rows.map((row) => Math.max(Math.abs(asNumber(row['📈 Inversiones'])), Math.abs(asNumber(row['Dinero Invertido'])))),
    1,
  )

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
