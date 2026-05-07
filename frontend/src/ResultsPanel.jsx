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
  return (
    <section className="panel result-panel" aria-busy={isProcessing}>
      <div className="panel-header">
        <h2>Resumen</h2>
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
                const delta = (Number(selectedRow[field]) || 0) - (Number(previousRow[field]) || 0)
                return (
                  <article className={delta >= 0 ? 'comparison-item positive' : 'comparison-item negative'} key={field}>
                    <span>{label} vs {previousRow.Mes}</span>
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
                      <span style={{ width: `${Math.max(3, (Math.abs(Number(row.total) || 0) / trendMax) * 100)}%` }} />
                    </div>
                    <div className="trend-bar gasto">
                      <span style={{ width: `${Math.max(3, (Math.abs(Number(row['💳 Gasto del mes']) || 0) / trendMax) * 100)}%` }} />
                    </div>
                    <div className="trend-bar presupuesto">
                      <span style={{ width: `${Math.max(3, (Math.abs(Number(row['💸 Presupuesto Mes']) || 0) / trendMax) * 100)}%` }} />
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

          <div className="table-wrap">
            <h3 className="table-title">Historial reciente</h3>
            <table>
              <thead>
                <tr>
                  <th>Mes</th>
                  <th>Total</th>
                  <th>Ahorros</th>
                  <th>Gasto</th>
                  <th>Presupuesto</th>
                </tr>
              </thead>
              <tbody>
                {recentRows.map((row) => (
                  <tr key={row.Mes}>
                    <td>{row.Mes}</td>
                    <td>{formatMoney(row.total)}</td>
                    <td>{formatMoney(row['💰 Ahorros'])}</td>
                    <td>{formatMoney(row['💳 Gasto del mes'])}</td>
                    <td>{formatMoney(row['💸 Presupuesto Mes'])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {objectiveRows.length > 0 && (
            <div className="table-wrap">
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
          )}
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
