import { useEffect, useMemo, useState } from 'react'
import { requestJson } from './api'
import { downloadJson, downloadText, toCsv } from './exporters'

const initialParams = {
  fecha_inicio: '2024-10-01',
  porcentaje_gasto: 0.3,
  porcentaje_inversion: 0.1,
  porcentaje_vacaciones: 0.05,
}

const createObjective = () => ({
  id: crypto.randomUUID(),
  nombre: '',
  etiquetas: '',
  fraccion_presupuesto: 0.1,
  duracion_meses: 1,
  mes_inicio: '2024-10',
  saldo_inicial: 0,
})

const moneyFields = [
  'total',
  '💰 Ahorros',
  '💳 Gasto del mes',
  '💸 Presupuesto Mes',
  '🧾 Presupuesto Disponible',
  '📈 Inversiones',
  'Dinero Invertido',
  '💼 Vacaciones',
  '🎁 Regalos',
]

function formatMoney(value) {
  if (typeof value !== 'number') return '-'
  return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(value)
}

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`
}

function formatDelta(value) {
  if (typeof value !== 'number') return '-'
  const sign = value > 0 ? '+' : ''
  return `${sign}${formatMoney(value)}`
}

function objectiveToPayload({ id, etiquetas, ...objective }) {
  return {
    ...objective,
    etiquetas: etiquetas
      .split(',')
      .map((tag) => tag.trim().toLowerCase())
      .filter(Boolean),
  }
}

function objectiveFromApi(objective) {
  return {
    id: crypto.randomUUID(),
    nombre: objective.nombre ?? '',
    etiquetas: Array.isArray(objective.etiquetas) ? objective.etiquetas.join(', ') : String(objective.etiquetas ?? ''),
    fraccion_presupuesto: Number(objective.fraccion_presupuesto ?? 0),
    duracion_meses: Number(objective.duracion_meses ?? 1),
    mes_inicio: String(objective.mes_inicio ?? '2024-10').slice(0, 7),
    saldo_inicial: Number(objective.saldo_inicial ?? 0),
  }
}

export function App() {
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(initialParams)
  const [objectives, setObjectives] = useState([])
  const [health, setHealth] = useState('pendiente')
  const [isChecking, setIsChecking] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSavingObjectives, setIsSavingObjectives] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [objectivesStatus, setObjectivesStatus] = useState('')
  const [selectedObjective, setSelectedObjective] = useState('all')
  const [selectedMonth, setSelectedMonth] = useState('')

  const latest = result?.historial?.ultimo_mes
  const rows = useMemo(() => result?.historial?.resumen ?? [], [result])
  const selectedRow = useMemo(
    () => rows.find((row) => row.Mes === selectedMonth) ?? latest,
    [latest, rows, selectedMonth],
  )
  const previousRow = useMemo(() => {
    if (!selectedRow) return null
    const index = rows.findIndex((row) => row.Mes === selectedRow.Mes)
    return index > 0 ? rows[index - 1] : null
  }, [rows, selectedRow])
  const comparisonRows = useMemo(
    () => [
      { label: 'Total', field: 'total' },
      { label: 'Ahorros', field: '💰 Ahorros' },
      { label: 'Gasto', field: '💳 Gasto del mes' },
      { label: 'Presupuesto', field: '💸 Presupuesto Mes' },
    ],
    [],
  )
  const recentRows = useMemo(() => rows.slice(-8), [rows])
  const trendMax = useMemo(
    () =>
      Math.max(
        1,
        ...recentRows.flatMap((row) => [
          Math.abs(Number(row.total) || 0),
          Math.abs(Number(row['💳 Gasto del mes']) || 0),
          Math.abs(Number(row['💸 Presupuesto Mes']) || 0),
        ]),
      ),
    [recentRows],
  )
  const objectiveRows = useMemo(() => result?.historial?.objetivos ?? [], [result])
  const objectiveNames = useMemo(
    () => Array.from(new Set(objectiveRows.map((row) => row.Objetivo))).sort(),
    [objectiveRows],
  )
  const filteredObjectiveRows = useMemo(
    () =>
      selectedObjective === 'all'
        ? objectiveRows
        : objectiveRows.filter((row) => row.Objetivo === selectedObjective),
    [objectiveRows, selectedObjective],
  )
  const objectiveTotals = useMemo(
    () =>
      filteredObjectiveRows.reduce(
        (totals, row) => ({
          aporte: totals.aporte + (Number(row.aporte_mes) || 0),
          gasto: totals.gasto + (Number(row.gastos_etiquetados_mes) || 0),
          liquidacion: totals.liquidacion + (Number(row.liquidacion) || 0),
        }),
        { aporte: 0, gasto: 0, liquidacion: 0 },
      ),
    [filteredObjectiveRows],
  )

  useEffect(() => {
    loadObjectives({ silent: true })
  }, [])

  const checkHealth = async () => {
    setIsChecking(true)
    setError('')
    try {
      const data = await requestJson('/health', {}, 'Backend no disponible')
      setHealth(data.status)
    } catch (err) {
      setHealth('error')
      setError(err.message)
    } finally {
      setIsChecking(false)
    }
  }

  const processWorkbook = async (event) => {
    event.preventDefault()
    if (!file) {
      setError('Selecciona un Excel antes de procesar.')
      return
    }

    setIsProcessing(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    const objetivos = objectives
      .filter((objective) => objective.nombre.trim())
      .map(objectiveToPayload)

    if (objetivos.length > 0) {
      formData.append('objetivos_json', JSON.stringify(objetivos))
    }

    const query = new URLSearchParams({
      fecha_inicio: params.fecha_inicio,
      porcentaje_gasto: String(params.porcentaje_gasto),
      porcentaje_inversion: String(params.porcentaje_inversion),
      porcentaje_vacaciones: String(params.porcentaje_vacaciones),
    })

    try {
      const data = await requestJson(`/api/v1/process?${query.toString()}`, {
        method: 'POST',
        body: formData,
      }, 'No se pudo procesar el Excel')
      setResult(data.result)
      setSelectedObjective('all')
      setSelectedMonth(data.result?.historial?.ultimo_mes?.Mes ?? '')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const updateParam = (key, value) => {
    setParams((current) => ({
      ...current,
      [key]: key === 'fecha_inicio' ? value : Number(value),
    }))
  }

  const addObjective = () => {
    setObjectives((current) => [...current, createObjective()])
  }

  const updateObjective = (id, key, value) => {
    setObjectives((current) =>
      current.map((objective) =>
        objective.id === id
          ? {
              ...objective,
              [key]: ['fraccion_presupuesto', 'duracion_meses', 'saldo_inicial'].includes(key) ? Number(value) : value,
            }
          : objective,
      ),
    )
  }

  const removeObjective = (id) => {
    setObjectives((current) => current.filter((objective) => objective.id !== id))
  }

  const loadObjectives = async ({ silent = false } = {}) => {
    if (!silent) setObjectivesStatus('Cargando...')
    try {
      const data = await requestJson('/api/v1/objectives', {}, 'No se pudieron cargar los objetivos')
      setObjectives((data.objetivos ?? []).map(objectiveFromApi))
      setObjectivesStatus(`${data.objetivos?.length ?? 0} objetivos cargados`)
    } catch (err) {
      setObjectivesStatus(silent ? '' : err.message)
    }
  }

  const saveObjectives = async () => {
    setIsSavingObjectives(true)
    setError('')
    setObjectivesStatus('Guardando...')

    try {
      const payload = {
        objetivos: objectives.filter((objective) => objective.nombre.trim()).map(objectiveToPayload),
      }
      const data = await requestJson('/api/v1/objectives', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }, 'No se pudieron guardar los objetivos')
      setObjectives((data.objetivos ?? []).map(objectiveFromApi))
      setObjectivesStatus(`${data.objetivos?.length ?? 0} objetivos guardados`)
    } catch (err) {
      setError(err.message)
      setObjectivesStatus('')
    } finally {
      setIsSavingObjectives(false)
    }
  }

  const exportResult = () => {
    if (!result) return
    const month = selectedRow?.Mes ?? latest?.Mes ?? 'resultado'
    downloadJson(`gestiones-${month}.json`, {
      exported_at: new Date().toISOString(),
      source_file: file?.name ?? null,
      params,
      objectives: objectives.map(objectiveToPayload),
      result,
    })
  }

  const exportHistoryCsv = () => {
    if (!rows.length) return
    const month = selectedRow?.Mes ?? latest?.Mes ?? 'resultado'
    downloadText(`gestiones-historial-${month}.csv`, toCsv(rows), 'text/csv;charset=utf-8')
  }

  const exportObjectivesCsv = () => {
    if (!filteredObjectiveRows.length) return
    const label =
      selectedObjective === 'all'
        ? 'objetivos'
        : selectedObjective
            .toLowerCase()
            .replaceAll(' ', '-')
            .replaceAll('/', '-')
    downloadText(`gestiones-${label}.csv`, toCsv(filteredObjectiveRows), 'text/csv;charset=utf-8')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Gestiones</h1>
          <p>Procesado de finanzas personales</p>
        </div>
        <button className="ghost-button" type="button" onClick={checkHealth} disabled={isChecking}>
          {isChecking ? 'Comprobando...' : `Backend: ${health}`}
        </button>
      </header>

      <section className="workspace">
        <form className="panel" onSubmit={processWorkbook}>
          <div className="panel-header">
            <h2>Entrada</h2>
            <span>{file ? file.name : 'Sin archivo'}</span>
          </div>

          <label className="file-input">
            <input
              type="file"
              accept=".xlsx,.xlsm,.xls"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <span>Seleccionar Excel</span>
          </label>

          <div className="field-grid">
            <label>
              Fecha inicio
              <input
                type="date"
                value={params.fecha_inicio}
                onChange={(event) => updateParam('fecha_inicio', event.target.value)}
              />
            </label>
            <label>
              Gasto
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={params.porcentaje_gasto}
                onChange={(event) => updateParam('porcentaje_gasto', event.target.value)}
              />
              <strong>{formatPercent(params.porcentaje_gasto)}</strong>
            </label>
            <label>
              Inversión
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={params.porcentaje_inversion}
                onChange={(event) => updateParam('porcentaje_inversion', event.target.value)}
              />
              <strong>{formatPercent(params.porcentaje_inversion)}</strong>
            </label>
            <label>
              Vacaciones
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={params.porcentaje_vacaciones}
                onChange={(event) => updateParam('porcentaje_vacaciones', event.target.value)}
              />
              <strong>{formatPercent(params.porcentaje_vacaciones)}</strong>
            </label>
          </div>

          <div className="objectives-section">
            <div className="section-heading">
              <h3>Objetivos</h3>
              <div className="button-row">
                <button className="text-button" type="button" onClick={() => loadObjectives()}>
                  Cargar
                </button>
                <button className="text-button" type="button" onClick={saveObjectives} disabled={isSavingObjectives}>
                  {isSavingObjectives ? 'Guardando...' : 'Guardar'}
                </button>
                <button className="text-button" type="button" onClick={addObjective}>
                  Añadir
                </button>
              </div>
            </div>

            {objectives.length === 0 ? (
              <p className="muted-text">Sin objetivos configurados para este cálculo.</p>
            ) : (
              <div className="objective-list">
                {objectives.map((objective) => (
                  <div className="objective-row" key={objective.id}>
                    <label>
                      Nombre
                      <input
                        type="text"
                        value={objective.nombre}
                        onChange={(event) => updateObjective(objective.id, 'nombre', event.target.value)}
                      />
                    </label>
                    <label>
                      Etiquetas
                      <input
                        type="text"
                        value={objective.etiquetas}
                        onChange={(event) => updateObjective(objective.id, 'etiquetas', event.target.value)}
                      />
                    </label>
                    <label>
                      Fracción
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={objective.fraccion_presupuesto}
                        onChange={(event) =>
                          updateObjective(objective.id, 'fraccion_presupuesto', event.target.value)
                        }
                      />
                    </label>
                    <label>
                      Meses
                      <input
                        type="number"
                        min="1"
                        step="1"
                        value={objective.duracion_meses}
                        onChange={(event) => updateObjective(objective.id, 'duracion_meses', event.target.value)}
                      />
                    </label>
                    <label>
                      Inicio
                      <input
                        type="month"
                        value={objective.mes_inicio}
                        onChange={(event) => updateObjective(objective.id, 'mes_inicio', event.target.value)}
                      />
                    </label>
                    <label>
                      Saldo inicial
                      <input
                        type="number"
                        step="0.01"
                        value={objective.saldo_inicial}
                        onChange={(event) => updateObjective(objective.id, 'saldo_inicial', event.target.value)}
                      />
                    </label>
                    <button className="text-button danger" type="button" onClick={() => removeObjective(objective.id)}>
                      Quitar
                    </button>
                  </div>
                ))}
              </div>
            )}
            {objectivesStatus && <p className="status-message">{objectivesStatus}</p>}
          </div>

          <button className="primary-button" type="submit" disabled={isProcessing}>
            {isProcessing ? 'Procesando...' : 'Procesar'}
          </button>

          {error && <p className="error-message">{error}</p>}
        </form>

        <section className="panel result-panel">
          <div className="panel-header">
            <h2>Resumen</h2>
            <div className="panel-actions">
              {rows.length > 0 ? (
                <label className="month-selector">
                  <span>Mes</span>
                  <select value={selectedRow?.Mes ?? ''} onChange={(event) => setSelectedMonth(event.target.value)}>
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
                  <button className="text-button" type="button" onClick={exportResult}>
                    JSON
                  </button>
                  <button className="text-button" type="button" onClick={exportHistoryCsv}>
                    Historial CSV
                  </button>
                  {objectiveRows.length > 0 && (
                    <button className="text-button" type="button" onClick={exportObjectivesCsv}>
                      Objetivos CSV
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>

          {selectedRow ? (
            <>
              <div className="metrics-grid">
                {moneyFields.slice(0, 6).map((field) => (
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
                          <span
                            style={{
                              width: `${Math.max(3, (Math.abs(Number(row['💳 Gasto del mes']) || 0) / trendMax) * 100)}%`,
                            }}
                          />
                        </div>
                        <div className="trend-bar presupuesto">
                          <span
                            style={{
                              width: `${Math.max(3, (Math.abs(Number(row['💸 Presupuesto Mes']) || 0) / trendMax) * 100)}%`,
                            }}
                          />
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
                      <select value={selectedObjective} onChange={(event) => setSelectedObjective(event.target.value)}>
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
      </section>
    </main>
  )
}
