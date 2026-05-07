import { useMemo, useState } from 'react'

const API_URL = 'http://localhost:8000'

const initialParams = {
  fecha_inicio: '2024-10-01',
  porcentaje_gasto: 0.3,
  porcentaje_inversion: 0.1,
  porcentaje_vacaciones: 0.05,
}

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

export function App() {
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(initialParams)
  const [health, setHealth] = useState('pendiente')
  const [isChecking, setIsChecking] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const latest = result?.historial?.ultimo_mes
  const rows = useMemo(() => result?.historial?.resumen ?? [], [result])

  const checkHealth = async () => {
    setIsChecking(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/health`)
      if (!response.ok) throw new Error('Backend no disponible')
      const data = await response.json()
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

    const query = new URLSearchParams({
      fecha_inicio: params.fecha_inicio,
      porcentaje_gasto: String(params.porcentaje_gasto),
      porcentaje_inversion: String(params.porcentaje_inversion),
      porcentaje_vacaciones: String(params.porcentaje_vacaciones),
    })

    try {
      const response = await fetch(`${API_URL}/api/v1/process?${query.toString()}`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'No se pudo procesar el Excel')
      setResult(data.result)
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

          <button className="primary-button" type="submit" disabled={isProcessing}>
            {isProcessing ? 'Procesando...' : 'Procesar'}
          </button>

          {error && <p className="error-message">{error}</p>}
        </form>

        <section className="panel result-panel">
          <div className="panel-header">
            <h2>Resumen</h2>
            <span>{latest?.Mes ?? 'Pendiente'}</span>
          </div>

          {latest ? (
            <>
              <div className="metrics-grid">
                {moneyFields.slice(0, 6).map((field) => (
                  <article className="metric" key={field}>
                    <span>{field}</span>
                    <strong>{formatMoney(latest[field])}</strong>
                  </article>
                ))}
              </div>

              <div className="movement-strip">
                <span>{result.movimientos.gastos} gastos</span>
                <span>{result.movimientos.ingresos} ingresos</span>
                <span>{result.movimientos.transferencias} transferencias</span>
                <span>{result.movimientos.cuentas} cuentas</span>
              </div>

              <div className="table-wrap">
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
                    {rows.slice(-8).map((row) => (
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
