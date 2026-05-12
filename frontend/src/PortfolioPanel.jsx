import { useEffect, useMemo, useRef, useState } from 'react'
import { requestJson } from './api'

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 's/d'
  return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(Number(value))
}

function formatNumber(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 's/d'
  const text = new Intl.NumberFormat('es-ES', { maximumFractionDigits: 2 }).format(Number(value))
  return suffix ? `${text} ${suffix}` : text
}

function getWeight(value, total) {
  if (!total) return '0%'
  return `${Math.max(0, Math.min(100, (Number(value) / total) * 100))}%`
}

const chartModes = [
  { id: 'position', label: 'Valores' },
  { id: 'region', label: 'Continente' },
  { id: 'sector', label: 'Sector' },
  { id: 'focus', label: 'Foco' },
  { id: 'broker', label: 'Broker' },
  { id: 'asset_type', label: 'Tipo' },
]

export function PortfolioPanel() {
  const fileInputRef = useRef(null)
  const [snapshot, setSnapshot] = useState(null)
  const [files, setFiles] = useState([])
  const [snapshots, setSnapshots] = useState([])
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [isSummaryOpen, setIsSummaryOpen] = useState(false)
  const [chartMode, setChartMode] = useState('position')
  const [selectedSnapshotMonth, setSelectedSnapshotMonth] = useState('')

  const snapshotOptions = useMemo(() => buildSnapshotOptions(snapshot, snapshots), [snapshot, snapshots])
  const selectedSnapshot = useMemo(
    () => snapshotOptions.find((item) => item.snapshot_month === selectedSnapshotMonth) ?? snapshot ?? snapshotOptions.at(-1) ?? null,
    [selectedSnapshotMonth, snapshot, snapshotOptions],
  )
  const positions = selectedSnapshot?.positions ?? []
  const brokers = selectedSnapshot?.brokers ?? []
  const summary = selectedSnapshot?.summary
  const investmentPositions = useMemo(() => positions.filter((position) => position.asset_type !== 'cash'), [positions])
  const investedTotal = useMemo(
    () => investmentPositions.reduce((total, position) => total + Number(position.current_value ?? 0), 0),
    [investmentPositions],
  )
  const costTotal = Number(summary?.known_cost ?? 0)
  const chartItems = useMemo(() => buildPortfolioChart(investmentPositions, chartMode), [chartMode, investmentPositions])
  const evolutionRows = useMemo(() => buildEvolutionRows(snapshots), [snapshots])
  const evolutionMax = useMemo(
    () => Math.max(...evolutionRows.flatMap((row) => [row.invested, row.cost]), 0),
    [evolutionRows],
  )
  const donutBackground = useMemo(() => buildDonutBackground(chartItems), [chartItems])

  useEffect(() => {
    loadPortfolio()
    loadSnapshots()
  }, [])

  const loadPortfolio = async () => {
    setIsLoading(true)
    setError('')
    try {
      const data = await requestJson('/api/v1/portfolio', {}, 'No se pudo cargar la cartera')
      setSnapshot(data.result)
      setStatus(data.result ? 'Cartera local cargada' : 'Sin cartera local importada')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const loadSnapshots = async () => {
    try {
      const data = await requestJson('/api/v1/portfolio/snapshots', {}, 'No se pudo cargar el histórico de cartera')
      const loadedSnapshots = data.result ?? []
      setSnapshots(loadedSnapshots)
      setSelectedSnapshotMonth((current) => current || loadedSnapshots.at(-1)?.snapshot_month || '')
    } catch (err) {
      setError(err.message)
    }
  }

  const importPortfolio = async (event) => {
    event.preventDefault()
    if (!files.length) {
      setError('Selecciona los extractos PDF/XLSX de cartera.')
      return
    }
    setIsImporting(true)
    setError('')
    setStatus('Importando...')

    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))

    try {
      const data = await requestJson(
        '/api/v1/portfolio/import',
        {
          method: 'POST',
          body: formData,
        },
        'No se pudo importar la cartera',
      )
      setSnapshot(data.result)
      const loadedSnapshots = data.snapshots ?? []
      setSnapshots(loadedSnapshots)
      setSelectedSnapshotMonth(data.result?.snapshot_month || loadedSnapshots.at(-1)?.snapshot_month || '')
      setStatus(`${files.length} documentos importados y guardados en local`)
      setFiles([])
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (err) {
      setError(err.message)
      setStatus('')
    } finally {
      setIsImporting(false)
    }
  }

  return (
    <section className="panel portfolio-panel">
      <div className="panel-header">
        <div>
          <h2>Cartera</h2>
          <span>Importación local de brokers, posiciones y rentabilidad</span>
        </div>
      </div>

      <form className="portfolio-import" onSubmit={importPortfolio}>
        <label className="file-input compact-file-input">
          <input
            accept=".pdf,.xlsx"
            multiple
            onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
            ref={fileInputRef}
            type="file"
          />
          <span>{files.length ? `${files.length} documentos seleccionados` : 'Seleccionar PDFs/XLSX de cartera'}</span>
        </label>
        <button className="primary-button inline-primary" type="submit" disabled={isImporting}>
          {isImporting ? 'Importando...' : 'Importar cartera'}
        </button>
        <button className="ghost-button" type="button" onClick={loadPortfolio} disabled={isLoading}>
          {isLoading ? 'Cargando...' : 'Recargar local'}
        </button>
      </form>

      {status ? <p className="status-message">{status}</p> : null}
      {error ? <p className="error-message">{error}</p> : null}

      {!selectedSnapshot ? (
        <div className="empty-state compact-empty">
          <strong>Sin cartera local</strong>
          <span>Importa los extractos de Trade Republic, MyInvestor y DeGiro para generar el resumen.</span>
        </div>
      ) : (
        <div className="portfolio-dashboard">
          <section className="portfolio-card portfolio-snapshot-card">
            <div>
              <h3>Snapshot seleccionado</h3>
              <span className="muted-text">
                {selectedSnapshot.snapshot_date ?? 'Sin fecha de referencia'} · {selectedSnapshot.positions?.length ?? 0} posiciones ·{' '}
                {selectedSnapshot.transactions?.length ?? 0} movimientos
              </span>
            </div>
            <label>
              Mes
              <select value={selectedSnapshot.snapshot_month ?? ''} onChange={(event) => setSelectedSnapshotMonth(event.target.value)}>
                {snapshotOptions.map((item) => (
                  <option key={item.snapshot_month ?? 'actual'} value={item.snapshot_month ?? ''}>
                    {item.snapshot_month ?? 'Sin fecha'} · {formatMoney(item.summary?.known_cost)} invertido
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="portfolio-hero">
            <article>
              <span>Valor actual invertido</span>
              <strong>{formatMoney(investedTotal)}</strong>
            </article>
            <article>
              <span>Dinero invertido</span>
              <strong>{formatMoney(costTotal)}</strong>
              <small>Coste histórico conocido</small>
            </article>
            <article>
              <span>Rentabilidad conocida</span>
              <strong>{formatMoney(summary?.known_unrealized_gain)}</strong>
              <small>{formatNumber(summary?.known_unrealized_gain_pct, '%')}</small>
            </article>
            <article>
              <span>Dividendos netos</span>
              <strong>{formatMoney(summary?.dividends)}</strong>
            </article>
          </section>

          {selectedSnapshot.warnings?.length ? (
            <section className="warning-panel">
              <strong>Avisos de cálculo</strong>
              {selectedSnapshot.warnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </section>
          ) : null}

          <section className="portfolio-card">
            <div className="portfolio-chart-header">
              <h3>Distribución dinámica</h3>
              <div className="portfolio-chart-tabs" role="tablist" aria-label="Vista de distribución de cartera">
                {chartModes.map((mode) => (
                  <button
                    className={chartMode === mode.id ? 'active' : ''}
                    key={mode.id}
                    type="button"
                    onClick={() => setChartMode(mode.id)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="portfolio-structure">
              <div
                className="portfolio-donut"
                aria-label="Estructura porcentual de cartera"
                style={{ background: donutBackground }}
              >
                <strong>{formatMoney(investedTotal)}</strong>
                <span>valor actual</span>
              </div>
              <div className="portfolio-stacked-bar" aria-label="Distribución porcentual de cartera invertida">
                {chartItems.map((item, index) => (
                  <span
                    key={item.label}
                    style={{
                      background: getChartColor(index),
                      width: getWeight(item.value, investedTotal),
                    }}
                    title={`${item.label}: ${formatMoney(item.value)} · ${formatNumber(item.weight, '%')}`}
                  />
                ))}
              </div>
            </div>
            <div className="portfolio-distribution">
              {chartItems.map((item, index) => (
                <article key={item.label}>
                  <div>
                    <strong>
                      <i style={{ background: getChartColor(index) }} />
                      {item.label}
                    </strong>
                    <span>{formatMoney(item.value)}</span>
                  </div>
                  <span>{formatNumber(item.weight, '%')}</span>
                  <div className="portfolio-bar">
                    <span style={{ background: getChartColor(index), width: getWeight(item.value, investedTotal) }} />
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="portfolio-card">
            <button className="portfolio-summary-toggle" type="button" onClick={() => setIsSummaryOpen((current) => !current)}>
              {isSummaryOpen ? 'Ocultar resumen' : 'Mostrar resumen'}
            </button>
            {isSummaryOpen ? (
              <>
                <div className="portfolio-summary">
                  <article>
                    <span>Total local visible</span>
                    <strong>{formatMoney(summary?.total_value)}</strong>
                  </article>
                  <article>
                    <span>Valor actual invertido</span>
                    <strong>{formatMoney(summary?.invested)}</strong>
                  </article>
                  <article>
                    <span>Dinero invertido</span>
                    <strong>{formatMoney(summary?.known_cost)}</strong>
                  </article>
                  <article>
                    <span>Efectivo no principal</span>
                    <strong>{formatMoney(summary?.cash)}</strong>
                  </article>
                  <article>
                    <span>Rentabilidad conocida</span>
                    <strong>{formatMoney(summary?.known_unrealized_gain)}</strong>
                    <small>{formatNumber(summary?.known_unrealized_gain_pct, '%')}</small>
                  </article>
                  <article>
                    <span>Dividendos netos</span>
                    <strong>{formatMoney(summary?.dividends)}</strong>
                  </article>
                </div>

                <div className="broker-grid">
                  {brokers.map((broker) => (
                    <article key={broker.broker}>
                      <strong>{broker.broker}</strong>
                      <span>Total: {formatMoney(broker.total_value)}</span>
                      <span>Valor actual invertido: {formatMoney(broker.invested)}</span>
                      <span>Dinero invertido: {formatMoney(broker.known_cost)}</span>
                      <span>Efectivo: {formatMoney(broker.cash)}</span>
                      <span>Rentabilidad conocida: {formatMoney(broker.known_unrealized_gain)}</span>
                      <span>Dividendos: {formatMoney(broker.dividends)}</span>
                    </article>
                  ))}
                </div>
              </>
            ) : null}
          </section>

          <section className="portfolio-card">
            <h3>Posiciones</h3>
            <div className="portfolio-position-grid">
              {investmentPositions.map((position) => (
                <article key={`${position.broker}-${position.isin ?? position.name}-${position.asset_type}`}>
                  <div>
                    <strong>{position.ticker ?? position.name}</strong>
                    <span>{position.broker} · {position.region ?? 'Sin región'} · {position.sector ?? 'Sin sector'}</span>
                  </div>
                  <strong>{formatMoney(position.current_value)}</strong>
                  <div className="portfolio-bar">
                    <span style={{ width: getWeight(position.current_value, investedTotal) }} />
                  </div>
                  <span>{formatNumber((Number(position.current_value ?? 0) / investedTotal) * 100, '%')}</span>
                </article>
              ))}
            </div>
          </section>

          <section className="portfolio-card">
            <h3>Posiciones</h3>
            <div className="table-wrap compact-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Broker</th>
                    <th>Activo</th>
                    <th>Tipo</th>
                    <th>Cantidad</th>
                    <th>Valor</th>
                    <th>Coste</th>
                    <th>P/L</th>
                    <th>Sector</th>
                    <th>Región</th>
                    <th>Horizonte</th>
                  </tr>
                </thead>
                <tbody>
                  {investmentPositions.map((position) => (
                    <tr key={`${position.broker}-${position.isin ?? position.name}-${position.asset_type}`}>
                      <td>{position.broker}</td>
                      <td>{position.ticker ?? position.name}</td>
                      <td>{formatAssetType(position.asset_type)}</td>
                      <td>{formatNumber(position.quantity)}</td>
                      <td>{formatMoney(position.current_value)}</td>
                      <td>{formatMoney(position.cost)}</td>
                      <td>{formatMoney(position.unrealized_gain)}</td>
                      <td>{position.sector ?? 's/d'}</td>
                      <td>{position.region ?? 's/d'}</td>
                      <td>{position.horizon}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="portfolio-card">
            <div className="portfolio-chart-header">
              <h3>Evolución mensual</h3>
              <span className="muted-text">{evolutionRows.length} snapshots locales</span>
            </div>
            {evolutionRows.length ? (
              <div className="portfolio-evolution">
                {evolutionRows.map((row) => (
                  <article className={row.month === selectedSnapshot.snapshot_month ? 'active' : ''} key={row.month}>
                    <span>{row.month}</span>
                    <div className="portfolio-evolution-bars">
                      <div>
                        <small>Dinero</small>
                        <div className="portfolio-bar portfolio-cost-bar">
                          <span style={{ width: getWeight(row.cost, evolutionMax) }} />
                        </div>
                      </div>
                      <div>
                        <small>Valor</small>
                        <div className="portfolio-bar">
                          <span style={{ width: getWeight(row.invested, evolutionMax) }} />
                        </div>
                      </div>
                    </div>
                    <strong>{formatMoney(row.invested)}</strong>
                    <small>{formatMoney(row.cost)} invertido · {formatMoney(row.gain)} P/L</small>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted-text">Aún no hay snapshots mensuales guardados.</p>
            )}
          </section>

          <section className="portfolio-card">
            <h3>Archivos importados</h3>
            <div className="imported-files">
              {(selectedSnapshot.files ?? []).map((file) => (
                <article key={file.filename}>
                  <strong>{file.filename}</strong>
                  <span>{file.kind}</span>
                  <span>{file.positions} posiciones · {file.transactions} movimientos útiles</span>
                </article>
              ))}
            </div>
          </section>
        </div>
      )}
    </section>
  )
}

function buildPortfolioChart(positions, mode) {
  const total = positions.reduce((acc, position) => acc + Number(position.current_value ?? 0), 0)
  if (!positions.length || !total) return []
  const groups = positions.reduce((acc, position) => {
    const label = getChartLabel(position, mode)
    acc[label] = (acc[label] ?? 0) + Number(position.current_value ?? 0)
    return acc
  }, {})
  return Object.entries(groups)
    .map(([label, value]) => ({ label, value, weight: (value / total) * 100 }))
    .sort((left, right) => right.value - left.value)
}

function getChartLabel(position, mode) {
  if (mode === 'position') return position.ticker ?? position.name
  if (mode === 'region') return position.region ?? 'Sin clasificar'
  if (mode === 'sector') return position.sector ?? 'Sin clasificar'
  if (mode === 'focus') return position.focus ?? 'Sin clasificar'
  if (mode === 'broker') return position.broker
  if (mode === 'asset_type') return formatAssetType(position.asset_type)
  return position.name
}

function getChartColor(index) {
  const colors = ['#286b57', '#b85a4b', '#d2a53f', '#436a92', '#6c7a72', '#8a6d3b', '#4f8b6f', '#a55769']
  return colors[index % colors.length]
}

function buildDonutBackground(items) {
  if (!items.length) return '#e8eee5'
  let start = 0
  const segments = items.map((item, index) => {
    const end = start + item.weight
    const segment = `${getChartColor(index)} ${start}% ${end}%`
    start = end
    return segment
  })
  return `conic-gradient(${segments.join(', ')})`
}

function buildSnapshotOptions(snapshot, snapshots) {
  const byMonth = new Map()
  ;(snapshots ?? []).forEach((item) => {
    if (item?.snapshot_month) byMonth.set(item.snapshot_month, item)
  })
  if (snapshot?.snapshot_month) byMonth.set(snapshot.snapshot_month, snapshot)
  return Array.from(byMonth.values()).sort((left, right) =>
    (left.snapshot_month ?? '').localeCompare(right.snapshot_month ?? ''),
  )
}

function buildEvolutionRows(snapshots) {
  return (snapshots ?? [])
    .filter((snapshot) => snapshot?.snapshot_month)
    .map((snapshot) => ({
      month: snapshot.snapshot_month,
      date: snapshot.snapshot_date,
      invested: Number(snapshot.summary?.invested ?? 0),
      cost: Number(snapshot.summary?.known_cost ?? 0),
      gain: Number(snapshot.summary?.known_unrealized_gain ?? 0),
      dividends: Number(snapshot.summary?.dividends ?? 0),
    }))
    .sort((left, right) => left.month.localeCompare(right.month))
}

function formatAssetType(assetType) {
  const labels = {
    cash: 'Efectivo',
    crypto: 'Cripto',
    fund: 'Fondo',
    stock: 'Acción',
  }
  return labels[assetType] ?? assetType
}
