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

const defaultLifeContext = {
  homePrice: 400000,
  downPaymentPct: 30,
  ownSharePct: 60,
  maxMonthlyInvestment: '',
}

const lifeContextStorageKey = 'gestiones:portfolio-life-context'

export function PortfolioPanel({ financeRows = [] }) {
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
  const [lifeContext, setLifeContext] = useState(() => loadLifeContext())
  const [expandedPositionKey, setExpandedPositionKey] = useState('')
  const [positionAnalyses, setPositionAnalyses] = useState({})

  const financeByMonth = useMemo(() => buildFinanceByMonth(financeRows), [financeRows])
  const snapshotOptions = useMemo(() => buildSnapshotOptions(snapshot, snapshots, financeByMonth), [financeByMonth, snapshot, snapshots])
  const selectedSnapshot = useMemo(
    () => snapshotOptions.find((item) => item.snapshot_month === selectedSnapshotMonth) ?? snapshot ?? snapshotOptions.at(-1) ?? null,
    [selectedSnapshotMonth, snapshot, snapshotOptions],
  )
  const positions = selectedSnapshot?.positions ?? []
  const brokers = selectedSnapshot?.brokers ?? []
  const summary = selectedSnapshot?.summary
  const investmentPositions = useMemo(() => positions.filter((position) => position.asset_type !== 'cash'), [positions])
  const analyzableStockPositions = useMemo(
    () => investmentPositions.filter((position) => position.asset_type === 'stock' && position.ticker),
    [investmentPositions],
  )
  const investedTotal = useMemo(
    () => investmentPositions.reduce((total, position) => total + Number(position.current_value ?? 0), 0),
    [investmentPositions],
  )
  const financeInvested = summary?.finance_invested ?? null
  const costTotal = Number(financeInvested ?? summary?.known_cost ?? 0)
  const brokerCost = Number(summary?.known_cost ?? 0)
  const chartItems = useMemo(() => buildPortfolioChart(investmentPositions, chartMode), [chartMode, investmentPositions])
  const evolutionRows = useMemo(() => buildEvolutionRows(snapshots, financeByMonth), [financeByMonth, snapshots])
  const evolutionMax = useMemo(
    () => Math.max(...evolutionRows.flatMap((row) => [row.invested, row.cost]), 0),
    [evolutionRows],
  )
  const donutBackground = useMemo(() => buildDonutBackground(chartItems), [chartItems])
  const selectedFinance = useMemo(
    () => financeByMonth.get(selectedSnapshot?.snapshot_month) ?? getLatestFinance(financeByMonth),
    [financeByMonth, selectedSnapshot?.snapshot_month],
  )
  const globalAnalysis = useMemo(
    () => buildGlobalPortfolioAnalysis(investmentPositions, investedTotal, costTotal, selectedFinance, lifeContext),
    [costTotal, investedTotal, investmentPositions, lifeContext, selectedFinance],
  )

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

  const updateLifeContext = (key, value) => {
    const next = {
      ...lifeContext,
      [key]: key === 'maxMonthlyInvestment' ? value : Number(value),
    }
    setLifeContext(next)
    localStorage.setItem(lifeContextStorageKey, JSON.stringify(next))
  }

  const togglePositionAnalysis = async (position) => {
    const key = getPositionKey(position)
    const isOpening = expandedPositionKey !== key
    setExpandedPositionKey(isOpening ? key : '')
    if (!isOpening || !position.ticker || positionAnalyses[key]?.status === 'loaded' || positionAnalyses[key]?.status === 'loading') return

    setPositionAnalyses((current) => ({ ...current, [key]: { status: 'loading' } }))
    try {
      const data = await requestJson(
        `/api/v1/investments/analyze?ticker=${encodeURIComponent(position.ticker)}`,
        {},
        `No se pudo analizar ${position.ticker}`,
      )
      setPositionAnalyses((current) => ({ ...current, [key]: { status: 'loaded', result: data.result } }))
    } catch (err) {
      setPositionAnalyses((current) => ({ ...current, [key]: { status: 'error', error: err.message } }))
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
                    {item.snapshot_month ?? 'Sin fecha'} · {formatMoney(item.summary?.finance_invested ?? item.summary?.known_cost)} invertido
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
              <small>{financeInvested === null ? 'Coste broker detectado' : 'Según Finanzas'}</small>
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

          <section className="portfolio-card portfolio-global-card">
            <div className="portfolio-chart-header">
              <div>
                <h3>Análisis global</h3>
                <span className="muted-text">Cartera, liquidez y objetivo vivienda</span>
              </div>
            </div>
            <div className="life-context-grid">
              <label>
                Vivienda objetivo
                <input
                  min="0"
                  onChange={(event) => updateLifeContext('homePrice', event.target.value)}
                  step="1000"
                  type="number"
                  value={lifeContext.homePrice}
                />
              </label>
              <label>
                Entrada
                <input
                  min="0"
                  onChange={(event) => updateLifeContext('downPaymentPct', event.target.value)}
                  step="1"
                  type="number"
                  value={lifeContext.downPaymentPct}
                />
              </label>
              <label>
                Mi parte
                <input
                  min="0"
                  onChange={(event) => updateLifeContext('ownSharePct', event.target.value)}
                  step="1"
                  type="number"
                  value={lifeContext.ownSharePct}
                />
              </label>
              <label>
                Inversión mensual máx.
                <input
                  min="0"
                  onChange={(event) => updateLifeContext('maxMonthlyInvestment', event.target.value)}
                  placeholder={formatMoney(globalAnalysis.plannedMonthlyInvestment)}
                  step="50"
                  type="number"
                  value={lifeContext.maxMonthlyInvestment}
                />
              </label>
            </div>
            <div className="portfolio-analysis-grid">
              <article>
                <span>Objetivo entrada propia</span>
                <strong>{formatMoney(globalAnalysis.ownDownPaymentGoal)}</strong>
                <small>{formatMoney(globalAnalysis.downPaymentGoal)} entrada total</small>
              </article>
              <article>
                <span>Liquidez visible</span>
                <strong>{formatMoney(globalAnalysis.visibleLiquidity)}</strong>
                <small>{formatNumber(globalAnalysis.goalProgress, '%')} del objetivo</small>
              </article>
              <article>
                <span>Cartera invertida</span>
                <strong>{formatMoney(investedTotal)}</strong>
                <small>{formatNumber(globalAnalysis.investedWeight, '%')} del patrimonio visible</small>
              </article>
              <article>
                <span>Margen hasta objetivo</span>
                <strong>{formatMoney(globalAnalysis.remainingGoal)}</strong>
                <small>{globalAnalysis.monthsToGoal ? `${globalAnalysis.monthsToGoal} meses al ritmo actual` : 'sin ritmo suficiente'}</small>
              </article>
            </div>
            <div className="portfolio-guidance">
              {globalAnalysis.messages.map((message) => (
                <article key={message.title}>
                  <strong>{message.title}</strong>
                  <span>{message.text}</span>
                </article>
              ))}
            </div>
            <div className="portfolio-general-review">
              <section>
                <h4>Lectura de cartera completa</h4>
                <div className="review-pill-grid">
                  {globalAnalysis.reviewPills.map((pill) => (
                    <article key={pill.label}>
                      <span>{pill.label}</span>
                      <strong>{pill.value}</strong>
                      <small>{pill.detail}</small>
                    </article>
                  ))}
                </div>
              </section>
              <section>
                <h4>Siguientes pasos sugeridos</h4>
                <ol className="next-steps-list">
                  {globalAnalysis.nextSteps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </section>
            </div>
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
                    <strong>{formatMoney(summary?.finance_invested ?? summary?.known_cost)}</strong>
                  </article>
                  <article>
                    <span>Coste broker detectado</span>
                    <strong>{formatMoney(brokerCost)}</strong>
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
                      <span>Coste broker detectado: {formatMoney(broker.known_cost)}</span>
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
            <h3>Análisis por acción</h3>
            {analyzableStockPositions.length ? (
              <div className="portfolio-position-grid portfolio-analysis-list">
                {analyzableStockPositions.map((position) => (
                <article key={getPositionKey(position)}>
                  <div>
                    <strong>{position.ticker ?? position.name}</strong>
                    <span>{position.broker} · {position.region ?? 'Sin región'} · {position.sector ?? 'Sin sector'}</span>
                  </div>
                  <div className="position-score">
                    <span>Puntuación</span>
                    <strong>{formatPositionScore(positionAnalyses[getPositionKey(position)], position)}</strong>
                  </div>
                  <div className="portfolio-bar">
                    <span style={{ width: getWeight(position.current_value, investedTotal) }} />
                  </div>
                  <div className="position-card-footer">
                    <span>{formatMoney(position.current_value)} · {formatNumber((Number(position.current_value ?? 0) / investedTotal) * 100, '%')}</span>
                    <button
                      className="portfolio-summary-toggle"
                      type="button"
                      onClick={() => togglePositionAnalysis(position)}
                      disabled={!position.ticker}
                    >
                      {expandedPositionKey === getPositionKey(position) ? 'Cerrar análisis' : position.ticker ? 'Ver análisis' : 'Sin ticker'}
                    </button>
                  </div>
                  {expandedPositionKey === getPositionKey(position) ? (
                    <PositionAnalysis analysis={positionAnalyses[getPositionKey(position)]} position={position} />
                  ) : null}
                </article>
                ))}
              </div>
            ) : (
              <p className="muted-text">No hay acciones con ticker compatible para análisis automático en este snapshot.</p>
            )}
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

function buildSnapshotOptions(snapshot, snapshots, financeByMonth) {
  const byMonth = new Map()
  ;(snapshots ?? []).forEach((item) => {
    if (item?.snapshot_month) byMonth.set(item.snapshot_month, enrichSnapshotWithFinance(item, financeByMonth))
  })
  if (snapshot?.snapshot_month) byMonth.set(snapshot.snapshot_month, enrichSnapshotWithFinance(snapshot, financeByMonth))
  return Array.from(byMonth.values()).sort((left, right) =>
    (left.snapshot_month ?? '').localeCompare(right.snapshot_month ?? ''),
  )
}

function buildEvolutionRows(snapshots, financeByMonth) {
  return (snapshots ?? [])
    .filter((snapshot) => snapshot?.snapshot_month)
    .map((snapshot) => {
      const enriched = enrichSnapshotWithFinance(snapshot, financeByMonth)
      return {
        month: enriched.snapshot_month,
        date: enriched.snapshot_date,
        invested: Number(enriched.summary?.invested ?? 0),
        cost: Number(enriched.summary?.finance_invested ?? enriched.summary?.known_cost ?? 0),
        gain: Number(enriched.summary?.known_unrealized_gain ?? 0),
        dividends: Number(enriched.summary?.dividends ?? 0),
      }
    })
    .sort((left, right) => left.month.localeCompare(right.month))
}

function buildFinanceByMonth(rows) {
  return new Map(
    (rows ?? [])
      .filter((row) => row?.Mes)
      .map((row) => [
        row.Mes,
        {
          month: row.Mes,
          total: Number(row.total ?? 0),
          savings: Number(row['💰 Ahorros'] ?? 0),
          gifts: Number(row['🎁 Regalos'] ?? 0),
          holidays: Number(row['💼 Vacaciones'] ?? 0),
          emergencyFund: Number(row['Fondo de reserva cargado'] ?? 0),
          monthlyBudget: Number(row['💸 Presupuesto Mes'] ?? 0),
          monthlySpend: Number(row['💳 Gasto del mes'] ?? 0),
          finance_invested: Number(row['Dinero Invertido'] ?? 0),
          investment_bucket: Number(row['📈 Inversiones'] ?? row.Inversiones ?? 0),
          availableBudget: Number(row['🧾 Presupuesto Disponible'] ?? 0),
        },
      ]),
  )
}

function getLatestFinance(financeByMonth) {
  return Array.from(financeByMonth.values()).sort((left, right) => String(left.month).localeCompare(String(right.month))).at(-1) ?? null
}

function enrichSnapshotWithFinance(snapshot, financeByMonth) {
  const finance = financeByMonth.get(snapshot?.snapshot_month)
  if (!finance) return snapshot
  const summary = {
    ...(snapshot.summary ?? {}),
    finance_invested: finance.finance_invested,
    investment_bucket: finance.investment_bucket,
    investment_net_worth: finance.finance_invested + finance.investment_bucket,
  }
  return { ...snapshot, summary }
}

function loadLifeContext() {
  try {
    return { ...defaultLifeContext, ...JSON.parse(localStorage.getItem(lifeContextStorageKey) || '{}') }
  } catch (_err) {
    return defaultLifeContext
  }
}

function getPositionKey(position) {
  return `${position.broker}-${position.isin ?? position.ticker ?? position.name}-${position.asset_type}`
}

function buildGlobalPortfolioAnalysis(positions, investedTotal, costTotal, finance, context) {
  const homePrice = Number(context.homePrice || 0)
  const downPaymentGoal = homePrice * (Number(context.downPaymentPct || 0) / 100)
  const ownDownPaymentGoal = downPaymentGoal * (Number(context.ownSharePct || 0) / 100)
  const visibleLiquidity = Math.max(
    0,
    Number(finance?.savings ?? 0) +
      Number(finance?.investment_bucket ?? 0) +
      Number(finance?.holidays ?? 0) +
      Number(finance?.gifts ?? 0) +
      Number(finance?.emergencyFund ?? 0),
  )
  const visibleNetWorth = Math.max(0, visibleLiquidity + investedTotal)
  const plannedMonthlyInvestment = Number(finance?.investment_bucket ?? 0)
  const maxMonthlyInvestment = Number(context.maxMonthlyInvestment || plannedMonthlyInvestment || 0)
  const remainingGoal = Math.max(0, ownDownPaymentGoal - visibleLiquidity)
  const goalProgress = ownDownPaymentGoal > 0 ? (visibleLiquidity / ownDownPaymentGoal) * 100 : 0
  const investedWeight = visibleNetWorth > 0 ? (investedTotal / visibleNetWorth) * 100 : 0
  const monthsToGoal = maxMonthlyInvestment > 0 ? Math.ceil(remainingGoal / maxMonthlyInvestment) : null
  const concentration = getTopConcentration(positions, investedTotal)
  const returnPct = costTotal > 0 ? ((investedTotal - costTotal) / costTotal) * 100 : null
  const assetBreakdown = buildBreakdown(positions, 'asset_type', investedTotal)
  const regionBreakdown = buildBreakdown(positions, 'region', investedTotal)
  const sectorBreakdown = buildBreakdown(positions, 'sector', investedTotal)
  const stockWeight = assetBreakdown.find((item) => item.label === 'stock')?.weight ?? 0
  const fundWeight = assetBreakdown.find((item) => item.label === 'fund')?.weight ?? 0
  const topRegion = regionBreakdown[0]
  const topSector = sectorBreakdown[0]
  const topPosition = [...positions].sort((left, right) => Number(right.current_value ?? 0) - Number(left.current_value ?? 0))[0]
  const messages = [
    {
      title: 'Situación actual',
      text: `Tienes ${formatMoney(investedTotal)} invertidos y ${formatMoney(visibleLiquidity)} de liquidez visible. La cartera pesa ${formatNumber(investedWeight, '%')} sobre patrimonio visible de cartera + liquidez.`,
    },
    {
      title: 'Objetivo vivienda',
      text: `Para una vivienda de ${formatMoney(homePrice)}, una entrada del ${formatNumber(context.downPaymentPct, '%')} supone ${formatMoney(downPaymentGoal)}. Tu objetivo del ${formatNumber(context.ownSharePct, '%')} son ${formatMoney(ownDownPaymentGoal)}.`,
    },
    {
      title: 'Ritmo y prudencia',
      text:
        remainingGoal <= 0
          ? 'Con la liquidez visible ya cubrirías la referencia marcada para tu parte de la entrada. El siguiente paso sería proteger ese capital de volatilidad innecesaria.'
          : `Faltan ${formatMoney(remainingGoal)}. Con una aportación máxima de ${formatMoney(maxMonthlyInvestment)} al mes, el objetivo tardaría aproximadamente ${monthsToGoal ?? 's/d'} meses.`,
    },
    {
      title: 'Cartera',
      text: `La mayor posición pesa ${formatNumber(concentration, '%')}. La rentabilidad conocida de la cartera está en ${returnPct === null ? 's/d' : formatNumber(returnPct, '%')}; para un objetivo de vivienda cercano, conviene que las nuevas compras no comprometan la liquidez planificada.`,
    },
  ]
  const reviewPills = [
    {
      label: 'Fondos',
      value: formatNumber(fundWeight, '%'),
      detail: `${assetBreakdown.find((item) => item.label === 'fund')?.count ?? 0} posiciones diversificadas`,
    },
    {
      label: 'Acciones directas',
      value: formatNumber(stockWeight, '%'),
      detail: `${assetBreakdown.find((item) => item.label === 'stock')?.count ?? 0} valores individuales`,
    },
    {
      label: 'Mayor región',
      value: topRegion?.label ?? 's/d',
      detail: topRegion ? `${formatNumber(topRegion.weight, '%')} de la cartera` : 'sin clasificar',
    },
    {
      label: 'Mayor sector',
      value: topSector?.label ?? 's/d',
      detail: topSector ? `${formatNumber(topSector.weight, '%')} de la cartera` : 'sin clasificar',
    },
    {
      label: 'Mayor posición',
      value: topPosition?.ticker ?? topPosition?.name ?? 's/d',
      detail: `${formatNumber(concentration, '%')} de peso`,
    },
    {
      label: 'Rentabilidad',
      value: returnPct === null ? 's/d' : formatNumber(returnPct, '%'),
      detail: `${formatMoney(investedTotal - costTotal)} frente a dinero invertido`,
    },
  ]
  const nextSteps = buildNextSteps({
    concentration,
    fundWeight,
    goalProgress,
    investedWeight,
    maxMonthlyInvestment,
    monthsToGoal,
    remainingGoal,
    stockWeight,
    topRegion,
    topSector,
  })

  return {
    downPaymentGoal,
    goalProgress,
    investedWeight,
    messages,
    monthsToGoal,
    nextSteps,
    ownDownPaymentGoal,
    plannedMonthlyInvestment,
    remainingGoal,
    reviewPills,
    visibleLiquidity,
  }
}

function buildBreakdown(positions, field, investedTotal) {
  if (!investedTotal) return []
  const groups = positions.reduce((acc, position) => {
    const label = field === 'asset_type' ? position.asset_type : position[field] || 'Sin clasificar'
    if (!acc[label]) acc[label] = { count: 0, value: 0 }
    acc[label].count += 1
    acc[label].value += Number(position.current_value ?? 0)
    return acc
  }, {})
  return Object.entries(groups)
    .map(([label, item]) => ({ label, count: item.count, value: item.value, weight: (item.value / investedTotal) * 100 }))
    .sort((left, right) => right.value - left.value)
}

function buildNextSteps({
  concentration,
  fundWeight,
  goalProgress,
  investedWeight,
  maxMonthlyInvestment,
  monthsToGoal,
  remainingGoal,
  stockWeight,
  topRegion,
  topSector,
}) {
  const steps = []
  if (remainingGoal > 0) {
    steps.push(
      `Priorizar liquidez para vivienda: faltan ${formatMoney(remainingGoal)} y, al ritmo configurado, quedan aproximadamente ${monthsToGoal ?? 's/d'} meses.`,
    )
  } else {
    steps.push('Separar mentalmente la entrada de la vivienda del dinero de inversión: si ya está cubierta, conviene reducir volatilidad de esa parte.')
  }
  if (maxMonthlyInvestment > 0) {
    steps.push(`Mantener como techo de nuevas compras ${formatMoney(maxMonthlyInvestment)} al mes mientras el objetivo vivienda siga abierto.`)
  }
  if (investedWeight > 35 && goalProgress < 100) {
    steps.push('Evitar aumentar mucho el peso invertido hasta que la entrada esté más avanzada; la liquidez tiene prioridad temporal.')
  }
  if (concentration > 20) {
    steps.push(`Revisar concentración: la posición principal pesa ${formatNumber(concentration, '%')}, por encima de un nivel cómodo para una cartera en construcción.`)
  }
  if (stockWeight > fundWeight && goalProgress < 100) {
    steps.push('Para nuevas aportaciones, favorecer fondos diversificados o liquidez frente a acciones individuales mientras el objetivo vivienda sea dominante.')
  } else if (fundWeight >= stockWeight) {
    steps.push('La base de fondos ayuda a diversificar; las acciones individuales deberían añadirse de forma selectiva y con importes controlados.')
  }
  if (topRegion?.weight > 55) {
    steps.push(`Vigilar sesgo geográfico: ${topRegion.label} concentra ${formatNumber(topRegion.weight, '%')} de la cartera.`)
  }
  if (topSector?.weight > 35) {
    steps.push(`Vigilar sesgo sectorial: ${topSector.label} concentra ${formatNumber(topSector.weight, '%')} de la cartera.`)
  }
  return steps.slice(0, 6)
}

function getTopConcentration(positions, investedTotal) {
  if (!investedTotal) return 0
  const top = Math.max(...positions.map((position) => Number(position.current_value ?? 0)), 0)
  return (top / investedTotal) * 100
}

function formatPositionScore(analysis, position) {
  if (analysis?.status === 'loading') return '...'
  if (analysis?.result?.score !== undefined) return `${Math.round(Number(analysis.result.score))}/100`
  return position.asset_type === 'stock' && position.ticker ? 'Pendiente' : 's/d'
}

function PositionAnalysis({ analysis, position }) {
  if (!position.ticker) {
    return <p className="muted-text position-analysis">Esta posición no tiene ticker compatible para análisis automático.</p>
  }
  if (!analysis || analysis.status === 'loading') {
    return <p className="muted-text position-analysis">Analizando {position.ticker}...</p>
  }
  if (analysis.status === 'error') {
    return <p className="error-message position-analysis">{analysis.error}</p>
  }

  const result = analysis.result
  const metrics = result?.metrics ?? {}
  const aiText = result?.ai_analysis?.text ?? ''
  const paragraphs = aiText
    ? aiText
        .split(/\n{2,}/)
        .map((paragraph) => paragraph.trim())
        .filter(Boolean)
        .slice(0, 6)
    : []

  return (
    <div className="position-analysis">
      <div className="position-analysis-head">
        <article>
          <span>Resultado</span>
          <strong>{Math.round(Number(result.score ?? 0))}/100</strong>
          <small>{result.recommendation}</small>
        </article>
        <article>
          <span>Precio</span>
          <strong>{formatNumber(result.price, metrics.currency)}</strong>
          <small>{result.exchange?.name ?? 'Mercado s/d'}</small>
        </article>
        <article>
          <span>Sector</span>
          <strong>{result.sector ?? metrics.sector ?? 's/d'}</strong>
          <small>{result.exchange?.country ?? metrics.country ?? 'País s/d'}</small>
        </article>
      </div>
      <div className="position-ratio-grid">
        <span>RPD TTM: {formatNumber(metrics.rpd_ttm, '%')}</span>
        <span>DGR 5 años: {formatNumber(metrics.dgr5, '%')}</span>
        <span>Payout: {formatNumber(metrics.payout, '%')}</span>
        <span>PER: {formatNumber(metrics.per_ttm)}</span>
        <span>Deuda/capital: {formatNumber(metrics.de_ratio)}</span>
        <span>Racha dividendo: {formatNumber(metrics.streak_years, 'años')}</span>
      </div>
      {result.flags?.length ? (
        <div className="position-flags">
          {result.flags.map((flag) => (
            <span key={flag}>{flag}</span>
          ))}
        </div>
      ) : null}
      {paragraphs.length ? (
        <div className="position-ai-summary">
          {paragraphs.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </div>
      ) : (
        <p className="muted-text">Sin análisis cualitativo disponible para este ticker.</p>
      )}
    </div>
  )
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
