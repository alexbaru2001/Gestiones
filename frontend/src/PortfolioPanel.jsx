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

export function PortfolioPanel() {
  const fileInputRef = useRef(null)
  const [snapshot, setSnapshot] = useState(null)
  const [files, setFiles] = useState([])
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isImporting, setIsImporting] = useState(false)

  const positions = snapshot?.positions ?? []
  const brokers = snapshot?.brokers ?? []
  const summary = snapshot?.summary
  const distribution = useMemo(() => buildDistribution(positions, summary?.total_value), [positions, summary])

  useEffect(() => {
    loadPortfolio()
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

      {!snapshot ? (
        <div className="empty-state compact-empty">
          <strong>Sin cartera local</strong>
          <span>Importa los extractos de Trade Republic, MyInvestor y DeGiro para generar el resumen.</span>
        </div>
      ) : (
        <div className="portfolio-dashboard">
          <div className="portfolio-summary">
            <article>
              <span>Patrimonio visible</span>
              <strong>{formatMoney(summary?.total_value)}</strong>
            </article>
            <article>
              <span>Invertido</span>
              <strong>{formatMoney(summary?.invested)}</strong>
            </article>
            <article>
              <span>Efectivo</span>
              <strong>{formatMoney(summary?.cash)}</strong>
            </article>
            <article>
              <span>Rentabilidad conocida</span>
              <strong>{formatMoney(summary?.known_unrealized_gain)}</strong>
              <small>{formatNumber(summary?.known_unrealized_gain_pct, '%')}</small>
            </article>
            <article>
              <span>Dividendos</span>
              <strong>{formatMoney(summary?.dividends)}</strong>
            </article>
            <article>
              <span>Intereses</span>
              <strong>{formatMoney(summary?.interest)}</strong>
            </article>
          </div>

          {snapshot.warnings?.length ? (
            <section className="warning-panel">
              <strong>Avisos de cálculo</strong>
              {snapshot.warnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </section>
          ) : null}

          <section className="portfolio-card">
            <h3>Distribución</h3>
            <div className="portfolio-distribution">
              {distribution.map((item) => (
                <article key={item.label}>
                  <div>
                    <strong>{item.label}</strong>
                    <span>{formatMoney(item.value)}</span>
                  </div>
                  <div className="portfolio-bar">
                    <span style={{ width: getWeight(item.value, summary?.total_value) }} />
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="portfolio-card">
            <h3>Brokers</h3>
            <div className="broker-grid">
              {brokers.map((broker) => (
                <article key={broker.broker}>
                  <strong>{broker.broker}</strong>
                  <span>Total: {formatMoney(broker.total_value)}</span>
                  <span>Invertido: {formatMoney(broker.invested)}</span>
                  <span>Efectivo: {formatMoney(broker.cash)}</span>
                  <span>Rentabilidad conocida: {formatMoney(broker.known_unrealized_gain)}</span>
                  <span>Dividendos: {formatMoney(broker.dividends)}</span>
                  <span>Intereses: {formatMoney(broker.interest)}</span>
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
                    <th>Horizonte</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((position) => (
                    <tr key={`${position.broker}-${position.isin ?? position.name}-${position.asset_type}`}>
                      <td>{position.broker}</td>
                      <td>{position.ticker ?? position.name}</td>
                      <td>{formatAssetType(position.asset_type)}</td>
                      <td>{formatNumber(position.quantity)}</td>
                      <td>{formatMoney(position.current_value)}</td>
                      <td>{formatMoney(position.cost)}</td>
                      <td>{formatMoney(position.unrealized_gain)}</td>
                      <td>{position.horizon}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="portfolio-card">
            <h3>Archivos importados</h3>
            <div className="imported-files">
              {(snapshot.files ?? []).map((file) => (
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

function buildDistribution(positions, total) {
  if (!positions.length || !total) return []
  const groups = positions.reduce((acc, position) => {
    const label = position.asset_type === 'cash' ? 'Efectivo' : position.asset_type === 'fund' ? 'Fondos' : position.asset_type === 'crypto' ? 'Cripto' : 'Acciones'
    acc[label] = (acc[label] ?? 0) + Number(position.current_value ?? 0)
    return acc
  }, {})
  return Object.entries(groups)
    .map(([label, value]) => ({ label, value }))
    .sort((left, right) => right.value - left.value)
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
