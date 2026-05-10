import { useEffect, useMemo, useRef, useState } from 'react'
import { requestJson } from './api'
import { downloadJson, downloadText, toCsv } from './exporters'
import { InputPanel } from './InputPanel'
import { InvestmentPanel } from './InvestmentPanel'
import { PortfolioPanel } from './PortfolioPanel'
import { createObjective, objectiveFromApi, objectiveToPayload } from './objectives'
import {
  filterObjectiveRows,
  getHistoryRows,
  getObjectiveNames,
  getObjectiveRows,
  getObjectiveTotals,
  getPreviousRow,
  getRecentRows,
  getSelectedRow,
  getTrendMax,
} from './resultSelectors'
import { ResultsPanel } from './ResultsPanel'
import { validateFinanceInput } from './validation'

const initialParams = {
  fecha_inicio: '2024-10-01',
  porcentaje_gasto: 0.3,
  porcentaje_inversion: 0.1,
  porcentaje_vacaciones: 0.05,
}

export function App() {
  const errorRef = useRef(null)
  const [file, setFile] = useState(null)
  const [params, setParams] = useState(initialParams)
  const [objectives, setObjectives] = useState([])
  const [health, setHealth] = useState('pendiente')
  const [isChecking, setIsChecking] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isLoadingObjectives, setIsLoadingObjectives] = useState(false)
  const [isSavingObjectives, setIsSavingObjectives] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [objectivesStatus, setObjectivesStatus] = useState('')
  const [selectedObjective, setSelectedObjective] = useState('all')
  const [selectedMonth, setSelectedMonth] = useState('')
  const [activeArea, setActiveArea] = useState('finanzas')

  const latest = result?.historial?.ultimo_mes
  const rows = useMemo(() => getHistoryRows(result), [result])
  const selectedRow = useMemo(() => getSelectedRow(rows, latest, selectedMonth), [latest, rows, selectedMonth])
  const previousRow = useMemo(() => getPreviousRow(rows, selectedRow), [rows, selectedRow])
  const recentRows = useMemo(() => getRecentRows(rows), [rows])
  const trendMax = useMemo(() => getTrendMax(recentRows), [recentRows])
  const objectiveRows = useMemo(() => getObjectiveRows(result), [result])
  const objectiveNames = useMemo(() => getObjectiveNames(objectiveRows), [objectiveRows])
  const filteredObjectiveRows = useMemo(
    () => filterObjectiveRows(objectiveRows, selectedObjective),
    [objectiveRows, selectedObjective],
  )
  const objectiveTotals = useMemo(() => getObjectiveTotals(filteredObjectiveRows), [filteredObjectiveRows])
  const activeObjectives = useMemo(() => objectives.filter((objective) => objective.nombre.trim()), [objectives])
  const validationMessages = useMemo(() => validateFinanceInput(params, activeObjectives), [activeObjectives, params])
  const hasValidationErrors = validationMessages.length > 0
  const areaSubtitle =
    activeArea === 'finanzas'
      ? 'Procesado de finanzas personales'
      : activeArea === 'invertir'
        ? 'Análisis de inversión por dividendos'
        : 'Cartera local de inversión'

  useEffect(() => {
    loadObjectives({ silent: true })
  }, [])

  useEffect(() => {
    if (error) errorRef.current?.focus()
  }, [error])

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
    if (hasValidationErrors) {
      setError(validationMessages[0])
      return
    }

    setIsProcessing(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    const objetivos = activeObjectives.map(objectiveToPayload)
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
      const data = await requestJson(
        `/api/v1/process?${query.toString()}`,
        {
          method: 'POST',
          body: formData,
        },
        'No se pudo procesar el Excel',
      )
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
    setError('')
    setParams((current) => ({
      ...current,
      [key]: key === 'fecha_inicio' ? value : Number(value),
    }))
  }

  const addObjective = () => {
    setError('')
    setObjectives((current) => [...current, createObjective()])
  }

  const updateObjective = (id, key, value) => {
    setError('')
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
    setError('')
    setObjectives((current) => current.filter((objective) => objective.id !== id))
  }

  const loadObjectives = async ({ silent = false } = {}) => {
    setIsLoadingObjectives(true)
    setError('')
    if (!silent) setObjectivesStatus('Cargando...')
    try {
      const data = await requestJson('/api/v1/objectives', {}, 'No se pudieron cargar los objetivos')
      setObjectives((data.objetivos ?? []).map(objectiveFromApi))
      setObjectivesStatus(`${data.objetivos?.length ?? 0} objetivos cargados`)
    } catch (err) {
      setObjectivesStatus(silent ? '' : err.message)
    } finally {
      setIsLoadingObjectives(false)
    }
  }

  const saveObjectives = async () => {
    if (hasValidationErrors) {
      setError(validationMessages[0])
      setObjectivesStatus('')
      return
    }

    setIsSavingObjectives(true)
    setError('')
    setObjectivesStatus('Guardando...')

    try {
      const data = await requestJson(
        '/api/v1/objectives',
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ objetivos: activeObjectives.map(objectiveToPayload) }),
        },
        'No se pudieron guardar los objetivos',
      )
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
          <p>{areaSubtitle}</p>
        </div>
        <div className="topbar-actions">
          <nav className="area-switch" aria-label="Área de trabajo">
            <button
              className={activeArea === 'finanzas' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('finanzas')}
            >
              Finanzas
            </button>
            <button
              className={activeArea === 'invertir' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('invertir')}
            >
              Invertir
            </button>
            <button
              className={activeArea === 'cartera' ? 'active' : ''}
              type="button"
              onClick={() => setActiveArea('cartera')}
            >
              Cartera
            </button>
          </nav>
          <button className="ghost-button" type="button" onClick={checkHealth} disabled={isChecking}>
            {isChecking ? 'Comprobando...' : `Backend: ${health}`}
          </button>
        </div>
      </header>

      {activeArea === 'finanzas' ? (
        <section className="workspace">
          <InputPanel
            file={file}
            params={params}
            objectives={objectives}
            isLoadingObjectives={isLoadingObjectives}
            isSavingObjectives={isSavingObjectives}
            isProcessing={isProcessing}
            hasValidationErrors={hasValidationErrors}
            validationMessages={validationMessages}
            objectivesStatus={objectivesStatus}
            error={error}
            errorRef={errorRef}
            onFileChange={setFile}
            onParamChange={updateParam}
            onObjectiveAdd={addObjective}
            onObjectiveChange={updateObjective}
            onObjectiveRemove={removeObjective}
            onObjectivesLoad={() => loadObjectives()}
            onObjectivesSave={saveObjectives}
            onSubmit={processWorkbook}
          />

          <ResultsPanel
            isProcessing={isProcessing}
            rows={rows}
            selectedRow={selectedRow}
            selectedObjective={selectedObjective}
            previousRow={previousRow}
            recentRows={recentRows}
            trendMax={trendMax}
            result={result}
            objectiveRows={objectiveRows}
            objectiveNames={objectiveNames}
            filteredObjectiveRows={filteredObjectiveRows}
            objectiveTotals={objectiveTotals}
            onMonthChange={setSelectedMonth}
            onObjectiveFilterChange={setSelectedObjective}
            onExportResult={exportResult}
            onExportHistoryCsv={exportHistoryCsv}
            onExportObjectivesCsv={exportObjectivesCsv}
          />
        </section>
      ) : activeArea === 'invertir' ? (
        <InvestmentPanel />
      ) : (
        <PortfolioPanel />
      )}
    </main>
  )
}
