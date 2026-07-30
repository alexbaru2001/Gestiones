import { FileSpreadsheet, FolderOpen, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'
import { formatPercent } from './formatters'

export function InputPanel({
  file,
  params,
  objectives,
  isLoadingObjectives,
  isSavingObjectives,
  isProcessing,
  hasValidationErrors,
  validationMessages,
  objectivesStatus,
  error,
  errorRef,
  onFileChange,
  onParamChange,
  onObjectiveAdd,
  onObjectiveChange,
  onObjectiveRemove,
  onObjectivesLoad,
  onObjectivesSave,
  onSubmit,
}) {
  return (
    <form className="panel" onSubmit={onSubmit}>
      <div className="panel-header">
        <div>
          <span className="section-kicker">Configuración</span>
          <h2>Datos financieros</h2>
        </div>
        <span className={file ? 'file-status selected' : 'file-status'}>{file ? file.name : 'Sin archivo'}</span>
      </div>

      <label className="file-input">
        <input type="file" accept=".xlsx,.xlsm,.xls" onChange={(event) => onFileChange(event.target.files?.[0] ?? null)} />
        <span>
          <FileSpreadsheet aria-hidden="true" size={19} />
          Seleccionar Excel
        </span>
      </label>

      <div className="field-grid">
        <label>
          Fecha inicio
          <input type="date" value={params.fecha_inicio} onChange={(event) => onParamChange('fecha_inicio', event.target.value)} />
        </label>
        <label>
          Gasto
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={params.porcentaje_gasto}
            onChange={(event) => onParamChange('porcentaje_gasto', event.target.value)}
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
            onChange={(event) => onParamChange('porcentaje_inversion', event.target.value)}
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
            onChange={(event) => onParamChange('porcentaje_vacaciones', event.target.value)}
          />
          <strong>{formatPercent(params.porcentaje_vacaciones)}</strong>
        </label>
      </div>

      <div className="objectives-section">
        <div className="section-heading">
          <h3>Objetivos</h3>
          <div className="button-row">
            <button className="text-button" type="button" onClick={onObjectivesLoad} disabled={isLoadingObjectives}>
              <RefreshCw aria-hidden="true" size={15} />
              {isLoadingObjectives ? 'Cargando...' : 'Cargar'}
            </button>
            <button className="text-button" type="button" onClick={onObjectivesSave} disabled={isSavingObjectives || hasValidationErrors}>
              <Save aria-hidden="true" size={15} />
              {isSavingObjectives ? 'Guardando...' : 'Guardar'}
            </button>
            <button className="text-button" type="button" onClick={onObjectiveAdd}>
              <Plus aria-hidden="true" size={15} />
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
                  <input type="text" value={objective.nombre} onChange={(event) => onObjectiveChange(objective.id, 'nombre', event.target.value)} />
                </label>
                <label>
                  Etiquetas
                  <input type="text" value={objective.etiquetas} onChange={(event) => onObjectiveChange(objective.id, 'etiquetas', event.target.value)} />
                </label>
                <label>
                  Fracción
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={objective.fraccion_presupuesto}
                    onChange={(event) => onObjectiveChange(objective.id, 'fraccion_presupuesto', event.target.value)}
                  />
                </label>
                <label>
                  Meses
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={objective.duracion_meses}
                    onChange={(event) => onObjectiveChange(objective.id, 'duracion_meses', event.target.value)}
                  />
                </label>
                <label>
                  Inicio
                  <input type="month" value={objective.mes_inicio} onChange={(event) => onObjectiveChange(objective.id, 'mes_inicio', event.target.value)} />
                </label>
                <label>
                  Saldo inicial
                  <input
                    type="number"
                    step="0.01"
                    value={objective.saldo_inicial}
                    onChange={(event) => onObjectiveChange(objective.id, 'saldo_inicial', event.target.value)}
                  />
                </label>
                <button className="text-button danger" type="button" onClick={() => onObjectiveRemove(objective.id)}>
                  <Trash2 aria-hidden="true" size={15} />
                  Quitar
                </button>
              </div>
            ))}
          </div>
        )}
        {validationMessages.length > 0 && (
          <div className="validation-box" role="alert" aria-live="polite">
            {validationMessages.slice(0, 4).map((message) => (
              <p key={message}>{message}</p>
            ))}
          </div>
        )}
        {objectivesStatus && (
          <p className="status-message" aria-live="polite">
            {objectivesStatus}
          </p>
        )}
      </div>

      <button className="primary-button" type="submit" disabled={isProcessing || hasValidationErrors}>
        <FolderOpen aria-hidden="true" size={18} />
        {isProcessing ? 'Procesando...' : 'Procesar'}
      </button>

      {error && (
        <p className="error-message" role="alert" tabIndex="-1" ref={errorRef}>
          {error}
        </p>
      )}
    </form>
  )
}
