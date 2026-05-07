# Gestiones

Aplicación de finanzas personales en proceso de refactorización incremental a arquitectura hexagonal con backend y frontend separados.

## Estructura

- `backend/`: API FastAPI con capas hexagonales.
- `frontend/`: UI React + Vite mínima.
- `Personal_finanzas/`: código legado original preservado.
- `docker-compose.yml`: orquestación completa.

## Desarrollo local

### Backend
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e "backend[dev]"
uvicorn backend.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Docker
```bash
docker compose up --build
```

## Tests
```bash
pytest -q
docker compose --profile test run --rm backend-test
```

## Estado de migración

- Fase 0 completada: esqueleto hexagonal, API base y validación Docker.
- Fase 1 completada: endpoint de procesado con respuesta JSON serializable y frontend para subir Excel, ajustar parámetros y consultar el resumen.
- Fase 2 completada: objetivos presupuestarios enviados desde frontend/backend, persistencia en JSON y limpieza de la liquidación de objetivos en el pipeline legado.
- Fase 3 completada: extracción de repositorios de infraestructura y contratos de dominio (`PipelineConfig`, `ObjectiveConfig`, `PipelineResult`) para reducir dicts sueltos alrededor del pipeline legado.
- Fase 4 en curso: mejora de la experiencia de resultados en frontend, con selector de mes, comparación mensual, tendencia, filtro, resumen de objetivos calculados y exportación JSON/CSV.
