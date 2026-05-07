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

El frontend usa `VITE_API_URL` para localizar el backend. Si no se define, apunta a `http://localhost:8000`.

### Configuración

- `VITE_API_URL`: URL pública del backend para el frontend.
- `GESTIONES_CORS_ORIGINS`: orígenes permitidos por CORS, separados por comas.
- `GESTIONES_OBJECTIVES_PATH`: ruta del JSON donde se guardan los objetivos.

Puedes partir de `.env.example` si necesitas cambiar puertos, dominio o ruta de objetivos.

## Docker
```bash
docker compose up --build
```

El frontend espera a que `/health` del backend esté sano antes de arrancar.
Los objetivos se persisten en `Personal_finanzas/Data/objetivos_vista.json` mediante un volumen local.

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
- Fase 4 completada: mejora de la experiencia de resultados en frontend, con selector de mes, comparación mensual, tendencia, filtro, resumen de objetivos calculados y exportación JSON/CSV.
- Fase 5 completada: endurecimiento operativo con variables de entorno, healthcheck, persistencia local de objetivos, `.dockerignore` y targets Docker separados para runtime/tests.
- Fase 6 en curso: endurecimiento de experiencia de uso, cliente API frontend y mensajes de error más consistentes.

### Pendiente Fase 6

- Añadir pruebas unitarias ligeras para utilidades frontend (`api`, `exporters`, validación).
