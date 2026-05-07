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
- Fase 1 en curso: endpoint de procesado con respuesta JSON serializable y frontend para subir Excel, ajustar parámetros y consultar el resumen.
