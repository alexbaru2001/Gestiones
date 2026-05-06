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
pip install ./backend[dev]
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
pytest
```

## Estado de migración

Fase 1 completada parcialmente: se creó esqueleto hexagonal y API base reutilizando pipeline legado para conservar comportamiento.
