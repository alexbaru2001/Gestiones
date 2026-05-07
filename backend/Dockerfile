FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app
COPY backend/pyproject.toml /app/pyproject.toml
COPY backend /app/backend
RUN pip install --no-cache-dir .
COPY Personal_finanzas /app/Personal_finanzas
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
RUN pip install --no-cache-dir .[dev]
