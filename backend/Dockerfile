FROM python:3.11-slim
WORKDIR /app
COPY backend/pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir .[dev]
COPY backend /app/backend
COPY Personal_finanzas /app/Personal_finanzas
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
