FROM python:3.12-slim

WORKDIR /app

# Dependencias de compilación para hdbscan, umap-learn, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v -E '^(jupyter|ipywidgets|matplotlib|seaborn)' requirements.txt > requirements.prod.txt \
    && pip install --no-cache-dir -r requirements.prod.txt

COPY app ./app
COPY scripts ./scripts
COPY data/.gitkeep data/uploads/.gitkeep ./data/
RUN mkdir -p data/uploads

# El CSV de demo no está en git; se genera en el build.
RUN python scripts/generate_it_ops_dataset.py

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

# Dokploy y otros PaaS suelen inyectar PORT; si no, usa API_PORT o 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-${API_PORT:-8000}}"]
