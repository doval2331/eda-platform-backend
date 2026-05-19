# Despliegue — Plataforma EDA

Guía resumida para **PostgreSQL + API + frontend** (opción B en local con Docker; producción en la nube).

## 1. Local con Docker + DBeaver

Ver sección *Base de datos* en el [README](../README.md).

## 2. Producción recomendada (gratis / bajo coste)

### PostgreSQL (Neon o Supabase)

1. Crea un proyecto y una base de datos.
2. Copia la connection string (modo **psycopg2** / SQLAlchemy):

   `postgresql+psycopg2://USER:PASSWORD@HOST/DB?sslmode=require`

3. Pégala en `DATABASE_URL` del backend.

### Backend (Render)

1. Nuevo **Web Service** → conecta el repo `eda-platform-backend`.
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Variables de entorno: `DATABASE_URL`, `JWT_SECRET`, `CORS_ORIGINS`, `IT_OPS_DATASET_PATH`, credenciales demo.
5. Incluye `data/it_ops_synthetic_10000.csv` en el deploy o ejecuta el generador en build.
6. Shell / one-off job: `python scripts/wait_for_db.py && python scripts/seed_user.py`

### Frontend (Vercel)

1. Importa `eda-platform-frontend`.
2. Variable de entorno en build:

   `VITE_API_BASE=https://tu-servicio.onrender.com`

3. Build: `npm run build` — Output: `dist`.

### DBeaver contra producción

Usa host, puerto, SSL y credenciales del panel de Neon/Supabase (mismas tablas `users`, `analysis_runs`).

## 3. VPS con Docker

```bash
# Solo Postgres en el servidor
docker compose up -d

# API en el host o en otro contenedor con DATABASE_URL=postgresql+psycopg2://eda:...@127.0.0.1:5432/eda_platform
```

Restringe el puerto `5432` con firewall; accede a DBeaver por túnel SSH si hace falta.
