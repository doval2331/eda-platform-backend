-- Extensiones opcionales. Tablas de la app: SQLAlchemy init_db() + scripts/seed_user.py.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Base interna de Metabase. Si el volumen ya existia, crearla manualmente:
-- docker compose exec postgres psql -U eda -d eda_platform -c "CREATE DATABASE metabase OWNER eda;"
SELECT 'CREATE DATABASE metabase OWNER eda'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metabase')\gexec
