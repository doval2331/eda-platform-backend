from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./eda_platform.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174"
    )
    default_seed: int = 42
    default_n_samples: int = 2000
    it_ops_dataset_path: str = "data/it_ops_synthetic_10000.csv"
    uploads_dir: str = "data/uploads"
    max_upload_bytes: int = 250 * 1024 * 1024
    duckdb_path: str = "data/eda_platform.duckdb"
    bi_sync_enabled: bool = False
    bi_database_url: str = (
        "postgresql+psycopg2://eda:eda_local_dev@127.0.0.1:5432/eda_platform"
    )
    metabase_url: str = "http://localhost:3000"
    metabase_dashboard_url: str = ""
    metabase_username: str = ""
    metabase_password: str = ""
    metabase_database_name: str = "TFM IT Analytics"
    metabase_dashboard_name: str = "Dashboard IT - Evidencias conversacionales"
    metabase_dashboard_id: int | None = None
    metabase_embedding_secret: str = ""
    metabase_embed_expire_minutes: int = 10
    llm_enabled: bool = False
    llm_provider: str = "openai_compatible"
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_transcription_model: str = "gpt-4o-mini-transcribe"
    llm_api_version: str = ""
    llm_timeout_seconds: int = 20
    metabase_pg_host: str = "postgres"
    metabase_pg_port: int = 5432
    metabase_pg_dbname: str = "eda_platform"
    metabase_pg_user: str = "eda"
    metabase_pg_password: str = "eda_local_dev"
    jwt_secret: str = "change-me-in-production-use-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    demo_user_email: str = "analista@tfm.local"
    demo_user_password: str = "TfmDemo2026!"
    demo_user_nombre: str = "Analista TFM"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
