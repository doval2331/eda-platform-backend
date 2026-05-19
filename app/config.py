from functools import lru_cache

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
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    default_seed: int = 42
    default_n_samples: int = 2000
    it_ops_dataset_path: str = "data/it_ops_synthetic_10000.csv"
    uploads_dir: str = "data/uploads"
    max_upload_bytes: int = 15 * 1024 * 1024
    jwt_secret: str = "change-me-in-production-use-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    demo_user_email: str = "analista@tfm.local"
    demo_user_password: str = "TfmDemo2026!"
    demo_user_nombre: str = "Analista TFM"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
