from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://atbg:atbg_dev_password@localhost:5432/atbg"
    model_checkpoint_path: str = "./checkpoints/atbg_latest.pt"

    # Ensemble model settings
    ensemble_checkpoint_a: str | None = None
    ensemble_checkpoint_b: str | None = None
    ensemble_weight_a: float = 0.60

    # Threat-intel provider keys — all optional
    virustotal_api_key: str | None = None
    abuseipdb_api_key: str | None = None
    safe_browsing_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()