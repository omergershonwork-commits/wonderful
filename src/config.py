"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings loaded from environment variables or ``.env``."""

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = ""
    use_llm: bool = True

    use_live_data: bool = False
    http_timeout_seconds: int = Field(default=20, gt=0, le=120)

    long_haul_threshold_miles: int = Field(default=3000, gt=0)
    target_load_factor: float = 0.82
    min_annual_passengers: int = Field(default=100_000, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("target_load_factor")
    @classmethod
    def validate_target_load_factor(cls, value: float) -> float:
        """Require a percentage represented as a decimal in the interval ``(0, 1]``."""

        if not 0 < value <= 1:
            raise ValueError("TARGET_LOAD_FACTOR must be greater than 0 and at most 1")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached, validated settings instance for the process."""

    return Settings()
