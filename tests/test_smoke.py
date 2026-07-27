"""Phase 2 smoke tests for configuration and repository setup."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_required_phase_two_files_exist() -> None:
    required_paths = (
        "app.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "pytest.ini",
        "src/__init__.py",
        "src/config.py",
    )

    for relative_path in required_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path


def test_default_settings_match_approved_configuration() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_api_key == "lm-studio"
    assert settings.use_llm is True
    assert settings.use_live_data is False
    assert settings.http_timeout_seconds == 20
    assert settings.long_haul_threshold_miles == 3000
    assert settings.target_load_factor == pytest.approx(0.82)
    assert settings.min_annual_passengers == 100_000


@pytest.mark.parametrize("invalid_value", [0, -0.1, 1.01])
def test_target_load_factor_is_validated(invalid_value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(target_load_factor=invalid_value, _env_file=None)
