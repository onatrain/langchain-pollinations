from pathlib import Path

import pytest

from langchain_pollinations._auth import AuthConfig, ENV_API_KEY


@pytest.fixture
def api_key_from_env(monkeypatch) -> str:
    """
    Lee POLLINATIONS_API_KEY desde .env si existe, y lo inyecta en el entorno.
    Si no existe o no define la variable, usa un valor por defecto para tests.
    """
    env_path = Path(".env")
    api_key = "test_api_key_from_env"

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == ENV_API_KEY:
                api_key = value.strip().strip("'").strip('"')
                break

    monkeypatch.setenv(ENV_API_KEY, api_key)
    return api_key


def test_from_env_or_value_uses_explicit_value(monkeypatch, api_key_from_env):
    # Aunque el entorno tenga un API key, el valor explícito debe prevalecer.
    monkeypatch.setenv(ENV_API_KEY, api_key_from_env)
    explicit_key = "explicit_api_key"

    cfg = AuthConfig.from_env_or_value(explicit_key)

    assert isinstance(cfg, AuthConfig)
    assert cfg.api_key == explicit_key


def test_from_env_or_value_reads_from_env(api_key_from_env):
    # Si api_key es None, debe usar la variable de entorno.
    cfg = AuthConfig.from_env_or_value(None)

    assert isinstance(cfg, AuthConfig)
    assert cfg.api_key == api_key_from_env


def test_from_env_or_value_raises_if_missing(monkeypatch):
    # Sin valor explícito ni variable de entorno debe lanzar ValueError.
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    with pytest.raises(ValueError) as exc:
        AuthConfig.from_env_or_value(None)

    assert "API key missing" in str(exc.value)
