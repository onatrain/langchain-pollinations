import pytest

from langchain_pollinations._auth import AuthConfig, ENV_API_KEY


def test_auth_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_KEY, "k")
    cfg = AuthConfig.from_env_or_value(None)
    assert cfg.api_key == "k"


def test_auth_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(ValueError):
        AuthConfig.from_env_or_value(None)
