from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from langchain_pollinations.account import (
    AccountInformation,
    AccountUsageParams,
    AccountUsageDailyParams,
    DEFAULT_BASE_URL,
)


@dataclass
class DummyAuth:
    api_key: str


class DummyResponse:
    def __init__(self, data: Any):
        self._data = data

    def json(self) -> Any:
        return self._data


class DummyHttpClient:
    def __init__(self, *, config, api_key: str):
        self.config = config
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str, params: dict[str, Any] | None = None):
        self.calls.append({"method": "get", "path": path, "params": params})
        return DummyResponse({"path": path, "params": params})


@pytest.fixture(autouse=True)
def patch_auth_and_http(monkeypatch):
    # Evitar dependencias reales de entorno/API al inicializar AccountInformation.
    monkeypatch.setattr(
        "langchain_pollinations._auth.AuthConfig.from_env_or_value",
        staticmethod(lambda api_key: DummyAuth(api_key=api_key or "dummy")),
    )
    monkeypatch.setattr(
        "langchain_pollinations.account.PollinationsHttpClient",
        DummyHttpClient,
    )
    yield


def test_account_information_initializes_with_defaults():
    ai = AccountInformation()

    assert ai.base_url == DEFAULT_BASE_URL
    assert ai.timeout_s == 120.0
    assert isinstance(ai._http, DummyHttpClient)
    assert ai._http.api_key == "dummy"


def test_get_profile_calls_correct_endpoint():
    ai = AccountInformation(api_key="k")

    data = ai.get_profile()

    assert data["path"] == "/account/profile"
    assert ai._http.calls[-1]["path"] == "/account/profile"


def test_get_balance_calls_correct_endpoint():
    ai = AccountInformation(api_key="k")

    data = ai.get_balance()

    assert data["path"] == "/account/balance"
    assert ai._http.calls[-1]["path"] == "/account/balance"


def test_get_key_calls_correct_endpoint():
    ai = AccountInformation(api_key="k")

    data = ai.get_key()

    assert data["path"] == "/account/key"
    assert ai._http.calls[-1]["path"] == "/account/key"


def test_get_usage_with_default_params():
    ai = AccountInformation(api_key="k")

    data = ai.get_usage()

    assert data["path"] == "/account/usage"
    last_call = ai._http.calls[-1]
    assert last_call["path"] == "/account/usage"
    # Por defecto format=json, limit=100 y before=None (excluido por exclude_none).
    assert last_call["params"]["format"] == "json"
    assert last_call["params"]["limit"] == 100
    assert "before" not in last_call["params"]


def test_get_usage_with_custom_params():
    ai = AccountInformation(api_key="k")
    params = AccountUsageParams(format="csv", limit=10, before="2025-01-01T00:00:00Z")

    data = ai.get_usage(params=params)

    assert data["path"] == "/account/usage"
    last_call = ai._http.calls[-1]
    assert last_call["params"]["format"] == "csv"
    assert last_call["params"]["limit"] == 10
    assert last_call["params"]["before"] == "2025-01-01T00:00:00Z"


def test_get_usage_daily_with_default_params():
    ai = AccountInformation(api_key="k")

    data = ai.get_usage_daily()

    assert data["path"] == "/account/usage/daily"
    last_call = ai._http.calls[-1]
    assert last_call["path"] == "/account/usage/daily"
    # AccountUsageDailyParams sólo tiene format con default json.
    assert last_call["params"]["format"] == "json"


def test_get_usage_daily_with_custom_params():
    ai = AccountInformation(api_key="k")
    params = AccountUsageDailyParams(format="csv")

    data = ai.get_usage_daily(params=params)

    assert data["path"] == "/account/usage/daily"
    last_call = ai._http.calls[-1]
    assert last_call["params"]["format"] == "csv"


def test_account_usage_params_enforces_limit_range():
    # Límite fuera de rango [1, 50000] debe fallar.
    with pytest.raises(ValidationError):
        AccountUsageParams(limit=0)

    with pytest.raises(ValidationError):
        AccountUsageParams(limit=50001)

    # Un valor dentro de rango debe ser válido.
    ok = AccountUsageParams(limit=50000)
    assert ok.limit == 50000


def test_account_usage_params_forbids_extra_fields():
    # extra="forbid" en ConfigDict debe impedir campos desconocidos.
    with pytest.raises(ValidationError):
        AccountUsageParams(format="json", limit=10, before=None, extra_field="x")  # type: ignore[arg-type]


def test_account_usage_daily_params_forbids_extra_fields_and_validates_format():
    with pytest.raises(ValidationError):
        AccountUsageDailyParams(format="json", extra="x")  # type: ignore[arg-type]

    daily = AccountUsageDailyParams(format="csv")
    assert daily.format == "csv"
