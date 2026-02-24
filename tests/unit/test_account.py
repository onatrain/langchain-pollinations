from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from langchain_pollinations.account import (
    AccountInformation,
    AccountProfile,
    AccountUsageDailyParams,
    AccountUsageParams,
    AccountUsageRecord,
    AccountUsageResponse,
)

_PROFILE_FULL: dict = {
    "name": "Alice",
    "email": "alice@example.com",
    "githubUsername": "alice-gh",
    "image": "https://avatars.githubusercontent.com/u/99999",
    "tier": "seed",
    "createdAt": "2025-06-01T00:00:00Z",
    "nextResetAt": "2026-02-24T00:00:00Z",
}

_PROFILE_ALL_NULL: dict = {
    "name": None,
    "email": None,
    "githubUsername": None,
    "image": None,
    "tier": "anonymous",
    "createdAt": "2025-01-01T00:00:00Z",
    "nextResetAt": None,
}

_RECORD_FULL: dict = {
    "timestamp": "2026-02-23 10:00:00",
    "type": "generate.text",
    "model": "openai",
    "api_key": "sk_abc***",
    "api_key_type": "secret",
    "meter_source": "tier",
    "input_text_tokens": 100.0,
    "input_cached_tokens": 20.0,
    "input_audio_tokens": 0.0,
    "input_image_tokens": 0.0,
    "output_text_tokens": 50.0,
    "output_reasoning_tokens": 10.0,
    "output_audio_tokens": 0.0,
    "output_image_tokens": 1.0,
    "cost_usd": 0.002,
    "response_time_ms": 345.6,
}

_RECORD_MINIMAL: dict = {
    "timestamp": "2026-02-23 11:00:00",
    "type": "generate.image",
    "model": None,
    "api_key": None,
    "api_key_type": None,
    "meter_source": None,
    "input_text_tokens": 0.0,
    "input_cached_tokens": 0.0,
    "input_audio_tokens": 0.0,
    "input_image_tokens": 0.0,
    "output_text_tokens": 0.0,
    "output_reasoning_tokens": 0.0,
    "output_audio_tokens": 0.0,
    "output_image_tokens": 1.0,
    "cost_usd": 0.01,
    "response_time_ms": None,
}

_USAGE_RESPONSE: dict = {
    "usage": [_RECORD_FULL, _RECORD_MINIMAL],
    "count": 2,
}

_DAILY_RESPONSE: dict = {
    "usage": [
        {
            "date": "2026-02-23",
            "model": "openai",
            "meter_source": "tier",
            "requests": 5,
            "cost_usd": 0.01,
        }
    ],
    "count": 1,
}


def _json_resp(data: dict) -> MagicMock:
    """Mock de respuesta con Content-Type application/json."""
    resp = MagicMock()
    resp.headers = {"content-type": "application/json; charset=utf-8"}
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _csv_resp(text: str) -> MagicMock:
    """Mock de respuesta con Content-Type text/csv."""
    resp = MagicMock()
    resp.headers = {"content-type": "text/csv; charset=utf-8"}
    resp.text = text
    resp.json.side_effect = ValueError("not json")
    return resp


def _plain_resp(text: str) -> MagicMock:
    """Mock de respuesta con Content-Type text/plain."""
    resp = MagicMock()
    resp.headers = {"content-type": "text/plain"}
    resp.text = text
    resp.json.side_effect = ValueError("not json")
    return resp


def _unknown_json_resp(data: dict) -> MagicMock:
    """Mock de respuesta con Content-Type desconocido pero JSON parseable."""
    resp = MagicMock()
    resp.headers = {"content-type": "application/octet-stream"}
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _unknown_plain_resp(text: str) -> MagicMock:
    """Mock de respuesta con Content-Type desconocido y JSON no parseable."""
    resp = MagicMock()
    resp.headers = {"content-type": "application/octet-stream"}
    resp.json.side_effect = ValueError("not json")
    resp.text = text
    return resp


@pytest.fixture
def mock_http() -> MagicMock:
    """
    Fake PollinationsHttpClient with sync get() and async aget() pre-configured.

    ``get`` is a plain MagicMock; ``aget`` is an AsyncMock so it can be
    awaited without side-effects inside coroutines.
    """
    client = MagicMock()
    # aget necesita ser AsyncMock para que ``await client.aget(...)`` funcione.
    client.aget = AsyncMock()
    return client


@pytest.fixture
def account(mock_http: MagicMock):
    """
    AccountInformation instance whose internal _http is replaced by mock_http.

    Authentication and HTTP client construction are fully patched so no real
    network calls or environment variables are required.

    Returns:
        Tuple of (AccountInformation instance, mock_http MagicMock).
    """
    with (
        patch("langchain_pollinations.account.AuthConfig.from_env_or_value") as mock_auth,
        patch("langchain_pollinations.account.PollinationsHttpClient") as mock_cls,
    ):
        mock_auth.return_value = MagicMock(api_key="sk_test")
        # Cuando __post_init__ llame a PollinationsHttpClient(...), retornará mock_http.
        mock_cls.return_value = mock_http
        info = AccountInformation(api_key="sk_test")
    # info._http es ahora mock_http (asignado en __post_init__ vía el slot declarado).
    return info, mock_http


class TestAccountProfile:
    """Parsing and validation of /account/profile responses."""

    def test_full_payload_all_fields_mapped(self) -> None:
        """All camelCase fields from the API are mapped to snake_case correctly."""
        p = AccountProfile.model_validate(_PROFILE_FULL)
        assert p.name == "Alice"
        assert p.email == "alice@example.com"
        assert p.github_username == "alice-gh"
        assert p.image == "https://avatars.githubusercontent.com/u/99999"
        assert p.tier == "seed"
        assert p.created_at == "2025-06-01T00:00:00Z"
        assert p.next_reset_at == "2026-02-24T00:00:00Z"

    def test_nullable_fields_accept_null(self) -> None:
        """All Optional fields accept None from the backend."""
        p = AccountProfile.model_validate(_PROFILE_ALL_NULL)
        assert p.name is None
        assert p.email is None
        assert p.github_username is None
        assert p.image is None
        assert p.next_reset_at is None

    def test_image_url_preserved_verbatim(self) -> None:
        """The avatar URL is stored without modification, including query strings."""
        url = "https://avatars.githubusercontent.com/u/12345?v=4"
        p = AccountProfile.model_validate({**_PROFILE_FULL, "image": url})
        assert p.image == url

    def test_image_absent_defaults_to_none(self) -> None:
        """image defaults to None when the key is missing from the payload."""
        payload = {k: v for k, v in _PROFILE_FULL.items() if k != "image"}
        p = AccountProfile.model_validate(payload)
        assert p.image is None

    def test_all_valid_tier_values_accepted(self) -> None:
        """Every tier value documented in the API spec is accepted."""
        valid_tiers = [
            "anonymous", "microbe", "spore", "seed", "flower", "nectar", "router",
        ]
        for tier in valid_tiers:
            p = AccountProfile.model_validate({**_PROFILE_FULL, "tier": tier})
            assert p.tier == tier

    def test_unknown_tier_raises_validation_error(self) -> None:
        """An undocumented tier value raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountProfile.model_validate({**_PROFILE_FULL, "tier": "ultrabot"})

    def test_missing_required_tier_raises(self) -> None:
        """Omitting the required tier field raises ValidationError."""
        payload = {k: v for k, v in _PROFILE_FULL.items() if k != "tier"}
        with pytest.raises(ValidationError):
            AccountProfile.model_validate(payload)

    def test_missing_required_created_at_raises(self) -> None:
        """Omitting the required createdAt field raises ValidationError."""
        payload = {k: v for k, v in _PROFILE_FULL.items() if k != "createdAt"}
        with pytest.raises(ValidationError):
            AccountProfile.model_validate(payload)

    def test_extra_fields_are_preserved(self) -> None:
        """Unexpected backend fields are kept via extra='allow'."""
        p = AccountProfile.model_validate({**_PROFILE_FULL, "future_field": "beta"})
        assert p.model_extra.get("future_field") == "beta"

    def test_populate_by_snake_case_name(self) -> None:
        """Model can be constructed using Python snake_case names (populate_by_name=True)."""
        p = AccountProfile.model_validate(
            {
                "name": "Bob",
                "email": None,
                "github_username": "bob-dev",
                "image": None,
                "tier": "flower",
                "created_at": "2025-03-01T00:00:00Z",
                "next_reset_at": None,
            }
        )
        assert p.github_username == "bob-dev"
        assert p.created_at == "2025-03-01T00:00:00Z"

    def test_next_reset_at_absent_defaults_to_none(self) -> None:
        """next_reset_at defaults to None when the key is missing."""
        payload = {k: v for k, v in _PROFILE_FULL.items() if k != "nextResetAt"}
        p = AccountProfile.model_validate(payload)
        assert p.next_reset_at is None


class TestAccountUsageRecord:
    """Parsing and validation of individual usage records in /account/usage."""

    def test_full_record_all_fields(self) -> None:
        """All 16 fields of a complete record are parsed correctly."""
        r = AccountUsageRecord.model_validate(_RECORD_FULL)
        assert r.timestamp == "2026-02-23 10:00:00"
        assert r.type == "generate.text"
        assert r.model == "openai"
        assert r.api_key == "sk_abc***"
        assert r.api_key_type == "secret"
        assert r.meter_source == "tier"
        assert r.input_text_tokens == 100.0
        assert r.input_cached_tokens == 20.0
        assert r.input_audio_tokens == 0.0
        assert r.input_image_tokens == 0.0
        assert r.output_text_tokens == 50.0
        assert r.output_reasoning_tokens == 10.0
        assert r.output_audio_tokens == 0.0
        assert r.output_image_tokens == 1.0
        assert r.cost_usd == pytest.approx(0.002)
        assert r.response_time_ms == pytest.approx(345.6)

    def test_nullable_fields_accept_none(self) -> None:
        """model, api_key, api_key_type, meter_source and response_time_ms accept None."""
        r = AccountUsageRecord.model_validate(_RECORD_MINIMAL)
        assert r.model is None
        assert r.api_key is None
        assert r.api_key_type is None
        assert r.meter_source is None
        assert r.response_time_ms is None

    def test_token_fields_default_to_zero(self) -> None:
        """Token and cost fields default to 0.0 when absent from the payload."""
        r = AccountUsageRecord.model_validate(
            {"timestamp": "2026-02-01 00:00:00", "type": "generate.image"}
        )
        assert r.input_text_tokens == 0.0
        assert r.input_cached_tokens == 0.0
        assert r.input_audio_tokens == 0.0
        assert r.input_image_tokens == 0.0
        assert r.output_text_tokens == 0.0
        assert r.output_reasoning_tokens == 0.0
        assert r.output_audio_tokens == 0.0
        assert r.output_image_tokens == 0.0
        assert r.cost_usd == 0.0
        assert r.response_time_ms is None

    def test_integer_tokens_coerced_to_float(self) -> None:
        """JSON integers are accepted and stored as float (spec uses number)."""
        r = AccountUsageRecord.model_validate({**_RECORD_FULL, "input_text_tokens": 42})
        assert r.input_text_tokens == 42.0
        assert isinstance(r.input_text_tokens, float)

    def test_all_known_api_key_types_accepted(self) -> None:
        """Both documented api_key_type values are accepted."""
        for key_type in ("secret", "publishable"):
            r = AccountUsageRecord.model_validate({**_RECORD_FULL, "api_key_type": key_type})
            assert r.api_key_type == key_type

    def test_all_known_meter_sources_accepted(self) -> None:
        """All documented meter_source values are accepted."""
        for source in ("tier", "pack", "crypto"):
            r = AccountUsageRecord.model_validate({**_RECORD_FULL, "meter_source": source})
            assert r.meter_source == source

    def test_unknown_api_key_type_accepted_without_error(self) -> None:
        """A future api_key_type value is silently accepted (str, not Literal)."""
        r = AccountUsageRecord.model_validate({**_RECORD_FULL, "api_key_type": "enterprise"})
        assert r.api_key_type == "enterprise"

    def test_unknown_meter_source_accepted_without_error(self) -> None:
        """A future meter_source value is silently accepted (str, not Literal)."""
        r = AccountUsageRecord.model_validate({**_RECORD_FULL, "meter_source": "sponsorship"})
        assert r.meter_source == "sponsorship"

    def test_image_generation_type_accepted(self) -> None:
        """type='generate.image' is a valid request type."""
        r = AccountUsageRecord.model_validate({**_RECORD_FULL, "type": "generate.image"})
        assert r.type == "generate.image"

    def test_extra_fields_preserved(self) -> None:
        """Unexpected billing fields added by the backend are kept via extra='allow'."""
        r = AccountUsageRecord.model_validate({**_RECORD_FULL, "new_billing_field": 99})
        assert r.model_extra.get("new_billing_field") == 99

    def test_missing_required_timestamp_raises(self) -> None:
        """Omitting the required timestamp field raises ValidationError."""
        payload = {k: v for k, v in _RECORD_FULL.items() if k != "timestamp"}
        with pytest.raises(ValidationError):
            AccountUsageRecord.model_validate(payload)

    def test_missing_required_type_raises(self) -> None:
        """Omitting the required type field raises ValidationError."""
        payload = {k: v for k, v in _RECORD_FULL.items() if k != "type"}
        with pytest.raises(ValidationError):
            AccountUsageRecord.model_validate(payload)

    def test_output_image_tokens_for_image_request(self) -> None:
        """output_image_tokens is 1.0 per image generated, as documented."""
        r = AccountUsageRecord.model_validate(_RECORD_MINIMAL)
        assert r.output_image_tokens == 1.0


class TestAccountUsageResponse:
    """Parsing of the full /account/usage JSON envelope."""

    def test_full_response_with_two_records(self) -> None:
        """Envelope with two records is parsed into count and a typed list."""
        resp = AccountUsageResponse.model_validate(_USAGE_RESPONSE)
        assert resp.count == 2
        assert len(resp.usage) == 2

    def test_records_are_account_usage_record_instances(self) -> None:
        """Each item in usage is an AccountUsageRecord instance."""
        resp = AccountUsageResponse.model_validate(_USAGE_RESPONSE)
        for record in resp.usage:
            assert isinstance(record, AccountUsageRecord)

    def test_first_record_fields_preserved(self) -> None:
        """Fields of the first record survive nested parsing."""
        resp = AccountUsageResponse.model_validate(_USAGE_RESPONSE)
        assert resp.usage[0].model == "openai"
        assert resp.usage[0].cost_usd == pytest.approx(0.002)

    def test_second_record_nullable_fields(self) -> None:
        """Nullable fields of the second record survive nested parsing."""
        resp = AccountUsageResponse.model_validate(_USAGE_RESPONSE)
        assert resp.usage[1].model is None
        assert resp.usage[1].response_time_ms is None

    def test_empty_usage_list_is_valid(self) -> None:
        """An empty usage list is a valid response (e.g. new account)."""
        resp = AccountUsageResponse.model_validate({"usage": [], "count": 0})
        assert resp.count == 0
        assert resp.usage == []

    def test_single_record(self) -> None:
        """A response with exactly one record is handled correctly."""
        resp = AccountUsageResponse.model_validate({"usage": [_RECORD_FULL], "count": 1})
        assert resp.count == 1
        assert len(resp.usage) == 1

    def test_extra_envelope_fields_preserved(self) -> None:
        """Unexpected top-level envelope fields are preserved via extra='allow'."""
        payload = {**_USAGE_RESPONSE, "cursor": "next_page_token"}
        resp = AccountUsageResponse.model_validate(payload)
        assert resp.model_extra.get("cursor") == "next_page_token"

    def test_missing_count_raises(self) -> None:
        """Omitting the required count field raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageResponse.model_validate({"usage": []})

    def test_missing_usage_raises(self) -> None:
        """Omitting the required usage list raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageResponse.model_validate({"count": 0})


class TestAccountUsageParams:
    """Validation of query parameters for GET /account/usage."""

    def test_defaults_match_api_spec(self) -> None:
        """Default values are json, 100 records, no cursor."""
        p = AccountUsageParams()
        assert p.format == "json"
        assert p.limit == 100
        assert p.before is None

    def test_csv_format_accepted(self) -> None:
        """format='csv' is a valid value."""
        assert AccountUsageParams(format="csv").format == "csv"

    def test_limit_lower_boundary(self) -> None:
        """Minimum limit value of 1 is accepted."""
        assert AccountUsageParams(limit=1).limit == 1

    def test_limit_upper_boundary(self) -> None:
        """Maximum limit value of 50000 is accepted."""
        assert AccountUsageParams(limit=50000).limit == 50000

    def test_limit_zero_raises(self) -> None:
        """limit=0 is below the minimum and raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageParams(limit=0)

    def test_limit_above_max_raises(self) -> None:
        """limit=50001 exceeds the maximum and raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageParams(limit=50001)

    def test_before_cursor_stored(self) -> None:
        """The before pagination cursor is stored verbatim."""
        p = AccountUsageParams(before="2026-02-01T00:00:00Z")
        assert p.before == "2026-02-01T00:00:00Z"

    def test_unknown_field_raises(self) -> None:
        """Extra fields raise ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError):
            AccountUsageParams(unknown_param="x")

    def test_model_dump_excludes_none_before(self) -> None:
        """model_dump(exclude_none=True) omits before when it is None."""
        d = AccountUsageParams().model_dump(exclude_none=True)
        assert "before" not in d
        assert d == {"format": "json", "limit": 100}

    def test_model_dump_includes_before_when_set(self) -> None:
        """model_dump(exclude_none=True) includes before when it has a value."""
        d = AccountUsageParams(before="cursor_xyz").model_dump(exclude_none=True)
        assert d["before"] == "cursor_xyz"

    def test_invalid_format_raises(self) -> None:
        """An unsupported format value raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageParams(format="xml")  # type: ignore[arg-type]


class TestAccountUsageDailyParams:
    """Validation of query parameters for GET /account/usage/daily."""

    def test_default_format_is_json(self) -> None:
        """Default format is json."""
        assert AccountUsageDailyParams().format == "json"

    def test_csv_format_accepted(self) -> None:
        """format='csv' is accepted."""
        assert AccountUsageDailyParams(format="csv").format == "csv"

    def test_invalid_format_raises(self) -> None:
        """An unsupported format value raises ValidationError."""
        with pytest.raises(ValidationError):
            AccountUsageDailyParams(format="xml")  # type: ignore[arg-type]

    def test_extra_field_raises(self) -> None:
        """Extra fields raise ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError):
            AccountUsageDailyParams(limit=10)  # type: ignore[call-arg]

    def test_model_dump(self) -> None:
        """model_dump returns only the format key."""
        assert AccountUsageDailyParams().model_dump(exclude_none=True) == {"format": "json"}


class TestParseResponse:
    """All content-type branches of the static _parse_response method."""

    def test_json_content_type_returns_dict(self) -> None:
        """application/json → dict."""
        result = AccountInformation._parse_response(_json_resp({"key": "val"}))
        assert isinstance(result, dict)
        assert result["key"] == "val"

    def test_csv_content_type_returns_str(self) -> None:
        """text/csv → str."""
        result = AccountInformation._parse_response(_csv_resp("a,b\n1,2"))
        assert isinstance(result, str)
        assert "a,b" in result

    def test_plain_text_content_type_returns_str(self) -> None:
        """text/plain → str."""
        result = AccountInformation._parse_response(_plain_resp("hello"))
        assert isinstance(result, str)
        assert result == "hello"

    def test_unknown_content_type_json_parseable_returns_dict(self) -> None:
        """Unknown content-type with parseable JSON body → dict (fallback)."""
        result = AccountInformation._parse_response(_unknown_json_resp({"x": 1}))
        assert isinstance(result, dict)
        assert result["x"] == 1

    def test_unknown_content_type_non_json_returns_str(self) -> None:
        """Unknown content-type with non-JSON body → str (fallback)."""
        result = AccountInformation._parse_response(_unknown_plain_resp("raw text"))
        assert isinstance(result, str)
        assert result == "raw text"

    def test_json_content_type_with_charset_still_parsed(self) -> None:
        """application/json; charset=utf-8 is correctly detected as JSON."""
        resp = MagicMock()
        resp.headers = {"content-type": "application/json; charset=utf-8"}
        resp.json.return_value = {"n": 42}
        result = AccountInformation._parse_response(resp)
        assert result == {"n": 42}


class TestAccountInformationSync:
    """Synchronous methods of AccountInformation."""

    # --- get_profile ---

    def test_get_profile_returns_account_profile_instance(self, account) -> None:
        """get_profile() returns a typed AccountProfile, not a raw dict."""
        info, http = account
        http.get.return_value = _json_resp(_PROFILE_FULL)
        result = info.get_profile()
        assert isinstance(result, AccountProfile)

    def test_get_profile_fields_correctly_mapped(self, account) -> None:
        """get_profile() correctly maps all camelCase aliases."""
        info, http = account
        http.get.return_value = _json_resp(_PROFILE_FULL)
        result = info.get_profile()
        assert result.name == "Alice"
        assert result.github_username == "alice-gh"
        assert result.image == "https://avatars.githubusercontent.com/u/99999"
        assert result.tier == "seed"

    def test_get_profile_null_fields_handled(self, account) -> None:
        """get_profile() handles a profile where all optional fields are null."""
        info, http = account
        http.get.return_value = _json_resp(_PROFILE_ALL_NULL)
        result = info.get_profile()
        assert result.name is None
        assert result.image is None
        assert result.next_reset_at is None

    def test_get_profile_calls_correct_path(self, account) -> None:
        """get_profile() issues GET /account/profile with no extra params."""
        info, http = account
        http.get.return_value = _json_resp(_PROFILE_FULL)
        info.get_profile()
        http.get.assert_called_once_with("/account/profile")

    # --- get_balance ---

    def test_get_balance_returns_dict(self, account) -> None:
        """get_balance() returns the raw JSON dict."""
        info, http = account
        http.get.return_value = _json_resp({"balance": 42.5})
        result = info.get_balance()
        assert isinstance(result, dict)
        assert result["balance"] == 42.5

    def test_get_balance_calls_correct_path(self, account) -> None:
        """get_balance() issues GET /account/balance."""
        info, http = account
        http.get.return_value = _json_resp({"balance": 0})
        info.get_balance()
        http.get.assert_called_once_with("/account/balance")

    # --- get_key ---

    def test_get_key_returns_dict(self, account) -> None:
        """get_key() returns the raw JSON dict."""
        info, http = account
        http.get.return_value = _json_resp({"valid": True, "type": "secret"})
        result = info.get_key()
        assert isinstance(result, dict)
        assert result["valid"] is True

    def test_get_key_calls_correct_path(self, account) -> None:
        """get_key() issues GET /account/key."""
        info, http = account
        http.get.return_value = _json_resp({"valid": True})
        info.get_key()
        http.get.assert_called_once_with("/account/key")

    # --- get_usage (JSON) ---

    def test_get_usage_json_returns_account_usage_response(self, account) -> None:
        """get_usage() with default JSON format returns AccountUsageResponse."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        result = info.get_usage()
        assert isinstance(result, AccountUsageResponse)

    def test_get_usage_json_count_and_records(self, account) -> None:
        """get_usage() result has correct count and typed records."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        result = info.get_usage()
        assert isinstance(result, AccountUsageResponse)
        assert result.count == 2
        assert len(result.usage) == 2
        assert all(isinstance(r, AccountUsageRecord) for r in result.usage)

    def test_get_usage_json_empty_response(self, account) -> None:
        """get_usage() handles an empty usage list correctly."""
        info, http = account
        http.get.return_value = _json_resp({"usage": [], "count": 0})
        result = info.get_usage()
        assert isinstance(result, AccountUsageResponse)
        assert result.count == 0
        assert result.usage == []

    def test_get_usage_calls_correct_path_with_default_params(self, account) -> None:
        """get_usage() passes format=json and limit=100 to the HTTP client."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        info.get_usage()
        http.get.assert_called_once_with(
            "/account/usage",
            params={"format": "json", "limit": 100},
        )

    def test_get_usage_passes_custom_limit(self, account) -> None:
        """Custom limit is forwarded to the HTTP client."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        info.get_usage(AccountUsageParams(limit=25))
        _, kwargs = http.get.call_args
        assert kwargs["params"]["limit"] == 25

    def test_get_usage_passes_before_cursor(self, account) -> None:
        """before cursor is included in the query params when provided."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        info.get_usage(AccountUsageParams(before="cursor_abc"))
        _, kwargs = http.get.call_args
        assert kwargs["params"]["before"] == "cursor_abc"

    def test_get_usage_default_params_omit_before(self, account) -> None:
        """before is absent from query params when not provided."""
        info, http = account
        http.get.return_value = _json_resp(_USAGE_RESPONSE)
        info.get_usage()
        _, kwargs = http.get.call_args
        assert "before" not in kwargs.get("params", {})

    # --- get_usage (CSV) ---

    def test_get_usage_csv_returns_str(self, account) -> None:
        """get_usage() with format='csv' returns a plain string."""
        info, http = account
        csv = "timestamp,type,model\n2026-02-23 10:00:00,generate.text,openai"
        http.get.return_value = _csv_resp(csv)
        result = info.get_usage(AccountUsageParams(format="csv"))
        assert isinstance(result, str)
        assert "timestamp" in result

    def test_get_usage_csv_passes_format_param(self, account) -> None:
        """format=csv is forwarded in the query params."""
        info, http = account
        http.get.return_value = _csv_resp("a,b")
        info.get_usage(AccountUsageParams(format="csv"))
        _, kwargs = http.get.call_args
        assert kwargs["params"]["format"] == "csv"

    # --- get_usage_daily ---

    def test_get_usage_daily_json_returns_dict(self, account) -> None:
        """get_usage_daily() with JSON returns a dict (not yet a typed model)."""
        info, http = account
        http.get.return_value = _json_resp(_DAILY_RESPONSE)
        result = info.get_usage_daily()
        assert isinstance(result, dict)
        assert "usage" in result

    def test_get_usage_daily_csv_returns_str(self, account) -> None:
        """get_usage_daily() with format='csv' returns a string."""
        info, http = account
        http.get.return_value = _csv_resp("date,requests\n2026-02-23,5")
        result = info.get_usage_daily(AccountUsageDailyParams(format="csv"))
        assert isinstance(result, str)

    def test_get_usage_daily_calls_correct_path(self, account) -> None:
        """get_usage_daily() issues GET /account/usage/daily with format param."""
        info, http = account
        http.get.return_value = _json_resp(_DAILY_RESPONSE)
        info.get_usage_daily()
        http.get.assert_called_once_with(
            "/account/usage/daily",
            params={"format": "json"},
        )


class TestAccountInformationAsync:
    """Asynchronous methods of AccountInformation."""

    # --- aget_profile ---

    @pytest.mark.asyncio
    async def test_aget_profile_returns_account_profile_instance(self, account) -> None:
        """aget_profile() returns a typed AccountProfile, not a raw dict."""
        info, http = account
        http.aget.return_value = _json_resp(_PROFILE_FULL)
        result = await info.aget_profile()
        assert isinstance(result, AccountProfile)

    @pytest.mark.asyncio
    async def test_aget_profile_fields_correctly_mapped(self, account) -> None:
        """aget_profile() correctly maps all camelCase aliases."""
        info, http = account
        http.aget.return_value = _json_resp(_PROFILE_FULL)
        result = await info.aget_profile()
        assert result.name == "Alice"
        assert result.github_username == "alice-gh"
        assert result.image == "https://avatars.githubusercontent.com/u/99999"
        assert result.tier == "seed"

    @pytest.mark.asyncio
    async def test_aget_profile_null_fields_handled(self, account) -> None:
        """aget_profile() handles a profile where all optional fields are null."""
        info, http = account
        http.aget.return_value = _json_resp(_PROFILE_ALL_NULL)
        result = await info.aget_profile()
        assert result.name is None
        assert result.image is None
        assert result.next_reset_at is None

    @pytest.mark.asyncio
    async def test_aget_profile_calls_correct_path(self, account) -> None:
        """aget_profile() issues GET /account/profile."""
        info, http = account
        http.aget.return_value = _json_resp(_PROFILE_FULL)
        await info.aget_profile()
        http.aget.assert_awaited_once_with("/account/profile")

    # --- aget_balance ---

    @pytest.mark.asyncio
    async def test_aget_balance_returns_dict(self, account) -> None:
        """aget_balance() returns the raw JSON dict."""
        info, http = account
        http.aget.return_value = _json_resp({"balance": 100.0})
        result = await info.aget_balance()
        assert isinstance(result, dict)
        assert result["balance"] == 100.0

    @pytest.mark.asyncio
    async def test_aget_balance_calls_correct_path(self, account) -> None:
        """aget_balance() issues GET /account/balance."""
        info, http = account
        http.aget.return_value = _json_resp({"balance": 0})
        await info.aget_balance()
        http.aget.assert_awaited_once_with("/account/balance")

    # --- aget_key ---

    @pytest.mark.asyncio
    async def test_aget_key_returns_dict(self, account) -> None:
        """aget_key() returns the raw JSON dict."""
        info, http = account
        http.aget.return_value = _json_resp({"valid": True, "type": "secret"})
        result = await info.aget_key()
        assert isinstance(result, dict)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_aget_key_calls_correct_path(self, account) -> None:
        """aget_key() issues GET /account/key."""
        info, http = account
        http.aget.return_value = _json_resp({"valid": True})
        await info.aget_key()
        http.aget.assert_awaited_once_with("/account/key")

    # --- aget_usage (JSON) ---

    @pytest.mark.asyncio
    async def test_aget_usage_json_returns_account_usage_response(self, account) -> None:
        """aget_usage() with default JSON format returns AccountUsageResponse."""
        info, http = account
        http.aget.return_value = _json_resp(_USAGE_RESPONSE)
        result = await info.aget_usage()
        assert isinstance(result, AccountUsageResponse)

    @pytest.mark.asyncio
    async def test_aget_usage_json_count_and_records(self, account) -> None:
        """aget_usage() result has correct count and typed records."""
        info, http = account
        http.aget.return_value = _json_resp(_USAGE_RESPONSE)
        result = await info.aget_usage()
        assert isinstance(result, AccountUsageResponse)
        assert result.count == 2
        assert all(isinstance(r, AccountUsageRecord) for r in result.usage)

    @pytest.mark.asyncio
    async def test_aget_usage_json_empty_response(self, account) -> None:
        """aget_usage() handles an empty usage list correctly."""
        info, http = account
        http.aget.return_value = _json_resp({"usage": [], "count": 0})
        result = await info.aget_usage()
        assert isinstance(result, AccountUsageResponse)
        assert result.count == 0
        assert result.usage == []

    @pytest.mark.asyncio
    async def test_aget_usage_calls_correct_path_with_default_params(self, account) -> None:
        """aget_usage() passes format=json and limit=100 to the HTTP client."""
        info, http = account
        http.aget.return_value = _json_resp(_USAGE_RESPONSE)
        await info.aget_usage()
        http.aget.assert_awaited_once_with(
            "/account/usage",
            params={"format": "json", "limit": 100},
        )

    @pytest.mark.asyncio
    async def test_aget_usage_passes_custom_limit_and_before(self, account) -> None:
        """aget_usage() forwards limit and before cursor to the HTTP client."""
        info, http = account
        http.aget.return_value = _json_resp(_USAGE_RESPONSE)
        await info.aget_usage(AccountUsageParams(limit=10, before="tok_xyz"))
        http.aget.assert_awaited_once_with(
            "/account/usage",
            params={"format": "json", "limit": 10, "before": "tok_xyz"},
        )

    # --- aget_usage (CSV) ---

    @pytest.mark.asyncio
    async def test_aget_usage_csv_returns_str(self, account) -> None:
        """aget_usage() with format='csv' returns a plain string."""
        info, http = account
        csv = "timestamp,type,cost_usd\n2026-02-23 10:00:00,generate.text,0.002"
        http.aget.return_value = _csv_resp(csv)
        result = await info.aget_usage(AccountUsageParams(format="csv"))
        assert isinstance(result, str)
        assert "timestamp" in result

    # --- aget_usage_daily ---

    @pytest.mark.asyncio
    async def test_aget_usage_daily_json_returns_dict(self, account) -> None:
        """aget_usage_daily() with JSON returns a dict."""
        info, http = account
        http.aget.return_value = _json_resp(_DAILY_RESPONSE)
        result = await info.aget_usage_daily()
        assert isinstance(result, dict)
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_aget_usage_daily_csv_returns_str(self, account) -> None:
        """aget_usage_daily() with format='csv' returns a string."""
        info, http = account
        http.aget.return_value = _csv_resp("date,requests\n2026-02-23,10")
        result = await info.aget_usage_daily(AccountUsageDailyParams(format="csv"))
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_aget_usage_daily_calls_correct_path(self, account) -> None:
        """aget_usage_daily() issues GET /account/usage/daily with format param."""
        info, http = account
        http.aget.return_value = _json_resp(_DAILY_RESPONSE)
        await info.aget_usage_daily()
        http.aget.assert_awaited_once_with(
            "/account/usage/daily",
            params={"format": "json"},
        )
