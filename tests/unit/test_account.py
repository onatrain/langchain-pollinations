import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx

from langchain_pollinations.account import (
    AccountInformation,
    AccountUsageParams,
    AccountUsageDailyParams,
)


@pytest.fixture
def mock_auth():
    """Mock AuthConfig.from_env_or_value"""
    with patch("langchain_pollinations.account.AuthConfig.from_env_or_value") as mock:
        mock.return_value = Mock(api_key="test-api-key")
        yield mock


@pytest.fixture
def account(mock_auth):
    """AccountInformation instance con auth mockeado"""
    return AccountInformation(api_key="test-key")


@pytest.fixture
def mock_response_json():
    """Mock httpx.Response para JSON"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "application/json"}
    response.json.return_value = {"status": "ok", "data": "test"}
    response.text = '{"status": "ok", "data": "test"}'
    return response


@pytest.fixture
def mock_response_csv():
    """Mock httpx.Response para CSV"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "text/csv"}
    response.text = "date,model,cost\n2026-02-14,openai,100\n2026-02-13,claude,200"
    response.json.side_effect = ValueError("Not JSON")
    return response


@pytest.fixture
def mock_response_plain():
    """Mock httpx.Response para text/plain"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "text/plain"}
    response.text = "plain text response"
    response.json.side_effect = ValueError("Not JSON")
    return response


@pytest.fixture
def mock_response_unknown():
    """Mock httpx.Response con Content-Type desconocido"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "application/xml"}
    response.json.return_value = {"fallback": "json"}
    response.text = "<xml>data</xml>"
    return response


def test_init_with_apikey(mock_auth):
    """Test inicialización con API key explícito"""
    account = AccountInformation(api_key="my-key")
    assert account.api_key == "my-key"
    assert account.base_url == "https://gen.pollinations.ai"
    assert account.timeout_s == 120.0
    mock_auth.assert_called_once_with("my-key")


def test_init_without_apikey(mock_auth):
    """Test inicialización sin API key (usa env)"""
    account = AccountInformation()
    assert account.api_key is None
    mock_auth.assert_called_once_with(None)


def test_init_custom_baseurl(mock_auth):
    """Test inicialización con baseurl custom"""
    account = AccountInformation(base_url="https://custom.api.com")
    assert account.base_url == "https://custom.api.com"


def test_init_custom_timeout(mock_auth):
    """Test inicialización con timeout custom"""
    account = AccountInformation(timeout_s=60.0)
    assert account.timeout_s == 60.0


def test_parse_response_json(mock_response_json):
    """Test parsing de respuesta JSON"""
    result = AccountInformation._parse_response(mock_response_json)
    assert isinstance(result, dict)
    assert result == {"status": "ok", "data": "test"}


def test_parse_response_csv(mock_response_csv):
    """Test parsing de respuesta CSV"""
    result = AccountInformation._parse_response(mock_response_csv)
    assert isinstance(result, str)
    assert "date,model,cost" in result
    assert "2026-02-14,openai,100" in result


def test_parse_response_plain_text(mock_response_plain):
    """Test parsing de respuesta text/plain"""
    result = AccountInformation._parse_response(mock_response_plain)
    assert isinstance(result, str)
    assert result == "plain text response"


def test_parse_response_unknown_content_type_fallback_json(mock_response_unknown):
    """Test fallback a JSON con Content-Type desconocido"""
    result = AccountInformation._parse_response(mock_response_unknown)
    assert isinstance(result, dict)
    assert result == {"fallback": "json"}


def test_parse_response_unknown_content_type_fallback_text():
    """Test fallback a texto cuando JSON falla"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "application/unknown"}
    response.json.side_effect = Exception("Cannot parse JSON")
    response.text = "fallback text"

    result = AccountInformation._parse_response(response)
    assert isinstance(result, str)
    assert result == "fallback text"


def test_parse_response_no_content_type_header():
    """Test con header Content-Type faltante"""
    response = Mock(spec=httpx.Response)
    response.headers = {}
    response.json.return_value = {"data": "test"}
    response.text = '{"data": "test"}'

    result = AccountInformation._parse_response(response)
    assert isinstance(result, dict)
    assert result == {"data": "test"}


def test_parse_response_content_type_case_insensitive():
    """Test que Content-Type sea case-insensitive"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "APPLICATION/JSON"}
    response.json.return_value = {"test": "ok"}

    result = AccountInformation._parse_response(response)
    assert isinstance(result, dict)


def test_get_profile_success(account, mock_response_json):
    """Test get_profile exitoso"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_profile()

    assert result == {"status": "ok", "data": "test"}
    account._http.get.assert_called_once_with("/account/profile")


def test_get_profile_returns_dict(account, mock_response_json):
    """Test que get_profile siempre retorna dict"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_profile()

    assert isinstance(result, dict)


def test_get_balance_success(account, mock_response_json):
    """Test get_balance exitoso"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_balance()

    assert result == {"status": "ok", "data": "test"}
    account._http.get.assert_called_once_with("/account/balance")


def test_get_balance_returns_dict(account, mock_response_json):
    """Test que get_balance siempre retorna dict"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_balance()

    assert isinstance(result, dict)


def test_get_key_success(account, mock_response_json):
    """Test get_key exitoso"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_key()

    assert result == {"status": "ok", "data": "test"}
    account._http.get.assert_called_once_with("/account/key")


def test_get_key_returns_dict(account, mock_response_json):
    """Test que get_key siempre retorna dict"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_key()

    assert isinstance(result, dict)


def test_get_usage_without_params_returns_json(account, mock_response_json):
    """Test get_usage sin parámetros (default JSON)"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_usage()

    assert isinstance(result, dict)
    assert result == {"status": "ok", "data": "test"}
    account._http.get.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 100}
    )


def test_get_usage_with_json_format_returns_dict(account, mock_response_json):
    """Test get_usage con format=json explícito"""
    account._http.get = Mock(return_value=mock_response_json)
    params = AccountUsageParams(format="json", limit=50)

    result = account.get_usage(params)

    assert isinstance(result, dict)
    account._http.get.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 50}
    )


def test_get_usage_with_csv_format_returns_str(account, mock_response_csv):
    """Test get_usage con format=csv"""
    account._http.get = Mock(return_value=mock_response_csv)
    params = AccountUsageParams(format="csv", limit=100)

    result = account.get_usage(params)

    assert isinstance(result, str)
    assert "date,model,cost" in result
    account._http.get.assert_called_once_with(
        "/account/usage",
        params={"format": "csv", "limit": 100}
    )


def test_get_usage_with_custom_limit(account, mock_response_json):
    """Test get_usage con limit personalizado"""
    account._http.get = Mock(return_value=mock_response_json)
    params = AccountUsageParams(limit=500)

    result = account.get_usage(params)

    account._http.get.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 500}
    )


def test_get_usage_with_before_param(account, mock_response_json):
    """Test get_usage con parámetro before"""
    account._http.get = Mock(return_value=mock_response_json)
    params = AccountUsageParams(before="2026-02-14")

    result = account.get_usage(params)

    account._http.get.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 100, "before": "2026-02-14"}
    )


def test_get_usage_params_exclude_none(account, mock_response_json):
    """Test que parámetros None se excluyen"""
    account._http.get = Mock(return_value=mock_response_json)
    params = AccountUsageParams(before=None)

    result = account.get_usage(params)

    called_params = account._http.get.call_args[1]["params"]
    assert "before" not in called_params


def test_get_usage_daily_without_params_returns_json(account, mock_response_json):
    """Test get_usage_daily sin parámetros (default JSON)"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_usage_daily()

    assert isinstance(result, dict)
    account._http.get.assert_called_once_with(
        "/account/usage/daily",
        params={"format": "json"}
    )


def test_get_usage_daily_with_json_format(account, mock_response_json):
    """Test get_usage_daily con format=json"""
    account._http.get = Mock(return_value=mock_response_json)
    params = AccountUsageDailyParams(format="json")

    result = account.get_usage_daily(params)

    assert isinstance(result, dict)


def test_get_usage_daily_with_csv_format_returns_str(account, mock_response_csv):
    """Test get_usage_daily con format=csv"""
    account._http.get = Mock(return_value=mock_response_csv)
    params = AccountUsageDailyParams(format="csv")

    result = account.get_usage_daily(params)

    assert isinstance(result, str)
    assert "date,model,cost" in result
    account._http.get.assert_called_once_with(
        "/account/usage/daily",
        params={"format": "csv"}
    )


@pytest.mark.asyncio
async def test_aget_profile_success(account, mock_response_json):
    """Test aget_profile exitoso"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_profile()

    assert result == {"status": "ok", "data": "test"}
    account._http.aget.assert_called_once_with("/account/profile")


@pytest.mark.asyncio
async def test_aget_profile_returns_dict(account, mock_response_json):
    """Test que aget_profile siempre retorna dict"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_profile()

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_aget_balance_success(account, mock_response_json):
    """Test aget_balance exitoso"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_balance()

    assert result == {"status": "ok", "data": "test"}
    account._http.aget.assert_called_once_with("/account/balance")


@pytest.mark.asyncio
async def test_aget_balance_returns_dict(account, mock_response_json):
    """Test que aget_balance siempre retorna dict"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_balance()

    assert isinstance(result, dict)


# ============================================================================
# TESTS: ASYNC get_key
# ============================================================================

@pytest.mark.asyncio
async def test_aget_key_success(account, mock_response_json):
    """Test aget_key exitoso"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_key()

    assert result == {"status": "ok", "data": "test"}
    account._http.aget.assert_called_once_with("/account/key")


@pytest.mark.asyncio
async def test_aget_key_returns_dict(account, mock_response_json):
    """Test que aget_key siempre retorna dict"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_key()

    assert isinstance(result, dict)


# ============================================================================
# TESTS: ASYNC get_usage
# ============================================================================

@pytest.mark.asyncio
async def test_aget_usage_without_params_returns_json(account, mock_response_json):
    """Test aget_usage sin parámetros (default JSON)"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_usage()

    assert isinstance(result, dict)
    assert result == {"status": "ok", "data": "test"}
    account._http.aget.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 100}
    )


@pytest.mark.asyncio
async def test_aget_usage_with_json_format_returns_dict(account, mock_response_json):
    """Test aget_usage con format=json"""
    account._http.aget = AsyncMock(return_value=mock_response_json)
    params = AccountUsageParams(format="json", limit=50)

    result = await account.aget_usage(params)

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_aget_usage_with_csv_format_returns_str(account, mock_response_csv):
    """Test aget_usage con format=csv"""
    account._http.aget = AsyncMock(return_value=mock_response_csv)
    params = AccountUsageParams(format="csv")

    result = await account.aget_usage(params)

    assert isinstance(result, str)
    assert "date,model,cost" in result


@pytest.mark.asyncio
async def test_aget_usage_with_custom_limit(account, mock_response_json):
    """Test aget_usage con limit personalizado"""
    account._http.aget = AsyncMock(return_value=mock_response_json)
    params = AccountUsageParams(limit=1000)

    result = await account.aget_usage(params)

    account._http.aget.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 1000}
    )


@pytest.mark.asyncio
async def test_aget_usage_with_before_param(account, mock_response_json):
    """Test aget_usage con parámetro before"""
    account._http.aget = AsyncMock(return_value=mock_response_json)
    params = AccountUsageParams(before="2026-02-14", limit=200)

    result = await account.aget_usage(params)

    account._http.aget.assert_called_once_with(
        "/account/usage",
        params={"format": "json", "limit": 200, "before": "2026-02-14"}
    )


# ============================================================================
# TESTS: ASYNC get_usage_daily
# ============================================================================

@pytest.mark.asyncio
async def test_aget_usage_daily_without_params_returns_json(account, mock_response_json):
    """Test aget_usage_daily sin parámetros (default JSON)"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    result = await account.aget_usage_daily()

    assert isinstance(result, dict)
    account._http.aget.assert_called_once_with(
        "/account/usage/daily",
        params={"format": "json"}
    )


@pytest.mark.asyncio
async def test_aget_usage_daily_with_json_format(account, mock_response_json):
    """Test aget_usage_daily con format=json"""
    account._http.aget = AsyncMock(return_value=mock_response_json)
    params = AccountUsageDailyParams(format="json")

    result = await account.aget_usage_daily(params)

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_aget_usage_daily_with_csv_format_returns_str(account, mock_response_csv):
    """Test aget_usage_daily con format=csv"""
    account._http.aget = AsyncMock(return_value=mock_response_csv)
    params = AccountUsageDailyParams(format="csv")

    result = await account.aget_usage_daily(params)

    assert isinstance(result, str)
    assert "date,model,cost" in result


# ============================================================================
# TESTS: VALIDACIÓN DE PARÁMETROS (Pydantic)
# ============================================================================

def test_account_usage_params_valid():
    """Test validación de parámetros válidos"""
    params = AccountUsageParams(format="json", limit=100, before="2026-02-14")
    assert params.format == "json"
    assert params.limit == 100
    assert params.before == "2026-02-14"


def test_account_usage_params_defaults():
    """Test valores por defecto de parámetros"""
    params = AccountUsageParams()
    assert params.format == "json"
    assert params.limit == 100
    assert params.before is None


def test_account_usage_params_invalid_format():
    """Test validación de formato inválido"""
    with pytest.raises(Exception):  # Pydantic validation error
        AccountUsageParams(format="xml")


def test_account_usage_params_limit_too_low():
    """Test validación de limit menor al mínimo"""
    with pytest.raises(Exception):  # Pydantic validation error
        AccountUsageParams(limit=0)


def test_account_usage_params_limit_too_high():
    """Test validación de limit mayor al máximo"""
    with pytest.raises(Exception):  # Pydantic validation error
        AccountUsageParams(limit=50001)


def test_account_usage_params_limit_boundaries():
    """Test límites válidos de limit"""
    params_min = AccountUsageParams(limit=1)
    params_max = AccountUsageParams(limit=50000)
    assert params_min.limit == 1
    assert params_max.limit == 50000


def test_account_usage_daily_params_valid():
    """Test validación de parámetros daily válidos"""
    params = AccountUsageDailyParams(format="csv")
    assert params.format == "csv"


def test_account_usage_daily_params_defaults():
    """Test valores por defecto de parámetros daily"""
    params = AccountUsageDailyParams()
    assert params.format == "json"


def test_full_workflow_json(account, mock_response_json):
    """Test workflow completo con JSON"""
    account._http.get = Mock(return_value=mock_response_json)

    # Profile
    profile = account.get_profile()
    assert isinstance(profile, dict)

    # Balance
    balance = account.get_balance()
    assert isinstance(balance, dict)

    # Key
    key = account.get_key()
    assert isinstance(key, dict)

    # Usage JSON
    usage = account.get_usage()
    assert isinstance(usage, dict)

    # Usage daily JSON
    usage_daily = account.get_usage_daily()
    assert isinstance(usage_daily, dict)


def test_full_workflow_csv(account, mock_response_csv):
    """Test workflow completo con CSV"""
    account._http.get = Mock(return_value=mock_response_csv)

    # Usage CSV
    params_usage = AccountUsageParams(format="csv")
    usage_csv = account.get_usage(params_usage)
    assert isinstance(usage_csv, str)
    assert "," in usage_csv

    # Usage daily CSV
    params_daily = AccountUsageDailyParams(format="csv")
    daily_csv = account.get_usage_daily(params_daily)
    assert isinstance(daily_csv, str)
    assert "," in daily_csv


@pytest.mark.asyncio
async def test_async_full_workflow_json(account, mock_response_json):
    """Test workflow async completo con JSON"""
    account._http.aget = AsyncMock(return_value=mock_response_json)

    profile = await account.aget_profile()
    assert isinstance(profile, dict)

    balance = await account.aget_balance()
    assert isinstance(balance, dict)

    key = await account.aget_key()
    assert isinstance(key, dict)

    usage = await account.aget_usage()
    assert isinstance(usage, dict)

    usage_daily = await account.aget_usage_daily()
    assert isinstance(usage_daily, dict)


@pytest.mark.asyncio
async def test_async_full_workflow_csv(account, mock_response_csv):
    """Test workflow async completo con CSV"""
    account._http.aget = AsyncMock(return_value=mock_response_csv)

    params_usage = AccountUsageParams(format="csv")
    usage_csv = await account.aget_usage(params_usage)
    assert isinstance(usage_csv, str)

    params_daily = AccountUsageDailyParams(format="csv")
    daily_csv = await account.aget_usage_daily(params_daily)
    assert isinstance(daily_csv, str)


def test_get_usage_with_mixed_content_type(account):
    """Test con Content-Type que contiene charset"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "application/json; charset=utf-8"}
    response.json.return_value = {"data": "test"}
    account._http.get = Mock(return_value=response)

    result = account.get_usage()

    assert isinstance(result, dict)
    assert result == {"data": "test"}


def test_get_usage_csv_multiline(account):
    """Test CSV con múltiples líneas"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "text/csv"}
    csv_data = """date,model,cost,tokens
2026-02-14,openai,100,1500
2026-02-13,claude,200,3000
2026-02-12,gemini,150,2000"""
    response.text = csv_data
    response.json.side_effect = ValueError("Not JSON")
    account._http.get = Mock(return_value=response)

    params = AccountUsageParams(format="csv")
    result = account.get_usage(params)

    assert isinstance(result, str)
    assert result == csv_data
    lines = result.split("\n")
    assert len(lines) == 4


def test_response_with_empty_content_type(account):
    """Test con Content-Type vacío"""
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": ""}
    response.json.return_value = {"data": "test"}
    account._http.get = Mock(return_value=response)

    result = account.get_usage()

    assert isinstance(result, dict)


def test_http_error_propagates(account):
    """Test que errores HTTP se propagan"""
    from langchain_pollinations._errors import PollinationsAPIError

    account._http.get = Mock(side_effect=PollinationsAPIError(
        401,
        message="Unauthorized"
    ))

    with pytest.raises(PollinationsAPIError):
        account.get_profile()


@pytest.mark.asyncio
async def test_async_http_error_propagates(account):
    """Test que errores HTTP async se propagan"""
    from langchain_pollinations._errors import PollinationsAPIError

    account._http.aget = AsyncMock(side_effect=PollinationsAPIError(
        403,
        message="Forbidden"
    ))

    with pytest.raises(PollinationsAPIError):
        await account.aget_profile()


def test_get_usage_type_narrowing_json(account, mock_response_json):
    """Test type narrowing con isinstance para JSON"""
    account._http.get = Mock(return_value=mock_response_json)

    result = account.get_usage()

    if isinstance(result, dict):
        # Type checker sabe que result es dict aquí
        assert "status" in result
    else:
        pytest.fail("Expected dict, got str")


def test_get_usage_type_narrowing_csv(account, mock_response_csv):
    """Test type narrowing con isinstance para CSV"""
    account._http.get = Mock(return_value=mock_response_csv)
    params = AccountUsageParams(format="csv")

    result = account.get_usage(params)

    if isinstance(result, str):
        # Type checker sabe que result es str aquí
        assert len(result) > 0
    else:
        pytest.fail("Expected str, got dict")
