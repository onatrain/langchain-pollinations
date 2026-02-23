"""
Tests unitarios para manejo estructurado de errores del API.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock
from langchain_pollinations._errors import PollinationsAPIError
from langchain_pollinations._client import _parse_error_response, PollinationsHttpClient, HttpConfig


def test_error_creation_minimal():
    error = PollinationsAPIError(
        status_code=500,
        message="Internal error"
    )

    assert error.status_code == 500
    assert error.message == "Internal error"
    assert error.body is None
    assert error.error_code is None
    assert error.request_id is None
    assert error.timestamp is None
    assert error.details is None
    assert error.cause is None


def test_error_creation_full():
    error = PollinationsAPIError(
        status_code=403,
        message="Access denied",
        body='{"status":403}',
        error_code="FORBIDDEN",
        request_id="req_test123",
        timestamp="2026-02-17T00:00:00Z",
        details={"permission": "admin"},
        cause="Insufficient permissions"
    )

    assert error.status_code == 403
    assert error.message == "Access denied"
    assert error.body == '{"status":403}'
    assert error.error_code == "FORBIDDEN"
    assert error.request_id == "req_test123"
    assert error.timestamp == "2026-02-17T00:00:00Z"
    assert error.details == {"permission": "admin"}
    assert error.cause == "Insufficient permissions"


def test_str_minimal():
    error = PollinationsAPIError(status_code=400, message="Bad request")
    s = str(error)

    assert "400" in s
    assert "Bad request" in s
    assert "PollinationsAPIError" in s


def test_str_with_code_and_request_id():
    error = PollinationsAPIError(
        status_code=403,
        message="Forbidden",
        error_code="FORBIDDEN",
        request_id="req_xyz"
    )
    s = str(error)

    assert "403" in s
    assert "FORBIDDEN" in s
    assert "req_xyz" in s


def test_str_with_body():
    error = PollinationsAPIError(
        status_code=500,
        message="Error",
        body="x" * 1000
    )
    s = str(error)

    assert "1000 chars" in s


def test_repr_format():
    error = PollinationsAPIError(
        status_code=401,
        message="Unauthorized",
        error_code="UNAUTHORIZED"
    )
    r = repr(error)

    assert "PollinationsAPIError(" in r
    assert "status_code=401" in r
    assert "error_code='UNAUTHORIZED'" in r


def test_to_dict():
    error = PollinationsAPIError(
        status_code=400,
        message="Validation failed",
        error_code="BAD_REQUEST",
        request_id="req_abc",
        timestamp="2026-02-17T00:00:00Z",
        details={"field": "temperature"},
        cause="Invalid value",
        body='{"error":"..."}'
    )

    d = error.to_dict()

    assert d["status_code"] == 400
    assert d["message"] == "Validation failed"
    assert d["error_code"] == "BAD_REQUEST"
    assert d["request_id"] == "req_abc"
    assert d["timestamp"] == "2026-02-17T00:00:00Z"
    assert d["details"] == {"field": "temperature"}
    assert d["cause"] == "Invalid value"
    assert d["body"] == '{"error":"..."}'


def test_is_client_error():
    e400 = PollinationsAPIError(status_code=400, message="Bad request")
    e401 = PollinationsAPIError(status_code=401, message="Unauthorized")
    e403 = PollinationsAPIError(status_code=403, message="Forbidden")
    e404 = PollinationsAPIError(status_code=404, message="Not found")
    e499 = PollinationsAPIError(status_code=499, message="Client error")

    assert e400.is_client_error is True
    assert e401.is_client_error is True
    assert e403.is_client_error is True
    assert e404.is_client_error is True
    assert e499.is_client_error is True


def test_is_not_client_error():
    e399 = PollinationsAPIError(status_code=399, message="Unknown")
    e500 = PollinationsAPIError(status_code=500, message="Server error")

    assert e399.is_client_error is False
    assert e500.is_client_error is False


def test_is_server_error():
    e500 = PollinationsAPIError(status_code=500, message="Internal error")
    e502 = PollinationsAPIError(status_code=502, message="Bad gateway")
    e503 = PollinationsAPIError(status_code=503, message="Unavailable")
    e599 = PollinationsAPIError(status_code=599, message="Server error")

    assert e500.is_server_error is True
    assert e502.is_server_error is True
    assert e503.is_server_error is True
    assert e599.is_server_error is True


def test_is_not_server_error():
    e400 = PollinationsAPIError(status_code=400, message="Client error")
    e499 = PollinationsAPIError(status_code=499, message="Client error")
    e600 = PollinationsAPIError(status_code=600, message="Unknown")

    assert e400.is_server_error is False
    assert e499.is_server_error is False
    assert e600.is_server_error is False


def test_is_auth_error():
    e401 = PollinationsAPIError(status_code=401, message="Unauthorized")
    e403 = PollinationsAPIError(status_code=403, message="Forbidden")

    assert e401.is_auth_error is True
    assert e403.is_auth_error is True


def test_is_not_auth_error():
    e400 = PollinationsAPIError(status_code=400, message="Bad request")
    e404 = PollinationsAPIError(status_code=404, message="Not found")
    e500 = PollinationsAPIError(status_code=500, message="Server error")

    assert e400.is_auth_error is False
    assert e404.is_auth_error is False
    assert e500.is_auth_error is False


def test_is_validation_error():
    e = PollinationsAPIError(
        status_code=400,
        message="Validation failed",
        error_code="BAD_REQUEST"
    )

    assert e.is_validation_error is True


def test_is_not_validation_error_wrong_code():
    e = PollinationsAPIError(
        status_code=400,
        message="Other error",
        error_code="OTHER_ERROR"
    )

    assert e.is_validation_error is False


def test_is_not_validation_error_wrong_status():
    e = PollinationsAPIError(
        status_code=500,
        message="Error",
        error_code="BAD_REQUEST"
    )

    assert e.is_validation_error is False


def test_is_not_validation_error_no_code():
    e = PollinationsAPIError(
        status_code=400,
        message="Bad request"
    )

    assert e.is_validation_error is False


def test_parse_400_with_validation_details():
    body = json.dumps({
        "status": 400,
        "success": False,
        "error": {
            "code": "BAD_REQUEST",
            "message": "Something was wrong with the input data",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {
                "field": "temperature",
                "value": 10.0,
                "constraint": "maximum 2.0"
            },
            "requestId": "req_test123"
        }
    })

    error = _parse_error_response(400, body, "application/json")

    assert error.status_code == 400
    assert error.error_code == "BAD_REQUEST"
    assert error.message == "Something was wrong with the input data"
    assert error.request_id == "req_test123"
    assert error.timestamp == "2026-02-17T00:00:00Z"
    assert error.details is not None
    assert error.details["field"] == "temperature"
    assert error.details["value"] == 10.0
    assert error.details["constraint"] == "maximum 2.0"
    assert error.body == body


def test_parse_401_unauthorized():
    body = json.dumps({
        "status": 401,
        "success": False,
        "error": {
            "code": "UNAUTHORIZED",
            "message": "Authentication required",
            "timestamp": "2026-02-17T00:10:00Z",
            "details": {},
            "requestId": "req_auth456"
        }
    })

    error = _parse_error_response(401, body, "application/json")

    assert error.status_code == 401
    assert error.error_code == "UNAUTHORIZED"
    assert error.message == "Authentication required"
    assert error.request_id == "req_auth456"
    assert error.timestamp == "2026-02-17T00:10:00Z"


def test_parse_403_forbidden_with_details():
    body = json.dumps({
        "status": 403,
        "success": False,
        "error": {
            "code": "FORBIDDEN",
            "message": "Access denied",
            "timestamp": "2026-02-17T00:20:00Z",
            "details": {
                "required_permission": "model:gpt-4"
            },
            "requestId": "req_perm789"
        }
    })

    error = _parse_error_response(403, body, "application/json")

    assert error.status_code == 403
    assert error.error_code == "FORBIDDEN"
    assert error.message == "Access denied"
    assert error.request_id == "req_perm789"
    assert error.details["required_permission"] == "model:gpt-4"


def test_parse_500_with_cause():
    body = json.dumps({
        "status": 500,
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Oh snap, something went wrong",
            "timestamp": "2026-02-17T00:30:00Z",
            "details": {},
            "requestId": "req_server999",
            "cause": "Database connection timeout"
        }
    })

    error = _parse_error_response(500, body, "application/json")

    assert error.status_code == 500
    assert error.error_code == "INTERNAL_ERROR"
    assert error.message == "Oh snap, something went wrong"
    assert error.request_id == "req_server999"
    assert error.cause == "Database connection timeout"


def test_parse_non_json_content_type():
    body = "Internal Server Error"

    error = _parse_error_response(500, body, "text/html")

    assert error.status_code == 500
    assert error.message == "Internal Server Error"
    assert error.error_code is None
    assert error.request_id is None
    assert error.timestamp is None
    assert error.details is None
    assert error.cause is None
    assert error.body == body


def test_parse_invalid_json():
    body = "{ invalid json"

    error = _parse_error_response(400, body, "application/json")

    assert error.status_code == 400
    assert "invalid json" in error.message
    assert error.error_code is None
    assert error.request_id is None
    assert error.body == body


def test_parse_json_without_error_object():
    body = json.dumps({
        "message": "Something went wrong"
    })

    error = _parse_error_response(500, body, "application/json")

    assert error.status_code == 500
    assert error.message == "Something went wrong"
    assert error.error_code is None
    assert error.request_id is None


def test_parse_empty_body():
    error = _parse_error_response(500, "", "application/json")

    assert error.status_code == 500
    assert error.message == "HTTP error"
    assert error.error_code is None
    assert error.request_id is None


def test_parse_json_array():
    body = json.dumps(["error1", "error2"])

    error = _parse_error_response(400, body, "application/json")

    assert error.status_code == 400
    assert error.error_code is None


def test_parse_strips_whitespace():
    body = json.dumps({
        "status": 403,
        "success": False,
        "error": {
            "code": "  FORBIDDEN  ",
            "message": "  Access denied  ",
            "requestId": "  req_123  ",
            "timestamp": "  2026-02-17T00:00:00Z  ",
            "details": {}
        }
    })

    error = _parse_error_response(403, body, "application/json")

    assert error.error_code == "FORBIDDEN"
    assert error.message == "Access denied"
    assert error.request_id == "req_123"
    assert error.timestamp == "2026-02-17T00:00:00Z"


def test_parse_handles_non_string_values():
    body = json.dumps({
        "status": 400,
        "success": False,
        "error": {
            "code": 12345,
            "message": ["not", "a", "string"],
            "requestId": None,
            "timestamp": 123456789,
            "details": "not a dict"
        }
    })

    error = _parse_error_response(400, body, "application/json")

    assert error.status_code == 400
    assert error.error_code is None
    assert error.request_id is None
    assert error.details is None


def test_parse_long_body_in_message():
    body = "x" * 400

    error = _parse_error_response(500, body, "text/plain")

    assert error.status_code == 500
    assert len(error.message) == 400
    assert error.body == body


@pytest.fixture
def client():
    config = HttpConfig(base_url="https://test.api", timeout_s=30.0)
    return PollinationsHttpClient(config=config, api_key="test-key")


def test_raise_for_status_success_200(client):
    mock_response = Mock()
    mock_response.status_code = 200

    client.raise_for_status(mock_response)


def test_raise_for_status_success_201(client):
    mock_response = Mock()
    mock_response.status_code = 201

    client.raise_for_status(mock_response)


def test_raise_for_status_success_299(client):
    mock_response = Mock()
    mock_response.status_code = 299

    client.raise_for_status(mock_response)


def test_raise_for_status_400_with_json(client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = json.dumps({
        "status": 400,
        "success": False,
        "error": {
            "code": "BAD_REQUEST",
            "message": "Invalid input",
            "requestId": "req_400",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {"field": "model"}
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 400
    assert error.error_code == "BAD_REQUEST"
    assert error.message == "Invalid input"
    assert error.request_id == "req_400"
    assert error.details["field"] == "model"


def test_raise_for_status_401_unauthorized(client):
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = json.dumps({
        "status": 401,
        "success": False,
        "error": {
            "code": "UNAUTHORIZED",
            "message": "API key missing",
            "requestId": "req_401",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {}
        }
    })
    mock_response.headers = {"content-type": "application/json; charset=utf-8"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 401
    assert error.error_code == "UNAUTHORIZED"
    assert error.is_auth_error is True


def test_raise_for_status_403_forbidden(client):
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.text = json.dumps({
        "status": 403,
        "success": False,
        "error": {
            "code": "FORBIDDEN",
            "message": "Access denied",
            "requestId": "req_403",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {"required_permission": "admin"}
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 403
    assert error.error_code == "FORBIDDEN"
    assert error.is_auth_error is True


def test_raise_for_status_500_server_error(client):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = json.dumps({
        "status": 500,
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Server error",
            "requestId": "req_500",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {},
            "cause": "Database timeout"
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 500
    assert error.error_code == "INTERNAL_ERROR"
    assert error.is_server_error is True
    assert error.cause == "Database timeout"


def test_raise_for_status_non_json_error(client):
    mock_response = Mock()
    mock_response.status_code = 502
    mock_response.text = "<html>Bad Gateway</html>"
    mock_response.headers = {"content-type": "text/html"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 502
    assert error.error_code is None
    assert error.request_id is None
    assert "<html>Bad Gateway</html>" in error.message


def test_raise_for_status_text_exception(client):
    from unittest.mock import PropertyMock

    mock_response = Mock()
    mock_response.status_code = 500
    type(mock_response).text = PropertyMock(side_effect=Exception("Cannot read body"))
    mock_response.headers = {"content-type": "text/plain"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 500
    assert error.body == "" or error.body is None


def test_raise_for_status_empty_headers(client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Error"
    mock_response.headers = {}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 400


def test_raise_for_status_404_not_found(client):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = json.dumps({
        "status": 404,
        "success": False,
        "error": {
            "code": "NOT_FOUND",
            "message": "Resource not found",
            "requestId": "req_404",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {}
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.status_code == 404
    assert error.error_code == "NOT_FOUND"
    assert error.is_client_error is True


    def test_raise_for_status_429_rate_limit(self, client):
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = json.dumps({
            "status": 429,
            "success": False,
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests",
                "requestId": "req_429",
                "timestamp": "2026-02-17T00:00:00Z",
                "details": {"retry_after": 60}
            }
        })
        mock_response.headers = {"content-type": "application/json"}

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.raise_for_status(mock_response)

        error = exc_info.value
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.details["retry_after"] == 60


def test_validation_error_scenario(client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = json.dumps({
        "status": 400,
        "success": False,
        "error": {
            "code": "BAD_REQUEST",
            "message": "Validation failed",
            "requestId": "req_validation",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {
                "field": "temperature",
                "value": 10.0,
                "constraint": "must be <= 2.0"
            }
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.is_validation_error is True
    assert error.details["field"] == "temperature"


def test_auth_error_scenario(client):
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = json.dumps({
        "status": 401,
        "success": False,
        "error": {
            "code": "UNAUTHORIZED",
            "message": "Invalid API key",
            "requestId": "req_auth",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {}
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.is_auth_error is True
    assert not error.is_server_error
    assert not error.is_validation_error


def test_server_error_scenario(client):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = json.dumps({
        "status": 500,
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Database error",
            "requestId": "req_server",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {},
            "cause": "Connection timeout"
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    assert error.is_server_error is True
    assert not error.is_client_error
    assert error.cause == "Connection timeout"


def test_error_to_dict_for_logging(client):
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.text = json.dumps({
        "status": 403,
        "success": False,
        "error": {
            "code": "FORBIDDEN",
            "message": "Permission denied",
            "requestId": "req_logging",
            "timestamp": "2026-02-17T00:00:00Z",
            "details": {"required": "admin"}
        }
    })
    mock_response.headers = {"content-type": "application/json"}

    with pytest.raises(PollinationsAPIError) as exc_info:
        client.raise_for_status(mock_response)

    error = exc_info.value
    error_dict = error.to_dict()

    assert error_dict["status_code"] == 403
    assert error_dict["error_code"] == "FORBIDDEN"
    assert error_dict["request_id"] == "req_logging"
    assert error_dict["details"]["required"] == "admin"


def _make_mock_response(
    status_code: int,
    body: str = "",
    content_type: str = "application/json",
) -> MagicMock:
    """Build a minimal httpx.Response mock for raise_for_status tests.

    Assigns ``headers`` as a real dict so that ``.get()`` works natively
    without additional MagicMock configuration.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = body
    # Dict real para que resp.headers.get("content-type", "") funcione sin configuración extra
    mock_resp.headers = {"content-type": content_type}
    return mock_resp


# Cuerpo JSON canónico que el backend envía en un error 402 con envelope completo
_402_JSON_BODY = json.dumps({
    "status": 402,
    "success": False,
    "error": {
        "code": "PAYMENT_REQUIRED",
        "message": "Insufficient Pollen balance to complete the request.",
        "timestamp": "2026-02-22T22:00:00Z",
        "requestId": "req_test_abc123",
        "cause": "balance_exhausted",
    },
})


def test_is_payment_required_true_via_status_402():
    """is_payment_required is True when status_code is 402, regardless of error_code."""
    err = PollinationsAPIError(status_code=402, message="Payment required", error_code=None)

    assert err.is_payment_required is True


def test_is_payment_required_true_via_error_code_only():
    """is_payment_required is True via error_code alone (defensive OR branch).

    Covers the edge case where the gateway forwards PAYMENT_REQUIRED in the
    JSON envelope but uses a non-standard HTTP status code.
    """
    # status atípico: sólo el error_code activa la condición
    err = PollinationsAPIError(
        status_code=500,
        message="Payment required",
        error_code="PAYMENT_REQUIRED",
    )

    assert err.is_payment_required is True


def test_is_payment_required_true_via_both_conditions():
    """is_payment_required is True when both status_code=402 and error_code match."""
    err = PollinationsAPIError(
        status_code=402,
        message="Insufficient Pollen balance.",
        error_code="PAYMENT_REQUIRED",
    )

    assert err.is_payment_required is True


def test_is_payment_required_false_on_400():
    """is_payment_required is False for a 400 BAD_REQUEST error."""
    err = PollinationsAPIError(
        status_code=400, message="Bad request", error_code="BAD_REQUEST"
    )

    assert err.is_payment_required is False


def test_is_payment_required_false_on_401():
    """is_payment_required is False for a 401 UNAUTHORIZED error."""
    err = PollinationsAPIError(
        status_code=401, message="Unauthorized", error_code="UNAUTHORIZED"
    )

    assert err.is_payment_required is False


def test_is_payment_required_false_on_403():
    """is_payment_required is False for a 403 FORBIDDEN error."""
    err = PollinationsAPIError(
        status_code=403, message="Forbidden", error_code="FORBIDDEN"
    )

    assert err.is_payment_required is False


def test_is_payment_required_false_on_500():
    """is_payment_required is False for a 500 INTERNAL_ERROR."""
    err = PollinationsAPIError(
        status_code=500, message="Internal server error", error_code="INTERNAL_ERROR"
    )

    assert err.is_payment_required is False


def test_is_payment_required_false_when_neither_condition_matches():
    """is_payment_required is False when neither branch of the OR is satisfied."""
    # Ninguna condición activa: status != 402 y error_code es None
    err = PollinationsAPIError(status_code=403, message="Forbidden", error_code=None)

    assert err.is_payment_required is False


def test_parse_error_response_402_full_json_envelope():
    """_parse_error_response correctly extracts all fields from a canonical 402 envelope."""
    result = _parse_error_response(
        status_code=402,
        body_text=_402_JSON_BODY,
        content_type="application/json",
    )

    assert isinstance(result, PollinationsAPIError)
    assert result.status_code == 402
    assert result.error_code == "PAYMENT_REQUIRED"
    assert result.message == "Insufficient Pollen balance to complete the request."
    assert result.request_id == "req_test_abc123"
    assert result.timestamp == "2026-02-22T22:00:00Z"
    assert result.cause == "balance_exhausted"
    # Verificación end-to-end: el objeto parseado activa is_payment_required
    assert result.is_payment_required is True


def test_parse_error_response_402_empty_body():
    """_parse_error_response handles a 402 with an empty body gracefully.

    When body is empty, error_code remains None; is_payment_required stays
    True via the status_code branch of the OR.
    """
    result = _parse_error_response(
        status_code=402,
        body_text="",
        content_type="application/json",
    )

    assert isinstance(result, PollinationsAPIError)
    assert result.status_code == 402
    assert result.error_code is None
    # is_payment_required sigue True porque status_code == 402
    assert result.is_payment_required is True


def test_parse_error_response_402_plain_text_content_type():
    """_parse_error_response handles a 402 with text/plain (no JSON parsing attempted)."""
    result = _parse_error_response(
        status_code=402,
        body_text="Insufficient balance",
        content_type="text/plain",
    )

    assert isinstance(result, PollinationsAPIError)
    assert result.status_code == 402
    # Sin JSON, el body raw se usa como mensaje
    assert result.message == "Insufficient balance"
    assert result.error_code is None
    assert result.is_payment_required is True


def test_parse_error_response_402_invalid_json():
    """_parse_error_response handles a 402 with malformed JSON without raising."""
    result = _parse_error_response(
        status_code=402,
        body_text="{not: valid, json,,",
        content_type="application/json",
    )

    assert isinstance(result, PollinationsAPIError)
    assert result.status_code == 402
    # JSON inválido → error_code no se puede parsear
    assert result.error_code is None
    # is_payment_required sigue True gracias al status_code
    assert result.is_payment_required is True


def test_parse_error_response_402_json_without_error_object():
    """_parse_error_response handles a 402 where the envelope lacks the 'error' key.

    Falls back to reading 'message' directly from the top-level JSON object.
    """
    body = json.dumps({
        "status": 402,
        "success": False,
        "message": "Pollen balance exhausted",
    })
    result = _parse_error_response(
        status_code=402,
        body_text=body,
        content_type="application/json",
    )

    assert isinstance(result, PollinationsAPIError)
    assert result.status_code == 402
    # Mensaje extraído del fallback data.get("message")
    assert result.message == "Pollen balance exhausted"
    # Sin objeto "error", error_code queda None
    assert result.error_code is None
    assert result.is_payment_required is True


def test_raise_for_status_402_raises_pollinations_api_error():
    """raise_for_status raises PollinationsAPIError (not a generic exception) on 402."""
    mock_resp = _make_mock_response(402, body=_402_JSON_BODY)

    with pytest.raises(PollinationsAPIError) as exc_info:
        PollinationsHttpClient.raise_for_status(mock_resp)

    assert exc_info.value.status_code == 402


def test_raise_for_status_402_error_is_payment_required():
    """raise_for_status: the raised PollinationsAPIError has is_payment_required=True."""
    mock_resp = _make_mock_response(402, body=_402_JSON_BODY)

    with pytest.raises(PollinationsAPIError) as exc_info:
        PollinationsHttpClient.raise_for_status(mock_resp)

    assert exc_info.value.is_payment_required is True


def test_raise_for_status_402_preserves_message():
    """raise_for_status: the raised exception carries the backend message from the envelope."""
    mock_resp = _make_mock_response(402, body=_402_JSON_BODY)

    with pytest.raises(PollinationsAPIError) as exc_info:
        PollinationsHttpClient.raise_for_status(mock_resp)

    assert "Insufficient Pollen balance" in exc_info.value.message


def test_raise_for_status_402_preserves_request_id():
    """raise_for_status: the raised exception carries the requestId from the envelope."""
    mock_resp = _make_mock_response(402, body=_402_JSON_BODY)

    with pytest.raises(PollinationsAPIError) as exc_info:
        PollinationsHttpClient.raise_for_status(mock_resp)

    assert exc_info.value.request_id == "req_test_abc123"


def test_raise_for_status_402_empty_body_still_raises():
    """raise_for_status raises on 402 even when the response body is empty."""
    mock_resp = _make_mock_response(402, body="")

    with pytest.raises(PollinationsAPIError) as exc_info:
        PollinationsHttpClient.raise_for_status(mock_resp)

    assert exc_info.value.status_code == 402
    assert exc_info.value.is_payment_required is True


def test_raise_for_status_200_does_not_raise():
    """raise_for_status does not raise for a successful 200 response (sanity check)."""
    mock_resp = _make_mock_response(200, body='{"id": "chatcmpl-123"}')

    # No debe lanzar ninguna excepción
    PollinationsHttpClient.raise_for_status(mock_resp)


def test_402_is_also_client_error():
    """A 402 is a 4xx status, so is_client_error must remain True."""
    err = PollinationsAPIError(status_code=402, message="Payment required")

    assert err.is_client_error is True


def test_402_is_not_auth_error():
    """A 402 must not be misclassified as an authentication/authorization error."""
    err = PollinationsAPIError(status_code=402, message="Payment required")

    assert err.is_auth_error is False


def test_402_is_not_validation_error():
    """A 402 must not be misclassified as a validation error, even with PAYMENT_REQUIRED code."""
    err = PollinationsAPIError(
        status_code=402, message="Payment required", error_code="PAYMENT_REQUIRED"
    )

    assert err.is_validation_error is False


def test_402_is_not_server_error():
    """A 402 must not be misclassified as a server-side error."""
    err = PollinationsAPIError(status_code=402, message="Payment required")

    assert err.is_server_error is False


def test_existing_error_properties_unaffected_by_402_addition():
    """Regression guard: all pre-existing error properties remain correct for non-402 codes."""
    # 400 BAD_REQUEST
    bad_req = PollinationsAPIError(status_code=400, message="Bad request", error_code="BAD_REQUEST")
    assert bad_req.is_validation_error is True
    assert bad_req.is_payment_required is False

    # 401 UNAUTHORIZED
    unauth = PollinationsAPIError(status_code=401, message="Unauthorized", error_code="UNAUTHORIZED")
    assert unauth.is_auth_error is True
    assert unauth.is_payment_required is False

    # 403 FORBIDDEN
    forbidden = PollinationsAPIError(status_code=403, message="Forbidden", error_code="FORBIDDEN")
    assert forbidden.is_auth_error is True
    assert forbidden.is_payment_required is False

    # 500 INTERNAL_ERROR
    server_err = PollinationsAPIError(status_code=500, message="Server error", error_code="INTERNAL_ERROR")
    assert server_err.is_server_error is True
    assert server_err.is_payment_required is False
