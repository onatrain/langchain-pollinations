"""
Tests unitarios para manejo estructurado de errores del API.
"""

import json
import pytest
from unittest.mock import Mock
from langchain_pollinations._errors import PollinationsAPIError
from langchain_pollinations._client import _parse_error_response, PollinationsHttpClient, HttpConfig


class TestPollinationsAPIError:
    """Tests para la clase PollinationsAPIError."""

    def test_error_creation_minimal(self):
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

    def test_error_creation_full(self):
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

    def test_str_minimal(self):
        error = PollinationsAPIError(status_code=400, message="Bad request")
        s = str(error)

        assert "400" in s
        assert "Bad request" in s
        assert "PollinationsAPIError" in s

    def test_str_with_code_and_request_id(self):
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

    def test_str_with_body(self):
        error = PollinationsAPIError(
            status_code=500,
            message="Error",
            body="x" * 1000
        )
        s = str(error)

        assert "1000 chars" in s

    def test_repr_format(self):
        error = PollinationsAPIError(
            status_code=401,
            message="Unauthorized",
            error_code="UNAUTHORIZED"
        )
        r = repr(error)

        assert "PollinationsAPIError(" in r
        assert "status_code=401" in r
        assert "error_code='UNAUTHORIZED'" in r

    def test_to_dict(self):
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

    def test_is_client_error(self):
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

    def test_is_not_client_error(self):
        e399 = PollinationsAPIError(status_code=399, message="Unknown")
        e500 = PollinationsAPIError(status_code=500, message="Server error")

        assert e399.is_client_error is False
        assert e500.is_client_error is False

    def test_is_server_error(self):
        e500 = PollinationsAPIError(status_code=500, message="Internal error")
        e502 = PollinationsAPIError(status_code=502, message="Bad gateway")
        e503 = PollinationsAPIError(status_code=503, message="Unavailable")
        e599 = PollinationsAPIError(status_code=599, message="Server error")

        assert e500.is_server_error is True
        assert e502.is_server_error is True
        assert e503.is_server_error is True
        assert e599.is_server_error is True

    def test_is_not_server_error(self):
        e400 = PollinationsAPIError(status_code=400, message="Client error")
        e499 = PollinationsAPIError(status_code=499, message="Client error")
        e600 = PollinationsAPIError(status_code=600, message="Unknown")

        assert e400.is_server_error is False
        assert e499.is_server_error is False
        assert e600.is_server_error is False

    def test_is_auth_error(self):
        e401 = PollinationsAPIError(status_code=401, message="Unauthorized")
        e403 = PollinationsAPIError(status_code=403, message="Forbidden")

        assert e401.is_auth_error is True
        assert e403.is_auth_error is True

    def test_is_not_auth_error(self):
        e400 = PollinationsAPIError(status_code=400, message="Bad request")
        e404 = PollinationsAPIError(status_code=404, message="Not found")
        e500 = PollinationsAPIError(status_code=500, message="Server error")

        assert e400.is_auth_error is False
        assert e404.is_auth_error is False
        assert e500.is_auth_error is False

    def test_is_validation_error(self):
        e = PollinationsAPIError(
            status_code=400,
            message="Validation failed",
            error_code="BAD_REQUEST"
        )

        assert e.is_validation_error is True

    def test_is_not_validation_error_wrong_code(self):
        e = PollinationsAPIError(
            status_code=400,
            message="Other error",
            error_code="OTHER_ERROR"
        )

        assert e.is_validation_error is False

    def test_is_not_validation_error_wrong_status(self):
        e = PollinationsAPIError(
            status_code=500,
            message="Error",
            error_code="BAD_REQUEST"
        )

        assert e.is_validation_error is False

    def test_is_not_validation_error_no_code(self):
        e = PollinationsAPIError(
            status_code=400,
            message="Bad request"
        )

        assert e.is_validation_error is False


class TestParseErrorResponse:
    """Tests para la función _parse_error_response."""

    def test_parse_400_with_validation_details(self):
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

    def test_parse_401_unauthorized(self):
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

    def test_parse_403_forbidden_with_details(self):
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

    def test_parse_500_with_cause(self):
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

    def test_parse_non_json_content_type(self):
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

    def test_parse_invalid_json(self):
        body = "{ invalid json"

        error = _parse_error_response(400, body, "application/json")

        assert error.status_code == 400
        assert "invalid json" in error.message
        assert error.error_code is None
        assert error.request_id is None
        assert error.body == body

    def test_parse_json_without_error_object(self):
        body = json.dumps({
            "message": "Something went wrong"
        })

        error = _parse_error_response(500, body, "application/json")

        assert error.status_code == 500
        assert error.message == "Something went wrong"
        assert error.error_code is None
        assert error.request_id is None

    def test_parse_empty_body(self):
        error = _parse_error_response(500, "", "application/json")

        assert error.status_code == 500
        assert error.message == "HTTP error"
        assert error.error_code is None
        assert error.request_id is None

    def test_parse_json_array(self):
        body = json.dumps(["error1", "error2"])

        error = _parse_error_response(400, body, "application/json")

        assert error.status_code == 400
        assert error.error_code is None

    def test_parse_strips_whitespace(self):
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

    def test_parse_handles_non_string_values(self):
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

    def test_parse_long_body_in_message(self):
        body = "x" * 400

        error = _parse_error_response(500, body, "text/plain")

        assert error.status_code == 500
        assert len(error.message) == 400
        assert error.body == body


class TestPollinationsHttpClientRaiseForStatus:
    """Tests para el método _raise_for_status de PollinationsHttpClient."""

    @pytest.fixture
    def client(self):
        config = HttpConfig(base_url="https://test.api", timeout_s=3_0.0)
        return PollinationsHttpClient(config=config, api_key="test-key")

    def test_raise_for_status_success_200(self, client):
        mock_response = Mock()
        mock_response.status_code = 200

        client.raise_for_status(mock_response)

    def test_raise_for_status_success_201(self, client):
        mock_response = Mock()
        mock_response.status_code = 201

        client.raise_for_status(mock_response)

    def test_raise_for_status_success_299(self, client):
        mock_response = Mock()
        mock_response.status_code = 299

        client.raise_for_status(mock_response)

    def test_raise_for_status_400_with_json(self, client):
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

    def test_raise_for_status_401_unauthorized(self, client):
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

    def test_raise_for_status_403_forbidden(self, client):
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

    def test_raise_for_status_500_server_error(self, client):
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

    def test_raise_for_status_non_json_error(self, client):
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

    def test_raise_for_status_text_exception(self, client):
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

    def test_raise_for_status_empty_headers(self, client):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Error"
        mock_response.headers = {}

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.raise_for_status(mock_response)

        error = exc_info.value
        assert error.status_code == 400

    def test_raise_for_status_404_not_found(self, client):
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


class TestIntegrationScenarios:
    """Tests de escenarios de integración."""

    @pytest.fixture
    def client(self):
        config = HttpConfig(base_url="https://test.api", timeout_s=30.0)
        return PollinationsHttpClient(config=config, api_key="test-key")

    def test_validation_error_scenario(self, client):
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

    def test_auth_error_scenario(self, client):
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

    def test_server_error_scenario(self, client):
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

    def test_error_to_dict_for_logging(self, client):
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
