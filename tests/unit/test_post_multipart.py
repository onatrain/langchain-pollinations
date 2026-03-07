from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_pollinations._client import HttpConfig, PollinationsHttpClient
from langchain_pollinations._errors import PollinationsAPIError

_TEST_BASE_URL = "https://test.pollinations.ai"
_TEST_API_KEY = "sk-test-key-abc123"
_TEST_PATH = "/v1/audio/transcriptions"

_SAMPLE_AUDIO_BYTES = b"\xff\xfb\x90\x00" * 16
_SAMPLE_FILES: dict[str, tuple[str, bytes, str]] = {
    "file": ("audio.mp3", _SAMPLE_AUDIO_BYTES, "audio/mpeg"),
}
_SAMPLE_DATA: dict[str, str] = {
    "model": "whisper-large-v3",
    "language": "en",
    "response_format": "json",
}


def _make_client() -> PollinationsHttpClient:
    """Instantiate a PollinationsHttpClient with test credentials."""
    config = HttpConfig(base_url=_TEST_BASE_URL, timeout_s=10.0)
    return PollinationsHttpClient(config=config, api_key=_TEST_API_KEY)


def _make_mock_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text_body: str = "",
    content_type: str = "application/json",
) -> MagicMock:
    """
    Build a MagicMock that satisfies PollinationsHttpClient.raise_for_status
    attribute access: .status_code, .text, .headers.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.text = json.dumps(json_body) if json_body is not None else text_body
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = json.JSONDecodeError("no JSON", "", 0)
    return resp


def _make_error_response(
    status_code: int,
    error_code: str,
    message: str,
) -> MagicMock:
    """Build a mock error response with the structured JSON envelope the API returns."""
    body = {
        "status": status_code,
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": "2026-01-01T00:00:00Z",
            "details": {},
            "requestId": "req_test_000",
            "cause": None,
        },
    }
    return _make_mock_response(
        status_code=status_code,
        json_body=body,
        content_type="application/json",
    )


def _make_client_with_sync_mock(
    mock_response: MagicMock,
) -> tuple[PollinationsHttpClient, MagicMock]:
    """
    Return a client whose internal _client is replaced with a MagicMock
    whose .post() returns mock_response.
    """
    client = _make_client()
    sync_mock = MagicMock()
    sync_mock.post.return_value = mock_response
    client._client = sync_mock
    return client, sync_mock


def _make_client_with_async_mock(
    mock_response: MagicMock,
) -> tuple[PollinationsHttpClient, MagicMock]:
    """
    Return a client whose internal _aclient is replaced with a MagicMock
    whose .post() is an AsyncMock returning mock_response.
    """
    client = _make_client()
    async_mock = MagicMock()
    async_mock.post = AsyncMock(return_value=mock_response)
    client._aclient = async_mock
    return client, async_mock


class TestPostMultipart:

    def test_url_is_base_url_plus_path(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        positional_args, _ = sync_mock.post.call_args
        assert positional_args[0] == f"{_TEST_BASE_URL}{_TEST_PATH}"

    def test_authorization_header_is_present(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = sync_mock.post.call_args
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == f"Bearer {_TEST_API_KEY}"

    def test_content_type_header_is_absent(self):
        """httpx auto-sets Content-Type + boundary for multipart; manual setting would corrupt it."""
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = sync_mock.post.call_args
        headers = kwargs["headers"]
        assert "Content-Type" not in headers
        assert "content-type" not in headers

    def test_files_kwarg_is_forwarded_verbatim(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = sync_mock.post.call_args
        assert kwargs["files"] is _SAMPLE_FILES

    def test_data_kwarg_is_forwarded_verbatim(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = sync_mock.post.call_args
        assert kwargs["data"] is _SAMPLE_DATA

    def test_json_kwarg_is_not_present(self):
        """post_multipart must never pass json= to httpx; that belongs to post_json."""
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = sync_mock.post.call_args
        assert "json" not in kwargs

    def test_returns_raw_response_on_200(self):
        resp = _make_mock_response(200, json_body={"text": "hello world"})
        client, _ = _make_client_with_sync_mock(resp)

        result = client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        assert result is resp

    def test_empty_data_dict_is_accepted(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        result = client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert result is resp
        _, kwargs = sync_mock.post.call_args
        assert kwargs["data"] == {}

    def test_wav_file_tuple_is_forwarded_correctly(self):
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        wav_files = {"file": ("grabacion.wav", wav_bytes, "audio/wav")}
        resp = _make_mock_response(200, json_body={"text": "hola"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        client.post_multipart(_TEST_PATH, files=wav_files, data={})

        _, kwargs = sync_mock.post.call_args
        assert kwargs["files"] == wav_files
        file_tuple = kwargs["files"]["file"]
        assert file_tuple[0] == "grabacion.wav"
        assert file_tuple[1] == wav_bytes
        assert file_tuple[2] == "audio/wav"

    def test_raises_api_error_on_400(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Unsupported audio format")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        err = exc_info.value
        assert err.status_code == 400
        assert err.error_code == "BAD_REQUEST"
        assert err.is_validation_error

    def test_raises_api_error_on_401(self):
        resp = _make_error_response(401, "UNAUTHORIZED", "Auth required")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 401
        assert err.is_auth_error

    def test_raises_api_error_on_402(self):
        resp = _make_error_response(402, "PAYMENT_REQUIRED", "Insufficient pollen")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 402
        assert err.is_payment_required

    def test_raises_api_error_on_403(self):
        resp = _make_error_response(403, "FORBIDDEN", "Permission denied")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 403
        assert err.is_auth_error

    def test_raises_api_error_on_500(self):
        resp = _make_error_response(500, "INTERNAL_ERROR", "Something went wrong")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 500
        assert err.is_server_error

    def test_error_request_id_is_parsed(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Bad file")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert exc_info.value.request_id == "req_test_000"

    def test_error_message_is_parsed(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Unsupported audio format")
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert "Unsupported audio format" in exc_info.value.message

    def test_non_json_error_body_is_handled(self):
        """A plain-text error response must not crash raise_for_status."""
        resp = _make_mock_response(
            status_code=503,
            text_body="Service Unavailable",
            content_type="text/plain",
        )
        client, _ = _make_client_with_sync_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert exc_info.value.status_code == 503

    def test_uses_internal_sync_client_not_async_client(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)
        async_mock = MagicMock()
        async_mock.post = AsyncMock()
        client._aclient = async_mock

        client.post_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        sync_mock.post.assert_called_once()
        async_mock.post.assert_not_called()

    def test_path_segments_are_not_altered(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, sync_mock = _make_client_with_sync_mock(resp)

        custom_path = "/some/custom/endpoint"
        client.post_multipart(custom_path, files=_SAMPLE_FILES, data={})

        positional_args, _ = sync_mock.post.call_args
        assert positional_args[0] == f"{_TEST_BASE_URL}{custom_path}"


@pytest.mark.asyncio
class TestApostMultipart:

    async def test_url_is_base_url_plus_path(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        positional_args, _ = async_mock.post.call_args
        assert positional_args[0] == f"{_TEST_BASE_URL}{_TEST_PATH}"

    async def test_authorization_header_is_present(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = async_mock.post.call_args
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == f"Bearer {_TEST_API_KEY}"

    async def test_content_type_header_is_absent(self):
        """httpx auto-sets Content-Type + boundary for multipart; manual setting would corrupt it."""
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = async_mock.post.call_args
        headers = kwargs["headers"]
        assert "Content-Type" not in headers
        assert "content-type" not in headers

    async def test_files_kwarg_is_forwarded_verbatim(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = async_mock.post.call_args
        assert kwargs["files"] is _SAMPLE_FILES

    async def test_data_kwarg_is_forwarded_verbatim(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = async_mock.post.call_args
        assert kwargs["data"] is _SAMPLE_DATA

    async def test_json_kwarg_is_not_present(self):
        """apost_multipart must never pass json= to httpx."""
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        _, kwargs = async_mock.post.call_args
        assert "json" not in kwargs

    async def test_returns_raw_response_on_200(self):
        resp = _make_mock_response(200, json_body={"text": "hello async"})
        client, _ = _make_client_with_async_mock(resp)

        result = await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        assert result is resp

    async def test_empty_data_dict_is_accepted(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        result = await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert result is resp
        _, kwargs = async_mock.post.call_args
        assert kwargs["data"] == {}

    async def test_wav_file_tuple_is_forwarded_correctly(self):
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        wav_files = {"file": ("grabacion.wav", wav_bytes, "audio/wav")}
        resp = _make_mock_response(200, json_body={"text": "hola"})
        client, async_mock = _make_client_with_async_mock(resp)

        await client.apost_multipart(_TEST_PATH, files=wav_files, data={})

        _, kwargs = async_mock.post.call_args
        assert kwargs["files"] == wav_files
        file_tuple = kwargs["files"]["file"]
        assert file_tuple[0] == "grabacion.wav"
        assert file_tuple[1] == wav_bytes
        assert file_tuple[2] == "audio/wav"

    async def test_raises_api_error_on_400(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Unsupported audio format")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        err = exc_info.value
        assert err.status_code == 400
        assert err.error_code == "BAD_REQUEST"
        assert err.is_validation_error

    async def test_raises_api_error_on_401(self):
        resp = _make_error_response(401, "UNAUTHORIZED", "Auth required")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 401
        assert err.is_auth_error

    async def test_raises_api_error_on_402(self):
        resp = _make_error_response(402, "PAYMENT_REQUIRED", "Insufficient pollen")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 402
        assert err.is_payment_required

    async def test_raises_api_error_on_403(self):
        resp = _make_error_response(403, "FORBIDDEN", "Permission denied")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 403
        assert err.is_auth_error

    async def test_raises_api_error_on_500(self):
        resp = _make_error_response(500, "INTERNAL_ERROR", "Something went wrong")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        err = exc_info.value
        assert err.status_code == 500
        assert err.is_server_error

    async def test_error_request_id_is_parsed(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Bad file")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert exc_info.value.request_id == "req_test_000"

    async def test_error_message_is_parsed(self):
        resp = _make_error_response(400, "BAD_REQUEST", "Unsupported audio format")
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert "Unsupported audio format" in exc_info.value.message

    async def test_non_json_error_body_is_handled(self):
        resp = _make_mock_response(
            status_code=503,
            text_body="Service Unavailable",
            content_type="text/plain",
        )
        client, _ = _make_client_with_async_mock(resp)

        with pytest.raises(PollinationsAPIError) as exc_info:
            await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data={})

        assert exc_info.value.status_code == 503

    async def test_uses_internal_async_client_not_sync_client(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)
        sync_mock = MagicMock()
        client._client = sync_mock

        await client.apost_multipart(_TEST_PATH, files=_SAMPLE_FILES, data=_SAMPLE_DATA)

        async_mock.post.assert_called_once()
        sync_mock.post.assert_not_called()

    async def test_path_segments_are_not_altered(self):
        resp = _make_mock_response(200, json_body={"text": "ok"})
        client, async_mock = _make_client_with_async_mock(resp)

        custom_path = "/some/custom/endpoint"
        await client.apost_multipart(custom_path, files=_SAMPLE_FILES, data={})

        positional_args, _ = async_mock.post.call_args
        assert positional_args[0] == f"{_TEST_BASE_URL}{custom_path}"
