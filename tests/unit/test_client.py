from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock
import logging

import httpx
import pytest

from langchain_pollinations._client import HttpConfig, PollinationsHttpClient
from langchain_pollinations._errors import PollinationsAPIError


def test_httpconfig_initialization():
    cfg = HttpConfig(base_url="https://example.com", timeout_s=10.0)

    assert cfg.base_url == "https://example.com"
    assert cfg.timeout_s == 10.0


def make_client():
    cfg = HttpConfig(base_url="https://example.com", timeout_s=5.0)
    return PollinationsHttpClient(config=cfg, api_key="secret-key")


def test_headers_without_accept():
    client = make_client()

    headers = client._headers()

    assert headers["Authorization"] == "Bearer secret-key"
    assert "Accept" not in headers


def test_headers_with_accept():
    client = make_client()

    headers = client._headers(accept="application/json")

    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Accept"] == "application/json"


def test_raise_for_status_success():
    resp = SimpleNamespace(status_code=200, reason_phrase="OK", text="ok")

    # No debe lanzar excepción en rango 2xx.
    PollinationsHttpClient.raise_for_status(resp)


def test_raise_for_status_error_raises():
    resp = SimpleNamespace(status_code=404, reason_phrase="Not Found", text="missing")

    with pytest.raises(PollinationsAPIError) as exc:
        PollinationsHttpClient.raise_for_status(resp)

    assert exc.value.status_code == 404
    assert "Not Found" in exc.value.message
    assert "missing" in (exc.value.body or "")


def test_post_json_calls_client_and_uses_stream_false(monkeypatch):
    client = make_client()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.post.return_value = mock_response
    client._client = mock_client

    called = {}

    def fake_rfs(resp):
        called["resp"] = resp

    # Patch en la instancia, no en la clase, para evitar el binding de método.
    monkeypatch.setattr(client, "raise_for_status", fake_rfs)

    payload = {"foo": "bar"}
    resp = client.post_json("/test", payload, stream=False)

    assert resp is mock_response
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0] == "https://example.com/test"
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"
    # Como stream=False no debe enviar Accept especial.
    assert "Accept" not in headers
    assert kwargs["json"] == payload
    assert called["resp"] is mock_response


@pytest.mark.asyncio
async def test_apost_json_calls_async_client_and_uses_stream_true(monkeypatch):
    client = make_client()
    mock_ac = AsyncMock()
    mock_response = MagicMock()
    mock_ac.post = AsyncMock(return_value=mock_response)
    client._aclient = mock_ac

    called = {}

    def fake_rfs(resp):
        called["resp"] = resp

    # Igual que arriba, se parchea en la instancia.
    monkeypatch.setattr(client, "raise_for_status", fake_rfs)

    payload = {"x": 1}
    resp = await client.apost_json("/stream", payload, stream=True)

    assert resp is mock_response
    mock_ac.post.assert_awaited_once()
    args, kwargs = mock_ac.post.call_args
    assert args[0] == "https://example.com/stream"
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"
    # Con stream=True debe mandar Accept text/event-stream.
    assert headers["Accept"] == "text/event-stream"
    assert kwargs["json"] == payload
    assert called["resp"] is mock_response


def test_get_calls_client_with_params():
    client = make_client()
    mock_client = MagicMock()
    mock_response = MagicMock()
    # raise_for_status necesita un status_code entero.
    mock_response.status_code = 200
    mock_client.get.return_value = mock_response
    client._client = mock_client

    params = {"q": "test"}
    resp = client.get("/items", params=params)

    assert resp is mock_response
    mock_client.get.assert_called_once()
    args, kwargs = mock_client.get.call_args
    assert args[0] == "https://example.com/items"
    assert kwargs["params"] == params
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"


@pytest.mark.asyncio
async def test_aget_calls_async_client_with_params():
    client = make_client()
    mock_ac = AsyncMock()
    mock_response = MagicMock()
    # Igual que en el caso síncrono, status_code debe ser un int.
    mock_response.status_code = 200
    mock_ac.get = AsyncMock(return_value=mock_response)
    client._aclient = mock_ac

    params = {"page": 2}
    resp = await client.aget("/items", params=params)

    assert resp is mock_response
    mock_ac.get.assert_awaited_once()
    args, kwargs = mock_ac.get.call_args
    assert args[0] == "https://example.com/items"
    assert kwargs["params"] == params
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"


def test_stream_post_json_returns_stream_context_manager():
    client = make_client()
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_client.stream.return_value = mock_stream
    client._client = mock_client

    payload = {"a": 1}
    cm = client.stream_post_json("/stream", payload)

    assert cm is mock_stream
    mock_client.stream.assert_called_once()
    args, kwargs = mock_client.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "https://example.com/stream"
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "text/event-stream"
    assert kwargs["json"] == payload


@pytest.mark.asyncio
async def test_astream_post_json_returns_async_stream_context_manager():
    client = make_client()
    mock_ac = AsyncMock()
    mock_stream = MagicMock()
    # En httpx.AsyncClient.stream es un método síncrono que devuelve un context manager,
    # así que aquí se simula con un MagicMock normal, no AsyncMock.
    mock_ac.stream = MagicMock(return_value=mock_stream)
    client._aclient = mock_ac

    payload = {"a": 1}
    cm = client.astream_post_json("/astream", payload)

    assert cm is mock_stream
    mock_ac.stream.assert_called_once()
    args, kwargs = mock_ac.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "https://example.com/astream"
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "text/event-stream"
    assert kwargs["json"] == payload


def test_close_closes_underlying_client():
    client = make_client()
    mock_client = MagicMock()
    client._client = mock_client

    client.close()

    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_underlying_async_client():
    client = make_client()
    mock_ac = AsyncMock()
    client._aclient = mock_ac

    await client.aclose()

    mock_ac.aclose.assert_awaited_once()

def test_log_request_and_redact_headers_debug_on(monkeypatch, caplog):
    # Activa el modo debug de HTTP para que se ejecuten los hooks de logging.
    monkeypatch.setenv("POLLINATIONS_HTTP_DEBUG", "1")
    client = make_client()

    # Primer hook de request registrado en httpx.Client.
    request_hook = client._client.event_hooks["request"][0]

    request = httpx.Request(
        "POST",
        "https://example.com/test",
        headers={"Authorization": "Bearer secret-token", "X-Other": "1"},
        content=b'{"data": 123}',
    )

    with caplog.at_level(logging.WARNING):
        request_hook(request)

    # Debe haber redactado el header Authorization.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "***REDACTED***" in messages
    assert "secret-token" not in messages


def test_log_request_binary_body_fallback(monkeypatch, caplog):
    monkeypatch.setenv("POLLINATIONS_HTTP_DEBUG", "1")
    client = make_client()
    request_hook = client._client.event_hooks["request"][0]

    class BadContent:
        def __len__(self):
            return 3

        def decode(self, *args, **kwargs):
            # Fuerza la rama de excepción al decodificar el body.
            raise ValueError("cannot decode")

    class DummyRequest:
        method = "POST"
        url = "https://example.com/binary"
        headers = {"Authorization": "Bearer top-secret"}
        content = BadContent()

    with caplog.at_level(logging.WARNING):
        request_hook(DummyRequest())

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    # La rama de excepción loguea que el cuerpo es binario.
    assert "binary" in messages


def test_log_response_sync_debug_on(monkeypatch, caplog):
    monkeypatch.setenv("POLLINATIONS_HTTP_DEBUG", "1")
    client = make_client()
    response_hook = client._client.event_hooks["response"][0]

    req = httpx.Request("GET", "https://example.com/items")

    # Caso normal: content-type != text/event-stream.
    resp_json = httpx.Response(
        200,
        request=req,
        text="ok",
        headers={"content-type": "application/json"},
    )

    with caplog.at_level(logging.WARNING):
        response_hook(resp_json)

    # Caso event-stream: debe tomar el early return sin leer el body.
    resp_stream = httpx.Response(
        200,
        request=req,
        text="",
        headers={"content-type": "text/event-stream"},
    )

    with caplog.at_level(logging.WARNING):
        response_hook(resp_stream)

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "HTTPX RESPONSE" in messages


@pytest.mark.asyncio
async def test_log_request_and_response_async_debug_on(monkeypatch, caplog):
    monkeypatch.setenv("POLLINATIONS_HTTP_DEBUG", "1")
    client = make_client()

    async_request_hook = client._aclient.event_hooks["request"][0]
    async_response_hook = client._aclient.event_hooks["response"][0]

    req = httpx.Request(
        "POST",
        "https://example.com/async",
        headers={"Authorization": "Bearer async-token"},
        content=b"{}",
    )
    resp_json = httpx.Response(
        200,
        request=req,
        text="async-ok",
        headers={"content-type": "application/json"},
    )
    resp_stream = httpx.Response(
        200,
        request=req,
        text="",
        headers={"content-type": "text/event-stream"},
    )

    with caplog.at_level(logging.WARNING):
        # _log_request_async delega en _log_request.
        await async_request_hook(req)
        # _log_response_async, rama normal y rama event-stream.
        await async_response_hook(resp_json)
        await async_response_hook(resp_stream)

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "HTTPX RESPONSE" in messages

def test_raise_for_status_handles_text_error():
    # Cubre la rama except al intentar leer resp.text en un error.

    class BadResp:
        status_code = 500
        reason_phrase = "Boom"

        @property
        def text(self):
            raise RuntimeError("broken .text")

    with pytest.raises(PollinationsAPIError) as exc:
        PollinationsHttpClient.raise_for_status(BadResp())

    # Cuando leer .text falla, body debe quedar en None.
    assert exc.value.status_code == 500
    assert exc.value.body is None