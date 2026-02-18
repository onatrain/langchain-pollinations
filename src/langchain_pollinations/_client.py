"""
This module provides a robust HTTP client for interacting with the Pollinations API.
It supports synchronous and asynchronous requests, JSON processing, and Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from langchain_pollinations._errors import PollinationsAPIError


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """
    Configuration settings for the HTTP client.
    Includes base URL and timeout durations for network requests.
    """
    base_url: str
    timeout_s: float = 120.0


def _parse_error_response(
    status_code: int,
    body_text: str,
    content_type: str,
) -> PollinationsAPIError:
    """
    Parse an API error response body into a structured exception.

    Args:
        status_code: The HTTP status code received.
        body_text: The raw body text of the response.
        content_type: The 'Content-Type' header of the response.

    Returns:
        An instance of PollinationsAPIError with parsed details if available.
    """
    message = "HTTP error"
    error_code: str | None = None
    request_id: str | None = None
    timestamp: str | None = None
    details: dict[str, Any] | None = None
    cause: Any | None = None

    # Solo parsear JSON si Content-Type lo indica
    if "application/json" not in content_type.lower():
        if body_text and body_text.strip():
            message = body_text
        return PollinationsAPIError(
            status_code=status_code,
            message=message,
            body=body_text,
            error_code=error_code,
            request_id=request_id,
            timestamp=timestamp,
            details=details,
            cause=cause,
        )

    # Intentar parsear JSON
    try:
        data = json.loads(body_text) if body_text else {}
    except (json.JSONDecodeError, ValueError):
        if body_text and body_text.strip():
            message = body_text
        return PollinationsAPIError(
            status_code=status_code,
            message=message,
            body=body_text,
            error_code=error_code,
            request_id=request_id,
            timestamp=timestamp,
            details=details,
            cause=cause,
        )

    if not isinstance(data, dict):
        return PollinationsAPIError(
            status_code=status_code,
            message=str(data) if data else message,
            body=body_text,
            error_code=error_code,
            request_id=request_id,
            timestamp=timestamp,
            details=details,
            cause=cause,
        )

    # Extraer campos del envelope: { "status": ..., "success": false, "error": {...} }
    error_obj = data.get("error")

    if isinstance(error_obj, dict):
        error_code = error_obj.get("code")
        if isinstance(error_code, str):
            error_code = error_code.strip()
        else:
            error_code = None

        msg = error_obj.get("message")
        if isinstance(msg, str) and msg.strip():
            message = msg.strip()

        req_id = error_obj.get("requestId")
        if isinstance(req_id, str) and req_id.strip():
            request_id = req_id.strip()

        ts = error_obj.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            timestamp = ts.strip()

        det = error_obj.get("details")
        if isinstance(det, dict):
            details = det

        cause = error_obj.get("cause")
    else:
        msg = data.get("message")
        if isinstance(msg, str) and msg.strip():
            message = msg.strip()

    return PollinationsAPIError(
        status_code=status_code,
        message=message,
        body=body_text,
        error_code=error_code,
        request_id=request_id,
        timestamp=timestamp,
        details=details,
        cause=cause,
    )


class PollinationsHttpClient:
    """
    A lightweight wrapper around httpx for handling API communication.
    Features include structured error parsing, automatic header management, and logging.
    """

    def __init__(self, *, config: HttpConfig, api_key: str) -> None:
        """
        Initialize the HTTP client with specific configuration and authentication.

        Args:
            config: Configuration object containing base URL and timeouts.
            api_key: The API key used for Bearer authentication.
        """
        self._config = config
        self._api_key = api_key
        self._debug_http = os.getenv("POLLINATIONS_HTTP_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
            """
            Remove sensitive information from headers for safe logging.

            Args:
                headers: The dictionary of HTTP headers to process.

            Returns:
                A copy of headers with the Authorization value redacted.
            """
            out = dict(headers)
            for k in ("authorization", "Authorization"):
                if k in out:
                    out[k] = "Bearer ***REDACTED***"
            return out

        def _log_request(request: httpx.Request) -> None:
            """
            Log the details of an outgoing HTTP request if debug mode is enabled.

            Args:
                request: The httpx Request object to log.
            """
            if not self._debug_http:
                return
            logging.warning("HTTPX REQUEST %s %s", request.method, request.url)
            logging.warning("HTTPX REQUEST headers=%s", _redact_headers(dict(request.headers)))
            if request.content:
                try:
                    logging.warning("HTTPX REQUEST body=%s", request.content.decode("utf-8", "ignore"))
                except Exception:
                    logging.warning("HTTPX REQUEST body=(binary) len=%s", len(request.content))

        def _log_response_sync(response: httpx.Response) -> None:
            """
            Log the details of a received HTTP response synchronously.

            Args:
                response: The httpx Response object to log.
            """
            if not self._debug_http:
                return
            req = response.request
            logging.warning("HTTPX RESPONSE %s %s -> %s", req.method, req.url, response.status_code)
            logging.warning("HTTPX RESPONSE headers=%s", dict(response.headers))

            ctype = response.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                logging.warning("HTTPX RESPONSE body=(event-stream; not auto-logged)")
                return

            try:
                response.read()
                logging.warning("HTTPX RESPONSE body=%s", response.text)
            except Exception as e:
                logging.warning("HTTPX RESPONSE body=(unreadable) err=%r", e)

        async def _log_request_async(request: httpx.Request) -> None:
            """
            Log the details of an outgoing HTTP request asynchronously.

            Args:
                request: The httpx Request object to log.
            """
            _log_request(request)

        async def _log_response_async(response: httpx.Response) -> None:
            """
            Log the details of a received HTTP response asynchronously.

            Args:
                response: The httpx Response object to log.
            """
            if not self._debug_http:
                return
            req = response.request
            logging.warning("HTTPX RESPONSE %s %s -> %s", req.method, req.url, response.status_code)
            logging.warning("HTTPX RESPONSE headers=%s", dict(response.headers))
            ctype = response.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                logging.warning("HTTPX RESPONSE body=(event-stream; not auto-logged)")
                return
            try:
                await response.aread()
                logging.warning("HTTPX RESPONSE body=%s", response.text)
            except Exception as e:
                logging.warning("HTTPX RESPONSE body=(unreadable) err=%r", e)

        EventHooksDict = dict[str, list[Callable[..., Any]]]

        hooks_sync: EventHooksDict = {"request": [_log_request], "response": [_log_response_sync]}
        hooks_async: EventHooksDict = {"request": [_log_request_async], "response": [_log_response_async]}

        self._client = httpx.Client(timeout=httpx.Timeout(config.timeout_s), event_hooks=hooks_sync)
        self._aclient = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_s), event_hooks=hooks_async)

    def close(self) -> None:
        """
        Close the underlying synchronous httpx client.
        """
        self._client.close()

    async def aclose(self) -> None:
        """
        Close the underlying asynchronous httpx client.
        """
        await self._aclient.aclose()

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        """
        Generate default headers for API requests including authentication.

        Args:
            accept: Optional string to specify the 'Accept' header value.

        Returns:
            A dictionary containing the required HTTP headers.
        """
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if accept:
            headers["Accept"] = accept
        return headers

    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        """
        Validate the response status and raise a structured PollinationsAPIError if it failed.

        Args:
            resp: The httpx Response object to evaluate.

        Raises:
            PollinationsAPIError: If the status code is not in the 2xx range.
        """
        if 200 <= resp.status_code < 300:
            return

        body_text: str | None = None
        try:
            body_text = resp.text
        except Exception:
            body_text = None

        content_type = resp.headers.get("content-type", "")

        error = _parse_error_response(
            status_code=resp.status_code,
            body_text=body_text or "",
            content_type=content_type,
        )

        raise error

    def post_json(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        """
        Execute a synchronous POST request with a JSON payload.

        Args:
            path: The API endpoint path.
            payload: Dictionary containing the JSON data to send.
            stream: Whether the request should be treated as a stream.

        Returns:
            The httpx Response object.
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream" if stream else None)
        headers["Content-Type"] = "application/json"
        resp = self._client.post(url, headers=headers, json=payload)
        self.raise_for_status(resp)
        return resp

    async def apost_json(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        """
        Execute an asynchronous POST request with a JSON payload.

        Args:
            path: The API endpoint path.
            payload: Dictionary containing the JSON data to send.
            stream: Whether the request should be treated as a stream.

        Returns:
            The httpx Response object.
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream" if stream else None)
        headers["Content-Type"] = "application/json"
        resp = await self._aclient.post(url, headers=headers, json=payload)
        self.raise_for_status(resp)
        return resp

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """
        Execute a synchronous GET request to the specified path.

        Args:
            path: The API endpoint path.
            params: Optional dictionary of query parameters.

        Returns:
            The httpx Response object.
        """
        url = f"{self._config.base_url}{path}"
        resp = self._client.get(url, headers=self._headers(), params=params)
        self.raise_for_status(resp)
        return resp

    async def aget(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """
        Execute an asynchronous GET request to the specified path.

        Args:
            path: The API endpoint path.
            params: Optional dictionary of query parameters.

        Returns:
            The httpx Response object.
        """
        url = f"{self._config.base_url}{path}"
        resp = await self._aclient.get(url, headers=self._headers(), params=params)
        self.raise_for_status(resp)
        return resp

    def stream_post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """
        Initiate a synchronous streaming POST request using a context manager.

        Args:
            path: The API endpoint path.
            payload: Dictionary containing the JSON data to send.

        Returns:
            A context manager for the httpx stream.
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"
        return self._client.stream("POST", url, headers=headers, json=payload)

    def astream_post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """
        Initiate an asynchronous streaming POST request using a context manager.

        Args:
            path: The API endpoint path.
            payload: Dictionary containing the JSON data to send.

        Returns:
            An asynchronous context manager for the httpx stream.
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"
        return self._aclient.stream("POST", url, headers=headers, json=payload)
