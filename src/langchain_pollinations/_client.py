from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import logging
import os

from langchain_pollinations._errors import PollinationsAPIError


@dataclass(frozen=True, slots=True)
class HttpConfig:
    base_url: str
    timeout_s: float = 120.0


class PollinationsHttpClient:
    """
    Wrapper HTTPX ligero con:
    - JSON requests
    - Streaming SSE via httpx.Client.stream / AsyncClient.stream
    - Debug logging opcional
    """

    def __init__(self, *, config: HttpConfig, api_key: str) -> None:
        self._config = config
        self._api_key = api_key
        self._debug_http = os.getenv("POLLINATIONS_HTTP_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
            out = dict(headers)
            for k in ("authorization", "Authorization"):
                if k in out:
                    out[k] = "Bearer ***REDACTED***"
            return out

        def _log_request(request: httpx.Request) -> None:
            if not self._debug_http:
                return
            logging.warning("HTTPX REQUEST %s %s", request.method, request.url)
            logging.warning("HTTPX REQUEST headers=%s", _redact_headers(dict(request.headers)))
            if request.content:
                try:
                    logging.warning("HTTPX REQUEST body=%s", request.content.decode("utf-8", "ignore"))
                except Exception:
                    logging.warning("HTTPX REQUEST body=(binary) len=%s", len(request.content))

        def _log_response(response: httpx.Response) -> None:
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

        hooks = {"request": [_log_request], "response": [_log_response]}
        self._client = httpx.Client(timeout=httpx.Timeout(config.timeout_s), event_hooks=hooks)
        self._aclient = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_s), event_hooks=hooks)

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if accept:
            headers["Accept"] = accept
        return headers

    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        body = None
        try:
            body = resp.text
        except Exception:
            body = None
        raise PollinationsAPIError(
            status_code=resp.status_code,
            message=resp.reason_phrase or "HTTP error",
            body=body,
        )

    def post_json(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream" if stream else None)
        headers["Content-Type"] = "application/json"
        resp = self._client.post(url, headers=headers, json=payload)
        self.raise_for_status(resp)
        return resp

    async def apost_json(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream" if stream else None)
        headers["Content-Type"] = "application/json"
        resp = await self._aclient.post(url, headers=headers, json=payload)
        self.raise_for_status(resp)
        return resp

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self._config.base_url}{path}"
        resp = self._client.get(url, headers=self._headers(), params=params)
        self.raise_for_status(resp)
        return resp

    async def aget(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self._config.base_url}{path}"
        resp = await self._aclient.get(url, headers=self._headers(), params=params)
        self.raise_for_status(resp)
        return resp

    def stream_post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """
        Retorna un httpx stream context manager.

        Uso:
            with client.stream_post_json(...) as r:
                for line in r.iter_lines():
                    ...
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"
        return self._client.stream("POST", url, headers=headers, json=payload)

    def astream_post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """
        Retorna un httpx stream context manager asíncrono.

        Usage:
            async with client.astream_post_json(...) as r:
                async for line in r.aiter_lines():
                    ...
        """
        url = f"{self._config.base_url}{path}"
        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"
        return self._aclient.stream("POST", url, headers=headers, json=payload)
