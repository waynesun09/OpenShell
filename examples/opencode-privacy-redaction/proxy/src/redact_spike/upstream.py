# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from redact_spike.config import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpstreamResponse:
    status_code: int
    body: Any


class NvidiaInferenceClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        require_api_key: bool = True,
    ) -> None:
        self.settings = settings
        self.client = client
        self.base_url = base_url or settings.upstream_base_url
        self.model = model or settings.upstream_model
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.require_api_key = require_api_key

    async def chat_completions(self, payload: dict[str, Any]) -> UpstreamResponse:
        if self.require_api_key and not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required for upstream inference calls")

        body = dict(payload)
        body.setdefault("model", self.model)

        headers = self._headers()

        close_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=60)
        url = chat_completions_url(self.base_url)
        logger.info("upstream_request url=%s model=%s stream=%s", url, body.get("model"), body.get("stream", False))
        try:
            response = await client.post(
                url,
                headers=headers,
                json=body,
            )
            try:
                response_body: Any = response.json()
            except ValueError:
                response_body = {"error": response.text}
            logger.info(
                "upstream_response url=%s status=%s body=%s",
                url,
                response.status_code,
                _sanitize_for_log(response_body),
            )
            return UpstreamResponse(status_code=response.status_code, body=response_body)
        finally:
            if close_client:
                await client.aclose()

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        if self.require_api_key and not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required for upstream inference calls")

        body = dict(payload)
        body.setdefault("model", self.model)

        headers = self._headers(accept="text/event-stream")

        close_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(60, read=None))
        url = chat_completions_url(self.base_url)
        logger.info("upstream_stream_request url=%s model=%s", url, body.get("model"))
        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
            ) as response:
                logger.info("upstream_stream_response url=%s status=%s", url, response.status_code)
                async for chunk in response.aiter_bytes():
                    yield chunk
        finally:
            if close_client:
                await client.aclose()

    def _headers(self, accept: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_log(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        if value.strip().startswith("data:image/"):
            return "[REDACTED_IMAGE_DATA_URL]"
        if len(value) > 500:
            return f"{value[:500]}...[TRUNCATED {len(value) - 500} chars]"
    return value
