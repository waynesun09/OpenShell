# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import httpx

from redact_spike.config import Settings
from redact_spike.upstream import NvidiaInferenceClient, chat_completions_url


async def test_upstream_client_adds_auth_header_and_default_model() -> None:
    seen_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(nvidia_api_key="secret", upstream_base_url="https://upstream.test", upstream_model="default-model")
    client = NvidiaInferenceClient(settings=settings, client=http_client)

    response = await client.chat_completions({"messages": []})

    assert response.status_code == 200
    assert response.body == {"choices": [{"message": {"content": "ok"}}]}
    assert seen_request is not None
    assert str(seen_request.url) == "https://upstream.test/v1/chat/completions"
    assert seen_request.headers["authorization"] == "Bearer secret"
    assert seen_request.headers["content-type"] == "application/json"
    assert b'"model":"default-model"' in seen_request.content

    await http_client.aclose()


def test_chat_completions_url_accepts_base_url_with_or_without_v1() -> None:
    assert chat_completions_url("https://inference-api.nvidia.com") == (
        "https://inference-api.nvidia.com/v1/chat/completions"
    )
    assert chat_completions_url("https://inference-api.nvidia.com/v1") == (
        "https://inference-api.nvidia.com/v1/chat/completions"
    )
    assert chat_completions_url("https://inference-api.nvidia.com/v1/") == (
        "https://inference-api.nvidia.com/v1/chat/completions"
    )


async def test_upstream_client_requires_api_key() -> None:
    client = NvidiaInferenceClient(settings=Settings(nvidia_api_key=None))

    try:
        await client.chat_completions({"messages": []})
    except RuntimeError as error:
        assert "NVIDIA_API_KEY" in str(error)
    else:
        raise AssertionError("expected missing API key error")


async def test_upstream_client_can_call_local_openai_compatible_server_without_auth() -> None:
    seen_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json={"choices": []})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = NvidiaInferenceClient(
        settings=Settings(),
        client=http_client,
        base_url="http://localhost:1234",
        model="qwen/qwen3.6-27b",
        api_key=None,
        require_api_key=False,
    )

    response = await client.chat_completions({"messages": []})

    assert response.status_code == 200
    assert seen_request is not None
    assert str(seen_request.url) == "http://localhost:1234/v1/chat/completions"
    assert "authorization" not in seen_request.headers
    assert b'"model":"qwen/qwen3.6-27b"' in seen_request.content

    await http_client.aclose()


async def test_upstream_client_logs_url_status_and_body(caplog) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(nvidia_api_key="secret", upstream_base_url="https://upstream.test/v1")
    client = NvidiaInferenceClient(settings=settings, client=http_client)

    with caplog.at_level(logging.INFO, logger="redact_spike.upstream"):
        response = await client.chat_completions({"model": "openai/openai/gpt-5.5", "messages": []})

    assert response.status_code == 404
    assert "upstream_request url=https://upstream.test/v1/chat/completions" in caplog.text
    assert "model=openai/openai/gpt-5.5" in caplog.text
    assert "upstream_response url=https://upstream.test/v1/chat/completions status=404" in caplog.text
    assert "not found" in caplog.text

    await http_client.aclose()
