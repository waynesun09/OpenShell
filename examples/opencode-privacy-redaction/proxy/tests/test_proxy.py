# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import httpx

from redact_spike.config import Settings
from redact_spike.proxy import create_app
from redact_spike.upstream import UpstreamResponse


class RecordingUpstream:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> UpstreamResponse:
        self.payloads.append(payload)
        return UpstreamResponse(status_code=200, body={"id": "chatcmpl-test", "choices": []})

    async def stream_chat_completions(self, payload: dict[str, Any]):
        self.payloads.append(payload)
        yield b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        yield b"data: [DONE]\n\n"


class ReasoningUpstream:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> UpstreamResponse:
        self.payloads.append(payload)
        return UpstreamResponse(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "private reasoning",
                        }
                    }
                ]
            },
        )

    async def stream_chat_completions(self, payload: dict[str, Any]):
        self.payloads.append(payload)
        yield b'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"private"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"reasoning_content":" reasoning"}}]}\n\n'
        yield b"data: [DONE]\n\n"


async def test_proxy_forwards_chat_completion_payload_when_redaction_disabled() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", redaction_enabled=False)
    app = create_app(settings=settings, upstream_client=upstream)
    payload = {
        "model": "openai/openai/gpt-5.5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl-test", "choices": []}
    assert upstream.payloads == [payload]


async def test_proxy_redacts_image_payload_before_forwarding() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", redaction_enabled=True)
    app = create_app(settings=settings, upstream_client=upstream)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ]
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    forwarded = upstream.payloads[0]
    assert "data:image/png" not in str(forwarded)
    assert "[REDACTED_IMAGE_1]" in str(forwarded)
    assert "pre-installed MCP tool `call_inference`" in forwarded["messages"][0]["content"][1]["text"]


async def test_proxy_streams_chat_completion_payloads() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", redaction_enabled=False)
    app = create_app(settings=settings, upstream_client=upstream)
    payload = {"stream": True, "messages": [{"role": "user", "content": "Hello"}]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
    assert upstream.payloads == [payload]


async def test_proxy_supports_chat_completion_route_aliases() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", redaction_enabled=False)
    app = create_app(settings=settings, upstream_client=upstream)
    payload = {"messages": [{"role": "user", "content": "Hello"}]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ["/chat/completions", "/chat/completion", "/v1/chat/completion"]:
            response = await client.post(path, json=payload)
            assert response.status_code == 200

    assert upstream.payloads == [payload, payload, payload]


async def test_proxy_lists_configured_model() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", upstream_model="openai/openai/gpt-5.5")
    app = create_app(settings=settings, upstream_client=upstream)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "openai/openai/gpt-5.5",
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


async def test_proxy_rejects_non_object_json() -> None:
    upstream = RecordingUpstream()
    app = create_app(settings=Settings(nvidia_api_key="test"), upstream_client=upstream)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=["not", "an", "object"])

    assert response.status_code == 400
    assert upstream.payloads == []


async def test_proxy_logs_unknown_requests(caplog) -> None:
    upstream = RecordingUpstream()
    app = create_app(settings=Settings(nvidia_api_key="test"), upstream_client=upstream)

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="redact_spike.proxy"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/definitely-unknown", headers={"Authorization": "Bearer local"})

    assert response.status_code == 404
    assert "method=GET path=/definitely-unknown" in caplog.text
    assert "path=/definitely-unknown status=404" in caplog.text
    assert "Bearer local" not in caplog.text
    assert "[REDACTED]" in caplog.text


async def test_proxy_writes_full_request_body_to_file(caplog, tmp_path) -> None:
    upstream = RecordingUpstream()
    app = create_app(settings=Settings(nvidia_api_key="test", request_log_dir=tmp_path), upstream_client=upstream)
    payload = {
        "model": "bad-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ],
    }

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="redact_spike.proxy"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "bad-model" in caplog.text
    assert "data:image/png" not in caplog.text
    request_log_files = list(tmp_path.glob("*incoming*.json"))
    outbound_log_files = list(tmp_path.glob("*outbound-redacted-request.json"))
    response_log_files = list(tmp_path.glob("*upstream-response.json"))
    assert len(request_log_files) == 1
    assert len(outbound_log_files) == 1
    assert len(response_log_files) == 1
    request_log = json.loads(request_log_files[0].read_text(encoding="utf-8"))
    outbound_log = json.loads(outbound_log_files[0].read_text(encoding="utf-8"))
    response_log = json.loads(response_log_files[0].read_text(encoding="utf-8"))
    assert request_log["path"] == "/v1/chat/completions"
    assert request_log["body_json"] == payload
    assert "data:image/png" in request_log["body_text"]
    assert "data:image/png" not in str(outbound_log)
    assert "[REDACTED_IMAGE_1]" in str(outbound_log)
    assert response_log == {"status_code": 200, "body": {"id": "chatcmpl-test", "choices": []}}


async def test_proxy_logs_redaction_and_upstream_lifecycle(caplog, tmp_path) -> None:
    upstream = RecordingUpstream()
    app = create_app(settings=Settings(nvidia_api_key="test", request_log_dir=tmp_path), upstream_client=upstream)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ]
    }

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="redact_spike.proxy"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "chat_completion_received" in caplog.text
    assert "image_redaction_started" in caplog.text
    assert "image_redaction_applied count=1" in caplog.text
    assert "redacted_request_captured" in caplog.text
    assert "calling_upstream stream=false" in caplog.text
    assert "upstream_response_captured" in caplog.text


async def test_proxy_captures_streaming_upstream_response(tmp_path) -> None:
    upstream = RecordingUpstream()
    app = create_app(
        settings=Settings(nvidia_api_key="test", redaction_enabled=False, request_log_dir=tmp_path),
        upstream_client=upstream,
    )
    payload = {"stream": True, "messages": [{"role": "user", "content": "Hello"}]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    response_log_files = list(tmp_path.glob("*upstream-stream-response.json"))
    assert len(response_log_files) == 1
    response_log = json.loads(response_log_files[0].read_text(encoding="utf-8"))
    assert response_log["body_text"] == "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"


async def test_private_switch_routes_last_message_image_to_private_model_without_redaction(tmp_path: Path) -> None:
    public_upstream = RecordingUpstream()
    private_upstream = RecordingUpstream()
    settings = Settings(
        nvidia_api_key="test",
        private_upstream_model="private-image-model",
        request_log_dir=tmp_path,
    )
    app = create_app(
        settings=settings,
        upstream_client=public_upstream,
        private_upstream_client=private_upstream,
        private_model_switch=True,
    )
    payload = {
        "model": "public-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            },
        ],
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert public_upstream.payloads == []
    forwarded = private_upstream.payloads[0]
    assert forwarded["model"] == "private-image-model"
    assert "data:image/png" in str(forwarded)


async def test_private_switch_redacts_older_images_when_last_message_has_no_image() -> None:
    upstream = RecordingUpstream()
    settings = Settings(nvidia_api_key="test", private_upstream_model="private-image-model")
    app = create_app(settings=settings, upstream_client=upstream, private_upstream_client=RecordingUpstream(), private_model_switch=True)
    payload = {
        "model": "public-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Old image."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            },
            {"role": "assistant", "content": "What should I do?"},
            {"role": "user", "content": "Answer without using that image."},
        ],
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    forwarded = upstream.payloads[0]
    assert forwarded["model"] == "public-model"
    assert "data:image/png" not in str(forwarded)
    assert "[REDACTED_IMAGE_1] image content redacted before upstream inference." in str(forwarded)
    assert "pre-installed MCP" not in str(forwarded)


async def test_private_switch_copies_reasoning_content_to_provider_specific_fields(tmp_path: Path) -> None:
    private_upstream = ReasoningUpstream()
    app = create_app(
        settings=Settings(nvidia_api_key="test", private_upstream_model="private-image-model", request_log_dir=tmp_path),
        upstream_client=RecordingUpstream(),
        private_upstream_client=private_upstream,
        private_model_switch=True,
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ]
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "private reasoning"
    assert message["provider_specific_fields"]["reasoning_content"] == "private reasoning"


async def test_private_switch_stream_copies_reasoning_content_to_provider_specific_fields(tmp_path: Path) -> None:
    private_upstream = ReasoningUpstream()
    app = create_app(
        settings=Settings(nvidia_api_key="test", private_upstream_model="private-image-model", request_log_dir=tmp_path),
        upstream_client=RecordingUpstream(),
        private_upstream_client=private_upstream,
        private_model_switch=True,
    )
    payload = {
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ],
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert b'"reasoning_content":"private"' in response.content
    assert b'"provider_specific_fields":{"reasoning_content":"private"}' in response.content
    assert b'"provider_specific_fields":{"reasoning_content":" reasoning"}' in response.content
