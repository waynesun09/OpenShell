# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from redact_spike import mcp_server
from redact_spike.config import Settings
from redact_spike.upstream import UpstreamResponse


class FakeNvidiaInferenceClient:
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    async def chat_completions(self, payload: dict[str, Any]) -> UpstreamResponse:
        return UpstreamResponse(status_code=200, body={"echo": payload, "model": self.kwargs["model"], "base_url": self.kwargs["base_url"]})


async def test_call_inference_impl_returns_openai_shaped_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "NvidiaInferenceClient", FakeNvidiaInferenceClient)

    payload = {"messages": [{"role": "user", "content": "describe image"}]}
    response = await mcp_server.call_inference_impl(
        payload,
        settings=Settings(private_upstream_base_url="http://localhost:1234", private_upstream_model="qwen/qwen3.6-27b"),
    )

    assert response == {
        "status_code": 200,
        "body": {
            "echo": {"messages": [{"role": "user", "content": "describe image"}], "model": "qwen/qwen3.6-27b"},
            "model": "qwen/qwen3.6-27b",
            "base_url": "http://localhost:1234",
        },
    }


def test_normalize_mcp_payload_forces_configured_model() -> None:
    payload = {
        "model": "local-image-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    normalized = mcp_server.normalize_mcp_payload(payload, Settings(private_upstream_model="qwen/qwen3.6-27b"))

    assert normalized["model"] == "qwen/qwen3.6-27b"
    assert payload["model"] == "local-image-model"


def test_normalize_mcp_payload_converts_image_placeholder_to_text() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_placeholder", "id": "REDACTED_IMAGE_1"},
                ],
            }
        ]
    }

    normalized = mcp_server.normalize_mcp_payload(payload, Settings(private_upstream_model="qwen/qwen3.6-27b"))

    assert normalized["model"] == "qwen/qwen3.6-27b"
    assert normalized["messages"][0]["content"] == [
        {"type": "text", "text": "Describe the image."},
        {
            "type": "text",
            "text": (
                "Image placeholder [REDACTED_IMAGE_1] was referenced, but no image bytes or local image reference were "
                "provided to this MCP call. Retry with a text part plus an image_reference part using the local path "
                "from the redaction context."
            ),
        },
    ]


def test_normalize_mcp_payload_converts_image_reference_to_data_url(tmp_path) -> None:
    image = tmp_path / "redacted.png"
    image.write_bytes(b"fake image bytes")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_reference", "path": str(image)},
                ],
            }
        ]
    }

    normalized = mcp_server.normalize_mcp_payload(payload, Settings(private_upstream_model="qwen/qwen3.6-27b"))

    assert normalized["messages"][0]["content"] == [
        {"type": "text", "text": "Describe the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZSBpbWFnZSBieXRlcw=="}},
    ]


def test_create_mcp_server() -> None:
    server = mcp_server.create_mcp_server(settings=Settings(nvidia_api_key="test"))

    assert server.name == "redacted-inference"
