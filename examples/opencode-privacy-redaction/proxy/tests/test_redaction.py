# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

from redact_spike.redaction import DiskImageStore, last_message_has_image, redact_images, summarize_payload


def image_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(b"fake png bytes").decode("ascii")


def test_redacts_image_parts_and_adds_mcp_context_at_image_location() -> None:
    payload = {
        "model": "openai/openai/gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Am I dressed for rain?"},
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            },
        ],
    }

    result = redact_images(payload)

    assert len(result.images) == 1
    assert result.images[0].placeholder == "[REDACTED_IMAGE_1]"
    assert image_data_url() not in str(result.payload)
    assert result.payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    replacement = result.payload["messages"][1]["content"][1]
    assert replacement["type"] == "text"
    assert "This image was redacted" in replacement["text"]
    assert "pre-installed MCP tool `call_inference`" in replacement["text"]
    assert "[REDACTED_IMAGE_1]" in replacement["text"]


def test_can_store_redacted_data_urls_on_disk(tmp_path) -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            }
        ]
    }

    result = redact_images(payload, store=DiskImageStore(tmp_path))

    assert result.images[0].reference is not None
    assert result.images[0].reference.endswith(".png")
    assert (tmp_path / result.images[0].reference.split("/")[-1]).read_bytes() == b"fake png bytes"
    replacement = result.payload["messages"][0]["content"][1]
    assert "Local image reference available to MCP" in replacement["text"]
    assert str(tmp_path) in replacement["text"]
    assert image_data_url() not in str(result.payload)


def test_short_redaction_omits_mcp_context() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            }
        ]
    }

    result = redact_images(payload, replacement_style="short")

    assert result.payload["messages"][0]["content"][1] == {
        "type": "text",
        "text": "[REDACTED_IMAGE_1] image content redacted before upstream inference.",
    }


def test_last_message_has_image_only_checks_latest_message() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Old image."},
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            },
            {"role": "assistant", "content": "What should I do with it?"},
            {"role": "user", "content": "Never mind, answer without the image."},
        ]
    }

    assert last_message_has_image(payload) is False

    payload["messages"][-1] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Now describe this."},
            {"type": "image_url", "image_url": {"url": image_data_url()}},
        ],
    }

    assert last_message_has_image(payload) is True


def test_summarize_payload_reports_structure_without_prompt_text() -> None:
    payload = {
        "model": "model-a",
        "messages": [
            {"role": "system", "content": "secret system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "secret prompt text"},
                    {"type": "image_url", "image_url": {"url": image_data_url()}},
                ],
            },
        ],
    }

    summary = summarize_payload(payload)

    assert summary == {
        "model": "model-a",
        "message_count": 2,
        "roles": ["system", "user"],
        "content_part_types": ["text", "text", "image_url"],
        "has_image": True,
    }
    assert "secret" not in str(summary)
    assert image_data_url() not in str(summary)
