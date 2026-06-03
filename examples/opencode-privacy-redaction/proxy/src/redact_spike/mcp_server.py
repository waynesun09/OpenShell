# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import mimetypes
import base64
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from redact_spike.config import Settings
from redact_spike.upstream import NvidiaInferenceClient


logger = logging.getLogger(__name__)


async def call_inference_impl(payload: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    call_id = uuid.uuid4().hex[:8]
    incoming_path = _write_mcp_capture(settings.request_log_dir, call_id, "mcp-incoming-request", payload)
    outbound_payload = normalize_mcp_payload(payload, settings)
    outbound_path = _write_mcp_capture(settings.request_log_dir, call_id, "mcp-normalized-request", outbound_payload)
    logger.info(
        "mcp_call_inference call_id=%s model=%s original_model=%s message_count=%s incoming=%s normalized=%s",
        call_id,
        outbound_payload.get("model"),
        payload.get("model"),
        len(outbound_payload.get("messages", [])),
        incoming_path,
        outbound_path,
    )
    client = NvidiaInferenceClient(
        settings,
        base_url=settings.private_upstream_base_url,
        model=settings.private_upstream_model,
        api_key=settings.private_upstream_api_key,
        require_api_key=False,
    )
    response = await client.chat_completions(outbound_payload)
    response_path = _write_mcp_capture(
        settings.request_log_dir,
        call_id,
        "mcp-upstream-response",
        {"status_code": response.status_code, "body": response.body},
    )
    logger.info("mcp_call_inference_response call_id=%s status=%s response=%s", call_id, response.status_code, response_path)
    return {"status_code": response.status_code, "body": response.body}


def normalize_mcp_payload(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    if settings.mcp_force_upstream_model or not normalized.get("model"):
        normalized["model"] = settings.private_upstream_model

    messages = normalized.get("messages")
    if isinstance(messages, list):
        normalized["messages"] = [_normalize_message(message) for message in messages]
    return normalized


def _normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [_normalize_content_part(part) for part in content]
    return normalized


def _normalize_content_part(part: Any) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": f"Unsupported content part: {json.dumps(part, sort_keys=True)}"}

    part_type = part.get("type")
    if part_type in {"text", "image_url", "input_image"}:
        return part
    if part_type == "image_reference":
        return _image_reference_to_image_url(part)
    if part_type == "image_placeholder":
        placeholder_id = str(part.get("id", "UNKNOWN_IMAGE"))
        return {
            "type": "text",
            "text": (
                f"Image placeholder [{placeholder_id}] was referenced, but no image bytes or local image reference were "
                "provided to this MCP call. Retry with a text part plus an image_reference part using the local path "
                "from the redaction context."
            ),
        }
    return {"type": "text", "text": f"Unsupported content part: {json.dumps(part, sort_keys=True)}"}


def _image_reference_to_image_url(part: dict[str, Any]) -> dict[str, Any]:
    reference = part.get("path") or part.get("reference") or part.get("url")
    if not isinstance(reference, str):
        return {"type": "text", "text": "Invalid image_reference part: missing path, reference, or url."}

    path = Path(reference.removeprefix("file://"))
    if not path.exists() or not path.is_file():
        return {"type": "text", "text": f"Invalid image_reference part: file does not exist at {reference}."}

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def create_mcp_server(settings: Settings | None = None) -> FastMCP:
    mcp = FastMCP("redacted-inference")

    @mcp.tool
    async def call_inference(payload: dict[str, Any]) -> dict[str, Any]:
        """Call the configured OpenAI-compatible inference endpoint.

        Provide an OpenAI-style payload. The server will use the configured model, so do not invent model names.
        If an image was redacted, include a user message whose content has both a text part and an image part.
        Use {"type":"image_reference","path":"<local path from redaction context>"} when the redaction context provides a local image path.
        A placeholder such as REDACTED_IMAGE_1 alone is only an identifier, not image data.
        """
        return await call_inference_impl(payload, settings=settings)

    return mcp


mcp = create_mcp_server()


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.proxy_log_level.upper())
    if settings.mcp_transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=settings.mcp_transport, host=settings.mcp_host, port=settings.mcp_port)


def _write_mcp_capture(log_dir: Path, call_id: str, label: str, payload: Any) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = log_dir / f"{timestamp}-{call_id}-{label}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    main()
