# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from redact_spike.config import Settings
from redact_spike.content_policy import load_image_content_rule
from redact_spike.redaction import DiskImageStore, last_message_has_image, redact_images, summarize_payload
from redact_spike.upstream import NvidiaInferenceClient, UpstreamResponse


class ChatCompletionsClient(Protocol):
    async def chat_completions(self, payload: dict[str, Any]) -> UpstreamResponse:
        pass

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        pass


logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    upstream_client: ChatCompletionsClient | None = None,
    private_upstream_client: ChatCompletionsClient | None = None,
    private_model_switch: bool = False,
) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.proxy_log_level.upper())

    app = FastAPI(title="Redacted Inference Proxy", version="0.1.0")
    app.state.settings = settings
    app.state.upstream_client = upstream_client or NvidiaInferenceClient(settings)
    app.state.private_upstream_client = private_upstream_client or NvidiaInferenceClient(
        settings,
        base_url=settings.private_upstream_base_url,
        model=settings.private_upstream_model,
        api_key=settings.private_upstream_api_key,
        require_api_key=False,
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        started_at = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        headers = _sanitized_headers(request)
        body = await request.body()
        request_log_path = _write_request_file(settings.request_log_dir, request, headers, body, request_id)
        logger.info(
            "request_started request_id=%s method=%s path=%s query=%s headers=%s request_log=%s",
            request_id,
            request.method,
            request.url.path,
            request.url.query,
            headers,
            request_log_path,
        )
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "request_finished request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.upstream_model,
                    "object": "model",
                    "owned_by": "nvidia",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    @app.post("/v1/chat/completion")
    @app.post("/chat/completions")
    @app.post("/chat/completion")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object request body")

        payload_summary = summarize_payload(payload)
        logger.info(
            "chat_completion_received model=%s message_count=%s has_image=%s content_part_types=%s",
            payload_summary["model"],
            payload_summary["message_count"],
            payload_summary["has_image"],
            payload_summary["content_part_types"],
        )

        outbound_payload, use_private_upstream = _build_outbound_payload(payload, settings, private_model_switch)

        outbound_log_path = _write_json_capture_file(
            settings.request_log_dir,
            request.state.request_id,
            "outbound-redacted-request",
            outbound_payload,
        )
        logger.info("redacted_request_captured request_id=%s path=%s", request.state.request_id, outbound_log_path)

        if outbound_payload.get("stream") is True:
            logger.info("calling_upstream stream=true private_upstream=%s", use_private_upstream)
            upstream_client = app.state.private_upstream_client if use_private_upstream else app.state.upstream_client
            chunks = upstream_client.stream_chat_completions(outbound_payload)
            if use_private_upstream:
                chunks = _normalize_reasoning_stream(chunks)
            stream = _capture_streaming_response(
                chunks,
                settings.request_log_dir,
                request.state.request_id,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
            )

        logger.info("calling_upstream stream=false private_upstream=%s", use_private_upstream)
        upstream_client = app.state.private_upstream_client if use_private_upstream else app.state.upstream_client
        response = await upstream_client.chat_completions(outbound_payload)
        response_body = _normalize_reasoning_response(response.body) if use_private_upstream else response.body
        response_log_path = _write_json_capture_file(
            settings.request_log_dir,
            request.state.request_id,
            "upstream-response",
            {"status_code": response.status_code, "body": response_body},
        )
        logger.info("upstream_response_captured request_id=%s path=%s", request.state.request_id, response_log_path)
        return JSONResponse(status_code=response.status_code, content=response_body)

    return app


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("redact_spike.proxy:app", host=settings.proxy_host, port=settings.proxy_port, reload=False)


def _build_outbound_payload(
    payload: dict[str, Any],
    settings: Settings,
    private_model_switch: bool,
) -> tuple[dict[str, Any], bool]:
    if private_model_switch and last_message_has_image(payload):
        outbound_payload = dict(payload)
        if settings.private_switch_force_model:
            outbound_payload["model"] = settings.private_upstream_model
        logger.info(
            "private_model_switch_applied private_base_url=%s private_model=%s force_model=%s",
            settings.private_upstream_base_url,
            outbound_payload.get("model"),
            settings.private_switch_force_model,
        )
        return outbound_payload, True

    if private_model_switch:
        logger.info("private_model_switch_skipped reason=last_message_has_no_image")

    should_redact, redaction_context = _resolve_image_redaction(settings)

    if should_redact:
        replacement_style = "short" if private_model_switch else "mcp"
        logger.info(
            "image_redaction_started store_redacted_images=%s replacement_style=%s custom_context=%s",
            settings.store_redacted_images,
            replacement_style,
            redaction_context is not None,
        )
        store = DiskImageStore(settings.redacted_image_dir) if settings.store_redacted_images and not private_model_switch else None
        redaction_result = redact_images(
            payload, store=store, replacement_style=replacement_style, redaction_context=redaction_context
        )
        if redaction_result.images:
            logger.info(
                "image_redaction_applied count=%s placeholders=%s references=%s",
                len(redaction_result.images),
                [image.placeholder for image in redaction_result.images],
                [image.reference for image in redaction_result.images if image.reference],
            )
        else:
            logger.info("image_redaction_skipped reason=no_images_found")
        return redaction_result.payload, False

    logger.info("image_redaction_skipped reason=allowed_by_policy")
    return payload, False


def _resolve_image_redaction(settings: Settings) -> tuple[bool, str | None]:
    """Decide whether to redact images on this request.

    When a `CONTENT_POLICY_FILE` is configured, the file is the source of truth
    and is re-read per request (the live demo toggle). Its
    `content_policy.content_type.image` value (`redact`/`allow`) decides, and
    `redact_description` overrides the note injected at the redaction site.
    With no content-policy decision, fall back to the static `REDACTION_ENABLED`.
    """
    rule = load_image_content_rule(settings.content_policy_file)
    if rule is not None:
        return rule.should_redact, rule.description
    return settings.redaction_enabled, None


def _sanitized_headers(request: Request) -> dict[str, str]:
    allowed_headers = {"accept", "content-type", "user-agent", "host"}
    secret_headers = {"authorization", "x-api-key", "api-key"}
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower_key = key.lower()
        if lower_key in secret_headers:
            headers[lower_key] = "[REDACTED]"
        elif lower_key in allowed_headers:
            headers[lower_key] = value
    return headers


def _write_request_file(log_dir: Path, request: Request, headers: dict[str, str], body: bytes, request_id: str) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_path = re.sub(r"[^a-zA-Z0-9._-]+", "-", request.url.path.strip("/") or "root")
    path = log_dir / f"{timestamp}-{request_id}-incoming-{request.method.lower()}-{safe_path}.json"

    content_type = request.headers.get("content-type", "")
    decoded_body = body.decode("utf-8", errors="replace")
    record: dict[str, Any] = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query": request.url.query,
        "headers": headers,
        "content_type": content_type,
        "body_bytes": len(body),
        "body_text": decoded_body,
    }

    if "application/json" in content_type.lower() and body:
        try:
            record["body_json"] = json.loads(body)
        except json.JSONDecodeError:
            record["body_json_parse_error"] = "invalid_json"

    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_json_capture_file(log_dir: Path, request_id: str, label: str, payload: Any) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = log_dir / f"{timestamp}-{request_id}-{label}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


async def _capture_streaming_response(
    chunks: AsyncIterator[bytes],
    log_dir: Path,
    request_id: str,
) -> AsyncIterator[bytes]:
    captured = bytearray()
    try:
        async for chunk in chunks:
            captured.extend(chunk)
            yield chunk
    finally:
        path = _write_json_capture_file(
            log_dir,
            request_id,
            "upstream-stream-response",
            {
                "body_text": captured.decode("utf-8", errors="replace"),
                "body_bytes": len(captured),
            },
        )
        logger.info("upstream_response_captured request_id=%s path=%s", request_id, path)


async def _normalize_reasoning_stream(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = ""
    async for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield (_normalize_sse_line(line) + "\n").encode("utf-8")
    if buffer:
        yield _normalize_sse_line(buffer).encode("utf-8")


def _normalize_sse_line(line: str) -> str:
    if not line.startswith("data: "):
        return line

    data = line.removeprefix("data: ")
    if data.strip() == "[DONE]":
        return line

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return line

    return "data: " + json.dumps(_normalize_reasoning_response(payload), separators=(",", ":"))


def _normalize_reasoning_response(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            message = choice.get(key)
            if isinstance(message, dict):
                _copy_reasoning_to_provider_specific_fields(message)
    return payload


def _copy_reasoning_to_provider_specific_fields(message: dict[str, Any]) -> None:
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is None:
        return

    provider_specific_fields = message.get("provider_specific_fields")
    if not isinstance(provider_specific_fields, dict):
        provider_specific_fields = {}
        message["provider_specific_fields"] = provider_specific_fields
    provider_specific_fields.setdefault("reasoning_content", reasoning_content)
