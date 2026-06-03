# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_URL_RE = re.compile(r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$", re.DOTALL)
IMAGE_KEYWORDS = {"image", "image_url", "input_image"}
REDACTION_CONTEXT = (
    "This image was redacted before this request was sent to you. "
    "If the user's task requires information from a redacted image, use the pre-installed MCP tool `call_inference` "
    "to request analysis from the allowed local image-capable model. The MCP request must include a text part plus actual "
    "image data or an `image_reference` path from this context; a placeholder ID alone is not image data."
)


@dataclass(frozen=True)
class RedactedImage:
    placeholder: str
    source: str
    reference: str | None = None


@dataclass(frozen=True)
class RedactionResult:
    payload: dict[str, Any]
    images: list[RedactedImage]


class DiskImageStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def store(self, placeholder: str, image_value: Any) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(image_value, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(serialized).hexdigest()[:16]

        data_url = _find_data_url(image_value)
        if data_url:
            match = DATA_URL_RE.match(data_url)
            if match:
                extension = _extension_for_mime(match.group("mime"))
                path = self.directory / f"{placeholder.lower()}-{digest}.{extension}"
                path.write_bytes(base64.b64decode(match.group("data")))
                return str(path)

        path = self.directory / f"{placeholder.lower()}-{digest}.json"
        path.write_text(json.dumps(image_value, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)


def redact_images(
    payload: dict[str, Any],
    store: DiskImageStore | None = None,
    replacement_style: str = "mcp",
    redaction_context: str | None = None,
) -> RedactionResult:
    redacted_payload = copy.deepcopy(payload)
    images: list[RedactedImage] = []

    for message in redacted_payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                _redact_content_part(part, images, store, replacement_style, redaction_context) for part in content
            ]
        elif isinstance(content, str) and _looks_like_image_data_url(content):
            message["content"] = _replacement_text(
                _record_image(content, "message.content", images, store), replacement_style, redaction_context
            )

    return RedactionResult(payload=redacted_payload, images=images)


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "roles": [],
        "content_part_types": [],
        "has_image": False,
    }

    if not isinstance(messages, list):
        return summary

    content_part_types: list[str] = []
    roles: list[str] = []
    has_image = False

    for message in messages:
        if not isinstance(message, dict):
            continue
        roles.append(str(message.get("role", "unknown")))
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type", "unknown"))
                    content_part_types.append(part_type)
                    has_image = has_image or _is_image_part(part)
        elif isinstance(content, str):
            content_part_types.append("text")
            has_image = has_image or _looks_like_image_data_url(content)

    summary["roles"] = roles
    summary["content_part_types"] = content_part_types
    summary["has_image"] = has_image
    return summary


def last_message_has_image(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    last_message = messages[-1]
    if not isinstance(last_message, dict):
        return False

    return message_has_image(last_message)


def message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, list):
        return any(isinstance(part, dict) and _is_image_part(part) for part in content)
    if isinstance(content, str):
        return _looks_like_image_data_url(content)
    return False


def _redact_content_part(
    part: Any,
    images: list[RedactedImage],
    store: DiskImageStore | None,
    replacement_style: str,
    redaction_context: str | None = None,
) -> Any:
    if not isinstance(part, dict):
        return part

    if _is_image_part(part):
        image = _record_image(part, "message.content[]", images, store)
        return {"type": "text", "text": _replacement_text(image, replacement_style, redaction_context)}

    return part


def _is_image_part(part: dict[str, Any]) -> bool:
    part_type = str(part.get("type", "")).lower()
    if part_type in IMAGE_KEYWORDS:
        return True
    if "image_url" in part or "input_image" in part:
        return True
    for value in part.values():
        if isinstance(value, str) and _looks_like_image_data_url(value):
            return True
        if isinstance(value, dict) and any(str(key).lower() in IMAGE_KEYWORDS for key in value):
            return True
    return False


def _looks_like_image_data_url(value: str) -> bool:
    return bool(DATA_URL_RE.match(value.strip()))


def _find_data_url(value: Any) -> str | None:
    if isinstance(value, str) and _looks_like_image_data_url(value):
        return value.strip()
    if isinstance(value, dict):
        for nested in value.values():
            data_url = _find_data_url(nested)
            if data_url:
                return data_url
    if isinstance(value, list):
        for nested in value:
            data_url = _find_data_url(nested)
            if data_url:
                return data_url
    return None


def _record_image(
    image_value: Any,
    source: str,
    images: list[RedactedImage],
    store: DiskImageStore | None,
) -> RedactedImage:
    placeholder = f"[REDACTED_IMAGE_{len(images) + 1}]"
    reference = store.store(placeholder.strip("[]"), image_value) if store else None
    image = RedactedImage(placeholder=placeholder, source=source, reference=reference)
    images.append(image)
    return image


def _replacement_text(image: RedactedImage, replacement_style: str, redaction_context: str | None = None) -> str:
    text = f"{image.placeholder} image content redacted before upstream inference."
    if replacement_style == "short":
        return text

    text += f" {redaction_context or REDACTION_CONTEXT}"
    if image.reference:
        text += (
            f" Local image reference available to MCP for {image.placeholder}: {image.reference}. "
            "Use MCP payload content like "
            "[{\"type\":\"text\",\"text\":\"Describe the image.\"}, "
            f"{{\"type\":\"image_reference\",\"path\":\"{image.reference}\"}}]."
        )
    return text


def _extension_for_mime(mime: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }.get(mime.lower(), "bin")
