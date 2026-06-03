# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str | None = None
    upstream_base_url: str = "https://inference-api.nvidia.com"
    upstream_model: str = "openai/openai/gpt-5.5"
    private_upstream_base_url: str = "http://localhost:1234"
    private_upstream_model: str = "qwen/qwen3.6-27b"
    private_upstream_api_key: str | None = None
    private_switch_force_model: bool = True
    mcp_force_upstream_model: bool = True
    mcp_transport: str = "http"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8000
    private_proxy_port: int = 8002
    proxy_log_level: str = "INFO"
    request_log_dir: Path = Path("request-logs")
    redaction_enabled: bool = True
    store_redacted_images: bool = True
    redacted_image_dir: Path = Path(tempfile.gettempdir()) / "redact-spike-images"
    content_policy_file: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_file()
        content_policy_file = os.getenv("CONTENT_POLICY_FILE")
        return cls(
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            upstream_base_url=os.getenv("UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_model=os.getenv("UPSTREAM_MODEL", cls.upstream_model),
            private_upstream_base_url=os.getenv("PRIVATE_UPSTREAM_BASE_URL", cls.private_upstream_base_url),
            private_upstream_model=os.getenv("PRIVATE_UPSTREAM_MODEL", cls.private_upstream_model),
            private_upstream_api_key=os.getenv("PRIVATE_UPSTREAM_API_KEY"),
            private_switch_force_model=_env_bool("PRIVATE_SWITCH_FORCE_MODEL", cls.private_switch_force_model),
            mcp_force_upstream_model=_env_bool("MCP_FORCE_UPSTREAM_MODEL", cls.mcp_force_upstream_model),
            mcp_transport=os.getenv("MCP_TRANSPORT", cls.mcp_transport),
            mcp_host=os.getenv("MCP_HOST", cls.mcp_host),
            mcp_port=int(os.getenv("MCP_PORT", str(cls.mcp_port))),
            proxy_host=os.getenv("PROXY_HOST", cls.proxy_host),
            proxy_port=int(os.getenv("PROXY_PORT", str(cls.proxy_port))),
            private_proxy_port=int(os.getenv("PRIVATE_PROXY_PORT", str(cls.private_proxy_port))),
            proxy_log_level=os.getenv("PROXY_LOG_LEVEL", cls.proxy_log_level),
            request_log_dir=Path(os.getenv("REQUEST_LOG_DIR", str(cls.request_log_dir))),
            redaction_enabled=_env_bool("REDACTION_ENABLED", cls.redaction_enabled),
            store_redacted_images=_env_bool("STORE_REDACTED_IMAGES", cls.store_redacted_images),
            redacted_image_dir=Path(os.getenv("REDACTED_IMAGE_DIR", str(cls.redacted_image_dir))),
            content_policy_file=Path(content_policy_file) if content_policy_file else None,
        )


def _load_dotenv_file() -> None:
    env_file = Path(os.getenv("ENV_FILE", ".env"))
    if env_file.exists():
        load_dotenv(env_file, override=True)
