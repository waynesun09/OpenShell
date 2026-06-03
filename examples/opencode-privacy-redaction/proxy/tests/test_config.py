# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from redact_spike.config import Settings


def test_settings_loads_dotenv_file(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NVIDIA_API_KEY=from-dotenv\nUPSTREAM_MODEL=model-from-dotenv\nPROXY_PORT=9999\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("UPSTREAM_MODEL", raising=False)
    monkeypatch.delenv("PROXY_PORT", raising=False)

    settings = Settings.from_env()

    assert settings.nvidia_api_key == "from-dotenv"
    assert settings.upstream_model == "model-from-dotenv"
    assert settings.proxy_port == 9999


def test_dotenv_values_override_environment_values(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NVIDIA_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("NVIDIA_API_KEY", "from-environment")

    settings = Settings.from_env()

    assert settings.nvidia_api_key == "from-dotenv"
