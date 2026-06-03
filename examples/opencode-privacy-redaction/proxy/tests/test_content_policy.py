# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from redact_spike.content_policy import load_image_content_rule


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "content-policy.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_returns_none_when_path_missing(tmp_path):
    assert load_image_content_rule(None) is None
    assert load_image_content_rule(tmp_path / "nope.yaml") is None


def test_returns_none_without_image_key(tmp_path):
    path = _write(tmp_path, "version: 1\ncontent_policy:\n  content_type: {}\n")
    assert load_image_content_rule(path) is None


def test_redact_rule_with_description(tmp_path):
    path = _write(
        tmp_path,
        "content_policy:\n"
        "  content_type:\n"
        "    image: redact\n"
        "    redact_description: Use the local_model MCP server.\n",
    )
    rule = load_image_content_rule(path)
    assert rule is not None
    assert rule.should_redact is True
    assert rule.description == "Use the local_model MCP server."


def test_allow_rule(tmp_path):
    path = _write(tmp_path, "content_policy:\n  content_type:\n    image: allow\n")
    rule = load_image_content_rule(path)
    assert rule is not None
    assert rule.should_redact is False


def test_live_toggle_is_reread(tmp_path):
    path = _write(tmp_path, "content_policy:\n  content_type:\n    image: allow\n")
    assert load_image_content_rule(path).should_redact is False
    path.write_text("content_policy:\n  content_type:\n    image: redact\n", encoding="utf-8")
    assert load_image_content_rule(path).should_redact is True
