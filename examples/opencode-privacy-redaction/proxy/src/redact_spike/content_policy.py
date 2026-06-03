# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageContentRule:
    """Resolved `content_policy.content_type` decision for image payloads.

    `action` is "redact" or "allow". `description` is the operator-authored note
    that the proxy substitutes at the redaction site (the `redact_description`
    field), telling the main model how to recover the data via the MCP tool.
    """

    action: str
    description: str | None = None

    @property
    def should_redact(self) -> bool:
        return self.action == "redact"


# Default when no content policy is in force: images pass through untouched.
ALLOW = ImageContentRule(action="allow")


def load_image_content_rule(path: Path | None) -> ImageContentRule | None:
    """Read the image content rule from a policy file, re-read on every call.

    Returns None when no decision is expressed (no path, missing file, no
    `content_policy.content_type.image` key) so the caller can fall back to its
    own default. This is what makes the demo a live toggle: edit the file and
    the next request observes the new rule, no restart required.

    The `content_policy` block is an OpenShell-style extension that the OpenShell
    policy engine ignores today; only this proxy interprets it.
    """
    if path is None:
        return None

    if not path.exists():
        logger.info("content_policy_absent path=%s -> no rule", path)
        return None

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        logger.warning("content_policy_unreadable path=%s error=%s -> no rule", path, error)
        return None

    content_type = (((document or {}).get("content_policy") or {}).get("content_type")) or {}
    action = content_type.get("image")
    if not isinstance(action, str):
        return None

    description = content_type.get("redact_description")
    rule = ImageContentRule(
        action=action.strip().lower(),
        description=description if isinstance(description, str) else None,
    )
    logger.info("content_policy_loaded path=%s image_action=%s", path, rule.action)
    return rule
