# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from redact_spike.mcp_server import call_inference_impl


async def _run(payload: dict[str, Any]) -> None:
    response = await call_inference_impl(payload)
    print(json.dumps(response, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the MCP inference implementation directly for debugging.")
    parser.add_argument("--payload", help="OpenAI-style JSON payload. Reads stdin when omitted.")
    args = parser.parse_args()
    raw_payload = args.payload if args.payload is not None else sys.stdin.read()
    payload = json.loads(raw_payload)
    asyncio.run(_run(payload))


if __name__ == "__main__":
    main()
