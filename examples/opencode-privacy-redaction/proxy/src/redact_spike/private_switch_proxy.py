# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uvicorn

from redact_spike.config import Settings
from redact_spike.proxy import create_app


app = create_app(private_model_switch=True)


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "redact_spike.private_switch_proxy:app",
        host=settings.proxy_host,
        port=settings.private_proxy_port,
        reload=False,
    )
