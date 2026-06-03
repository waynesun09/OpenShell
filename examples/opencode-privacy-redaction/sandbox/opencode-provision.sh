#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Provision opencode inside an OpenShell sandbox, wired to inference.local plus
# the host-side local-model MCP. opencode is preinstalled on the base image.
# The opencode.json config is fed on stdin (ssh '...' < opencode.json).
set -e
export HOME=/sandbox

CFG_DIR="$HOME/.config/opencode"
mkdir -p "$CFG_DIR"

# opencode treats its working directory as the project root and needs one to
# start a session in.
WORKSPACE="$HOME/workspace"
echo "== creating opencode workspace ($WORKSPACE) =="
mkdir -p "$WORKSPACE"

echo "== writing opencode config ($CFG_DIR/opencode.json) =="
cat > "$CFG_DIR/opencode.json"

echo "== opencode version =="
opencode --version

echo "== configured providers/models =="
opencode models 2>&1 | grep -i inference-local || true

echo "== registered MCP servers =="
opencode mcp list 2>&1 || true

echo "== provision complete =="
