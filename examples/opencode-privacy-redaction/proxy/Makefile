ENV_FILE ?= $(if $(wildcard .env),.env,.env.example)
CONTENT_POLICY ?= ../policies/content-policy.yaml

.PHONY: start-proxy stop-proxy restart-proxy tail-proxy-log \
        start-mcp stop-mcp restart-mcp tail-mcp-log \
        start stop demo-allow demo-redact demo-status test

## --- proxy (OpenAI-compatible, inference.local upstream + image redaction) ---
start-proxy:
	@if [ -f proxy.pid ] && kill -0 "$$(cat proxy.pid)" 2>/dev/null; then \
		echo "proxy already running: pid=$$(cat proxy.pid)"; \
	else \
		rm -f proxy.log proxy.pid; \
		ENV_FILE="$(ENV_FILE)" uv run redact-proxy > proxy.log 2>&1 & \
		echo $$! > proxy.pid; \
		echo "proxy started: pid=$$(cat proxy.pid), log=proxy.log, env=$(ENV_FILE)"; \
	fi

stop-proxy:
	@if [ -f proxy.pid ] && kill -0 "$$(cat proxy.pid)" 2>/dev/null; then \
		kill "$$(cat proxy.pid)"; echo "proxy stopped: pid=$$(cat proxy.pid)"; rm -f proxy.pid; \
	else echo "proxy not running"; rm -f proxy.pid; fi

restart-proxy: stop-proxy start-proxy

tail-proxy-log:
	tail -f proxy.log

## --- MCP (call_inference -> private/local model) ---
start-mcp:
	@if [ -f mcp.pid ] && kill -0 "$$(cat mcp.pid)" 2>/dev/null; then \
		echo "mcp already running: pid=$$(cat mcp.pid)"; \
	else \
		rm -f mcp.log mcp.pid; \
		ENV_FILE="$(ENV_FILE)" uv run redact-mcp > mcp.log 2>&1 & \
		echo $$! > mcp.pid; \
		echo "mcp started: pid=$$(cat mcp.pid), log=mcp.log, env=$(ENV_FILE)"; \
	fi

stop-mcp:
	@if [ -f mcp.pid ] && kill -0 "$$(cat mcp.pid)" 2>/dev/null; then \
		kill "$$(cat mcp.pid)"; echo "mcp stopped: pid=$$(cat mcp.pid)"; rm -f mcp.pid; \
	else echo "mcp not running"; rm -f mcp.pid; fi

restart-mcp: stop-mcp start-mcp

tail-mcp-log:
	tail -f mcp.log

## --- convenience ---
start: start-mcp start-proxy
stop: stop-proxy stop-mcp

## --- demo magic: flip the content policy live (proxy re-reads per request) ---
demo-allow:
	@sed -i.bak -E 's/^( *image: *).*/\1allow/' "$(CONTENT_POLICY)" && rm -f "$(CONTENT_POLICY).bak"
	@echo "content policy -> image: allow (images pass through to the public model)"

demo-redact:
	@sed -i.bak -E 's/^( *image: *).*/\1redact/' "$(CONTENT_POLICY)" && rm -f "$(CONTENT_POLICY).bak"
	@echo "content policy -> image: redact (images stripped; routed via local-model MCP)"

demo-status:
	@grep -E '^ *image:' "$(CONTENT_POLICY)" || echo "no image rule found in $(CONTENT_POLICY)"

test:
	uv run pytest
