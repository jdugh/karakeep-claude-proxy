# syntax=docker/dockerfile:1
#
# Claude Code is installed via npm, pinned to a version verified against this
# project (see README "Choix techniques" for why npm was chosen over the
# native installer: it lets us pin an exact, reproducible version here).
#
# Python dependencies are installed from uv.lock, so `docker compose build`
# installs the exact same dependency versions today and months from now,
# until uv.lock is regenerated on purpose (see README "Verrouillage des
# dépendances Python"). pyproject.toml keeps its version ranges as intent;
# uv.lock is the hash-pinned source of truth for the actual build.
FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim

ARG CLAUDE_CODE_VERSION=2.1.292
ARG NODE_MAJOR=20

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

COPY --from=uv /uv /usr/local/bin/uv

# --- Node.js + the official Claude Code CLI, pinned -------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/* /root/.npm

# --- Application: dependencies first (cached layer), then app code ---------
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
COPY app ./app
RUN uv sync --locked

# Default CLAUDE_WEB_PROMPT_FILE content, baked in so web mode works with no
# extra volume mount; override by bind-mounting ./prompts:/app/prompts:ro
# (see README "Web tools" / docker-compose.yml comment).
COPY prompts ./prompts

ENV PATH="/app/.venv/bin:${PATH}"

# --- Hermetic, non-root runtime user -------------------------------------
# /app/runtime is an empty, controlled cwd for the `claude` subprocess: no
# user repository, no CLAUDE.md, no plugins, no MCP config (PROMPT.md 3.3).
RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /app/runtime \
    && chown -R app:app /app /home/app

ENV HOME=/home/app
USER app
WORKDIR /app/runtime

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
