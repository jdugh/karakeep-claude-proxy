# syntax=docker/dockerfile:1
#
# Claude Code is installed via npm, pinned to a version verified against this
# project (see README "Choix techniques" for why npm was chosen over the
# native installer: it lets us pin an exact, reproducible version here).
FROM python:3.12-slim

ARG CLAUDE_CODE_VERSION=2.1.292
ARG NODE_MAJOR=20

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Node.js + the official Claude Code CLI, pinned -------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/* /root/.npm

# --- Application --------------------------------------------------------
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app"]
