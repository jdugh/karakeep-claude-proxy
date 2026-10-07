# karakeep-claude-proxy

A small, self-contained OpenAI-compatible HTTP proxy that lets [Karakeep](https://karakeep.app)
use a **Claude subscription** (via `claude setup-token` + Claude Code's
non-interactive mode) for tagging and summaries, while **embeddings stay on
Ollama** with `qwen3-embedding:0.6b`.

```text
                         ┌─────────────────────┐
                         │      Karakeep       │
                         └──────────┬──────────┘
                                    │
                  tagging / résumés │ OpenAI-compatible
                                    ▼
                         ┌─────────────────────┐
                         │  claude-proxy       │
                         │  FastAPI / Python   │
                         └──────────┬──────────┘
                                    │
                         claude -p  │  (no MCP, stdin only; no tools by
                                    │   default, optionally web search/fetch)
                                    ▼
                         ┌─────────────────────┐
                         │    Claude Code      │
                         │ setup-token OAuth   │
                         └─────────────────────┘

Karakeep
   │
   └── embeddings ───────────────► Your EMBEDDING_TEXT_MODEL config
```

It does **not** call the Anthropic API with an API key, does **not** fork or
modify Karakeep or Ollama, and is **not** exposed to the Internet.

---

## 1. How it works

- `POST /v1/chat/completions` converts an OpenAI chat payload into a single
  prompt and runs `claude -p --output-format json ...` as a subprocess, with
  the prompt sent over **stdin** (never as a shell argument).
- Claude Code is invoked with **no tools, no MCP, no project config, no
  interactive permission prompts** by default: `--tools "" --disallowedTools
  "mcp__*" --permission-prompts none --setting-sources ""`. It cannot run
  shell commands, read/write files, use git, or reach MCP servers. Setting
  `CLAUDE_ENABLE_WEB_TOOLS=true` additionally exposes and pre-authorizes
  *only* `WebSearch` and `WebFetch` (`--tools "WebSearch,WebFetch"
  --allowedTools "WebSearch,WebFetch"`) — MCP, every other built-in tool, and
  `--dangerously-skip-permissions` stay off regardless; see §5.
- `response_format.type=json_schema` / `json_object` is translated to Claude
  Code's `--json-schema <schema>` flag; the resulting `structured_output`
  object is serialized back into `choices[0].message.content` as a JSON
  string, exactly like OpenAI's Structured Outputs.
- Every invocation has a timeout, a hard cap on stdout/stderr size, and the
  subprocess is always killed and reaped (no zombies), even on timeout,
  cancellation, or exceptions.
- Concurrency is bounded by `MAX_CONCURRENT_REQUESTS` (default `1`) via an
  `asyncio.Semaphore`, so a Karakeep re-index job can't fork off a dozen
  simultaneous `claude` processes.
- The proxy is **stateless**: no database, no cache, no persisted prompts or
  content. Nothing from a request is logged by default except metadata
  (request id, endpoint, resolved model, duration, status code).

## 2. What was verified against the installed CLI (not assumed)

This was built and tested against **Claude Code 2.1.292**. Before writing any
code, the actual installed CLI was inspected and exercised directly:

- `--tools ""`, `--disallowedTools "mcp__*"`, `--permission-prompts none`,
  `--setting-sources ""`, `--disable-slash-commands`, `--no-session-persistence`
  all exist and combine cleanly with `-p --output-format json --model
  <alias> --json-schema <schema>`.
- **`--max-turns` no longer exists in this CLI version** (PROMPT.md assumed
  it). It isn't needed anyway: with all tools disabled there is no agentic
  loop to bound — Claude Code answers in a single turn.
- `--bare` is incompatible with `CLAUDE_CODE_OAUTH_TOKEN` (its own `--help`
  text says Anthropic auth under `--bare` is "strictly `ANTHROPIC_API_KEY` or
  `apiKeyHelper`... OAuth and keychain are never read"). **`--bare` is never
  used**, confirming PROMPT.md section 3.2.
- The JSON envelope on success looks like:
  ```json
  {"is_error": false, "result": "OK", "structured_output": {...}, "session_id": "...", ...}
  ```
  and on failure:
  ```json
  {"is_error": true, "result": "<human message>", "api_error_status": 429, ...}
  ```
  `app/claude_runner.py` parses exactly this shape.
- A real `docker build` was run; `claude --version` is checked **during the
  build** (the build fails if the CLI is broken). A real container was run
  with `--read-only` + `tmpfs` for `/tmp`, `/home/app`, `/app/runtime`, and
  `claude --version` plus the full HTTP surface (`/healthz`, `/readyz`,
  `/v1/models`, `/v1/chat/completions`) were exercised against it — read-only
  root filesystem works fine.
- Karakeep's current environment variable names
  (`docs.karakeep.app/configuration/environment-variables/`) were checked
  directly: all names used below are current. One nuance worth restating
  from that check: `EMBEDDING_CONTEXT_LENGTH` is a **character** count passed
  to the embedding model, while `INFERENCE_CONTEXT_LENGTH` is a **token**
  count passed to the text-inference model — don't confuse the two.
- Re-checked in a later corrective pass: there is still no `--max-tokens` /
  `--temperature` / `--stop` flag of any kind in `--help`, and the
  undocumented `--settings '{"maxOutputTokens": N}'` path was tried against
  a long-form prompt and had **zero effect** (the model ignored it and
  generated ~7.7k output tokens anyway). See §5 "OpenAI parameters accepted
  but currently ignored".
- `--tools "WebSearch,WebFetch"` combined with `--allowedTools
  "WebSearch,WebFetch"` was verified both for CLI acceptance and for real
  behavior: run with `--output-format stream-json --verbose`, the trace
  showed genuine `tool_use` blocks (`WebFetch` against the npm registry,
  `WebSearch` against a real search backend) with real `tool_result` content
  feeding the final answer — not just a URL mentioned in text. The system
  init line confirmed `"tools":["WebFetch","WebSearch"]` and
  `"mcp_servers":[]`: nothing else was ever on offer. The same flags were
  then confirmed compatible with `--output-format json` (what `ClaudeRunner`
  actually uses) and with `--json-schema` in the same call — the final
  envelope shape (`result`/`structured_output`/`is_error`) is unchanged
  either way, so no parsing logic needed to change, only argv construction.

## 3. Repository layout

```text
app/
  main.py            FastAPI routes, auth wiring, concurrency, error mapping
  claude_runner.py   Locked-down `claude -p` subprocess invocation
  openai_adapter.py  OpenAI <-> Claude message/schema conversion
  models.py          Pydantic request/response models (tolerant of extra fields)
  auth.py            Bearer auth dependency
  config.py          Environment-driven settings
  errors.py          OpenAI-shaped error responses
  logging_config.py  Secret-free, request-id-tagged logging
tests/               pytest suite (unit, mocked -- never calls real Claude)
scripts/test_openai_client.py   Smoke test using the official `openai` SDK
Dockerfile
docker-compose.yml
pyproject.toml       Dependency ranges + project metadata
uv.lock              Exact, hash-pinned dependency versions (see below)
.env.example
```

### Reproducible builds (`uv.lock`)

`pyproject.toml` declares version *ranges* (intent); `uv.lock` pins the
*exact* resolved versions, with hashes, that `docker compose build` actually
installs -- so a build today and a build in six months install byte-identical
dependencies until the lock file is regenerated on purpose.

The Dockerfile installs from the lock file with
[`uv`](https://docs.astral.sh/uv/) (`uv sync --locked`, which fails the build
if `pyproject.toml` and `uv.lock` have drifted apart), copying the static,
pinned `uv` binary from `ghcr.io/astral-sh/uv` -- no extra runtime dependency
ends up in the final image.

To intentionally update dependency versions:

```bash
# install uv locally: https://docs.astral.sh/uv/getting-started/installation/
uv lock -P fastapi          # bump one package, or
uv lock --upgrade           # bump everything to the latest allowed by pyproject.toml
git diff uv.lock            # review before committing
docker compose build        # verify the new lock still builds
```

`uv.lock` is committed to the repo; `pyproject.toml`'s own metadata and
ranges are untouched by this -- it stays the human-edited source of intent,
`uv.lock` is the machine-generated source of truth for builds.

## 4. Deployment

```bash
git clone <this-repo> karakeep-claude-proxy
cd karakeep-claude-proxy

cp .env.example .env
nano .env            # fill in AI_NETWORK_NAME, PROXY_API_KEY, CLAUDE_CODE_OAUTH_TOKEN
chmod 600 .env

docker compose build
docker compose up -d

docker compose ps
docker compose logs -f claude-proxy
```

### 4.1 Getting a `CLAUDE_CODE_OAUTH_TOKEN`

On any machine with Claude Code installed and logged into your subscription:

```bash
claude setup-token
```

Copy the resulting token into the server's `.env`:

```env
CLAUDE_CODE_OAUTH_TOKEN=...
```

This token is sensitive: never commit it, it is tied to your subscription,
it can expire or be revoked, and an auth failure will show up clearly in the
proxy's logs (`claude_error` / `Not logged in · Please run /login`) without
ever printing the token itself. There is deliberately no automated way to
generate or refresh it — that step stays manual.

### 4.2 Smoke-test the proxy directly

```bash
curl -sS http://claude-proxy:8080/healthz
curl -sS http://claude-proxy:8080/readyz

curl -sS \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  http://claude-proxy:8080/v1/chat/completions \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Réponds uniquement par OK"}]
  }'

curl -sS \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  http://claude-proxy:8080/v1/chat/completions \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Tag this bookmark about docker and homelab"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "tags",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["tags"],
          "additionalProperties": false
        }
      }
    }
  }'
```

Or with the official OpenAI SDK (`pip install openai`):

```bash
PROXY_API_KEY=$PROXY_API_KEY PROXY_BASE_URL=http://claude-proxy:8080/v1 \
  python scripts/test_openai_client.py
```

### 4.3 Testing DNS/reachability from the Karakeep container

Don't assume `curl` is installed inside Karakeep's own image. From the host:

```bash
docker exec <karakeep-container> curl -fsS http://claude-proxy:8080/healthz \
  || docker run --rm --network <AI_NETWORK_NAME> curlimages/curl:latest \
       http://claude-proxy:8080/healthz
```

### 4.4 Wiring it into Karakeep

Add to Karakeep's `.env` / compose env, then restart the Karakeep stack:

```env
# =================================================
# Claude pour tagging + summaries
# =================================================
OPENAI_BASE_URL=http://claude-proxy:8080/v1
OPENAI_API_KEY=<MEME_VALEUR_QUE_PROXY_API_KEY>

INFERENCE_TEXT_MODEL=claude-sonnet
INFERENCE_OUTPUT_SCHEMA=structured
OPENAI_TIMEOUT_SEC=300

# LLM OCR désactivé tant que la vision n'est pas supportée par ce proxy
OCR_USE_LLM=false

# =================================================
# Ollama pour embeddings -- NE PAS pointer vers claude-proxy
# =================================================
EMBEDDING_OPENAI_BASE_URL=http://ollama:11434/v1
EMBEDDING_OPENAI_API_KEY=ollama

EMBEDDING_TEXT_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSIONS=1024
EMBEDDING_ENABLE_AUTO_INDEXING=true
EMBEDDING_NUM_WORKERS=1

SEMANTIC_SEARCH_ENABLED=true
```

```bash
docker compose up -d   # in the Karakeep stack
```

`EMBEDDING_OPENAI_BASE_URL` is **mandatory** here: without it, Karakeep would
try to send embedding requests to `OPENAI_BASE_URL` (this proxy), which does
not implement `/v1/embeddings` on purpose (see §7).

### 4.5 End-to-end validation checklist

1. `docker compose ps` shows `claude-proxy` healthy.
2. `GET /healthz` → `200 {"status":"ok"}`.
3. `GET /v1/models` (with auth) lists the configured aliases.
4. The direct `curl` calls in §4.2 succeed.
5. Karakeep config updated as in §4.4 and the Karakeep stack restarted.
6. Add a test bookmark in Karakeep.
7. Confirm automatic tagging runs.
8. Confirm the summary is generated.
9. Check `claude-proxy` logs: `requested_model=... resolved_model=... structured=True duration_ms=...`.
10. Check Karakeep/Ollama logs to confirm **embeddings still go to Ollama**, not to `claude-proxy`.

## 5. Configuration reference (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `AI_NETWORK_NAME` | — | Name of the existing external Docker network shared with Karakeep/Ollama. |
| `PROXY_API_KEY` | — | Bearer secret Karakeep uses to call this proxy. **Different secret from `CLAUDE_CODE_OAUTH_TOKEN`**, never reused as it. |
| `CLAUDE_CODE_OAUTH_TOKEN` | — | From `claude setup-token`. Proxy → Claude Code auth. |
| `DEFAULT_MODEL` | `claude-sonnet` | Must be a key in `MODEL_MAP`. |
| `MODEL_MAP` | `claude-sonnet=sonnet,claude-haiku=haiku,claude-opus=opus` | Allowlist: public alias → real Claude Code model. No client-supplied model string reaches Claude Code directly. |
| `CLAUDE_TIMEOUT_SEC` | `180` | Hard timeout per `claude -p` invocation; process is killed on expiry. |
| `MAX_CONCURRENT_REQUESTS` | `1` | `asyncio.Semaphore` size. Raise to `2` cautiously once stable. |
| `MAX_REQUEST_BODY_BYTES` | `2097152` (2 MiB) | Oversized bodies get a clean `413`. |
| `MAX_RETRIES` | `0` | Reserved; no retry logic is implemented in V1 (quota errors must never be retried silently). |
| `LOG_LEVEL` | `INFO` | Never logs prompt/message content regardless of level. |
| `ENABLE_TEST_CLAUDE_ENDPOINT` | `false` | Enables `POST /internal/test-claude` (auth-protected, costs quota). |
| `CLAUDE_ENABLE_WEB_TOOLS` | `false` | See "Web tools" below. Recreate the container after changing. |
| `CLAUDE_WEB_PROMPT_FILE` | `/app/prompts/web-tools.txt` | Only read when `CLAUDE_ENABLE_WEB_TOOLS=true`. See "Web tools" below. |

Your model subscription may not grant access to every alias in
`MODEL_MAP` — only the aliases you list are *accepted*, not guaranteed to
succeed; an inaccessible model surfaces as a normal `502`/`429` from
Claude Code.

### Timeouts, end to end

```text
Karakeep  OPENAI_TIMEOUT_SEC=300   (Karakeep waiting for the proxy's HTTP response)
   ⇡ should be greater than ⇡
Proxy     CLAUDE_TIMEOUT_SEC=180   (proxy waiting for the `claude` subprocess)
```

Keep Karakeep's timeout comfortably above the proxy's, so the proxy always
has a chance to return a clean `504` before Karakeep gives up first.

### Web tools (`CLAUDE_ENABLE_WEB_TOOLS`, `CLAUDE_WEB_PROMPT_FILE`)

Off by default: Claude gets no tools at all, same as before. Set
`CLAUDE_ENABLE_WEB_TOOLS=true` and recreate the container to additionally
expose and pre-authorize **only** `WebSearch` and `WebFetch` — no other
built-in tool (Bash, Read, Write, Edit, ...), no MCP, and no interactive
permission prompt is ever introduced by this flag; `--dangerously-skip-permissions`
is never used either way. See `app/claude_runner.py::ClaudeRunner.run` for
the exact argv this toggles.

When web tools are on, the proxy also appends a search-guidance instruction
via `--append-system-prompt` (argv, not stdin) telling Claude *when* it
should actually use those tools — enabling the tools alone doesn't tell
Claude to prefer searching. That instruction's text lives in a plain UTF-8
file, not hardcoded:

- **`CLAUDE_WEB_PROMPT_FILE`** (default `/app/prompts/web-tools.txt`) points
  to it. The repo ships [`prompts/web-tools.txt`](prompts/web-tools.txt) with
  a ready-to-use instruction, baked into the Docker image at that exact
  default path — web mode works out of the box with **no extra volume
  mount**.
- **It is re-read on every request**, not cached, so editing the mounted
  file takes effect on the very next call — no proxy restart needed.
  Changing the `CLAUDE_WEB_PROMPT_FILE` *path* itself is an env var change
  and does need `docker compose up -d` to recreate the container, same as
  any other setting here.
- **To customize it** on your VPS: copy the shipped file out, edit it, then
  mount the *directory* (not the single file) read-only in
  `docker-compose.yml`:
  ```yaml
  volumes:
    - ./prompts:/app/prompts:ro
  ```
  Mounting the directory rather than one file matters: editors and sync
  tools that save via atomic rename-and-replace swap the file's inode,
  which breaks a single-file bind mount but not a directory mount. This is
  commented out in `docker-compose.yml` by default — it is optional, not
  required to use web tools.
- **Error handling is intentionally strict, not silent.** If
  `CLAUDE_ENABLE_WEB_TOOLS=true` and the file is missing, unreadable, or
  empty: the proxy refuses to start (clear error in
  `docker compose logs`), and if the file disappears *after* a successful
  start, the affected request gets a clean `503` instead of silently
  running without the intended guidance.

- **What it's for:** letting Claude look something up (confirm a fact, fetch
  a linked page) while generating a tag or summary, when the instruction in
  the request calls for it.
- **It does not guarantee a search happened.** The tools being available
  and the file's instruction nudging Claude toward using them is not a
  technical guarantee — Claude still decides, and nothing here verifies or
  retries based on whether a search actually occurred. Karakeep doesn't
  need any change to benefit or to keep working exactly as before.
- **Cost/latency:** a web lookup adds real latency (often several seconds)
  and consumes your Claude subscription's usage/quota on top of the
  inference call itself.
- **Data exposure:** `WebSearch` sends search terms derived from the request
  content (which may include bookmark text) to a search provider, and
  `WebFetch` retrieves whatever page it's pointed at. Don't enable this if
  that's not acceptable for your content.

### `INFERENCE_CONTEXT_LENGTH` / `INFERENCE_MAX_OUTPUT_TOKENS`

Don't default to Claude's maximum context window — three tags don't need
tens of thousands of tokens, though a long summary may need more than a
short tag list. Start modest (e.g. a few thousand tokens for
`INFERENCE_CONTEXT_LENGTH`, a few hundred for `INFERENCE_MAX_OUTPUT_TOKENS`)
and raise only if truncation is observed.

**Important:** these are Karakeep-side settings only. Setting
`INFERENCE_MAX_OUTPUT_TOKENS=1024` on the Karakeep side does **not** make
this proxy enforce a 1024-token cap on Claude Code's output — see the next
section for why.

### OpenAI parameters accepted but currently ignored

The request schema accepts the following OpenAI fields (so the SDK never
errors on them), but **none of them are currently mapped to a Claude Code
CLI equivalent** — they are parsed, validated for type, and then dropped:

```text
temperature
top_p
max_tokens
max_completion_tokens
frequency_penalty
presence_penalty
seed
stop
```

This was re-checked directly against the installed **Claude Code 2.1.292**
CLI for this corrective pass, both from `--help` and empirically:

- `claude --help` / `claude -p --help` expose no `--max-tokens`,
  `--max-output-tokens`, `--temperature`, or `--stop` flag of any kind.
- `--settings '{"maxOutputTokens": 20}'` was tried against a prompt designed
  to produce a long response: Claude Code ignored it and generated **7773
  output tokens** regardless (its own reported `maxOutputTokens` for the
  model stayed `128000`). So even the undocumented `--settings` JSON path is
  not a reliable lever for this — it was not adopted.

Conclusion: there is currently no officially documented, reliable way to
cap `-p` output length from outside the model's own judgment. Rather than
fabricate a pseudo-limit (e.g. post-hoc truncation, or stuffing "keep your
answer under N tokens" into the prompt), the proxy accepts these parameters
and ignores them honestly. If a future Claude Code release adds a real flag
for this, map it here and add a test — don't invent a workaround in the
meantime.

In practice this mostly matters for `max_tokens`/`max_completion_tokens`:
Claude will size its answer to the instruction (e.g. "return 3 tags") rather
than to an arbitrary cap, which is usually fine for Karakeep's tagging/
summary use case; if you see runaway-length responses, control it via your
prompt/instruction wording rather than these fields.

## 6. API surface

| Method | Path | Auth |
|---|---|---|
| `GET` | `/` | none |
| `GET` | `/healthz` | none |
| `GET` | `/readyz` | none |
| `GET` | `/v1/models` | Bearer |
| `POST` | `/v1/chat/completions` | Bearer |
| `POST` | `/internal/test-claude` | Bearer, disabled unless `ENABLE_TEST_CLAUDE_ENDPOINT=true` |

`/readyz` checks the `claude` binary is on `PATH`, the OAuth token and proxy
key are configured, and the model map is valid — **without** spending a
Claude request. It never returns the token itself.

### Error shape

```json
{"error": {"message": "...", "type": "upstream_error", "param": null, "code": "claude_timeout"}}
```

| HTTP | When |
|---|---|
| 400 | Invalid JSON/body, unsupported `response_format`, multimodal content, `stream=true` |
| 401 | Missing/incorrect `PROXY_API_KEY` |
| 404 | Unknown model alias |
| 413 | Body exceeds `MAX_REQUEST_BODY_BYTES` |
| 429 | Claude Code reports a rate limit / quota error |
| 502 | Claude Code failed or returned invalid output |
| 503 | `claude` binary missing, or not configured |
| 504 | `claude -p` exceeded `CLAUDE_TIMEOUT_SEC` |
| 500 | Unexpected bug — never a Python traceback |

Quota/rate-limit handling never retries automatically, never rotates
accounts, and never regenerates a token — it surfaces a clean `429`.

## 7. Embeddings stay on Ollama

`/v1/embeddings` is **intentionally not implemented**. Point Karakeep's
`EMBEDDING_OPENAI_BASE_URL` at Ollama directly (§4.4). This proxy only
answers `/v1/chat/completions` and `/v1/models`.

## 8. Known limitations

- Depends on Claude Code being installed, correctly authenticated, and not
  rate-limited; a future Claude Code release could change its JSON envelope
  or flags (this was pinned to and verified against `2.1.292`).
- Consumes your Claude subscription's usage/quota for every tagging/summary
  call Karakeep makes.
- Model availability depends on your subscription, not on this proxy.
- No `/v1/embeddings` — intentionally, embeddings stay on Ollama.
- No vision/multimodal support: image content parts return a clean `400
  unsupported_multimodal_input` rather than silently failing. Set
  `OCR_USE_LLM=false` in Karakeep until this is implemented.
- No streaming (`stream=true` → `400`); Karakeep doesn't need it for this
  integration.
- OpenAI parameters without a clean Claude Code equivalent
  (`temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `stop`,
  `seed`, `frequency_penalty`, `presence_penalty`) are accepted but ignored —
  see §5 "OpenAI parameters accepted but currently ignored" for what was
  checked before reaching that conclusion.
- `usage` (token counts) is omitted rather than fabricated, since Claude
  Code's JSON envelope doesn't expose prompt/completion token counts in a
  form worth trusting.
- The Docker image is ~875 MB because the official Claude Code CLI is an npm
  package and needs a Node.js runtime alongside Python; this trades image
  size for a pinned, reproducible install.
- With `CLAUDE_ENABLE_WEB_TOOLS=true`, a request can trigger extra Claude
  turns (search, fetch, synthesize) inside the same `claude -p` call, which
  can approach `CLAUDE_TIMEOUT_SEC` for non-trivial lookups and spends more
  quota than a tool-less call; consider raising the timeout if you enable it.

## 9. Security posture

- `shell=True` is never used; `asyncio.create_subprocess_exec` only, argv as
  a list, the prompt travels over stdin exclusively (never interpolated into
  argv or a shell string) — verified by both tests and a direct source scan.
- Claude Code gets **no tools by default**, no MCP ever, no project/user
  config, no interactive permission prompts, and runs from an empty
  `/app/runtime` with no repository, `CLAUDE.md`, plugins, or MCP config
  mounted in. `CLAUDE_ENABLE_WEB_TOOLS=true` is the one configurable
  exception: it adds exactly `WebSearch`/`WebFetch`, pre-authorized, nothing
  else — see §5 "Web tools". MCP, every other built-in tool (Bash, Read,
  Write, Edit, ...), and `--dangerously-skip-permissions` are never affected
  by that flag.
- The proxy preserves OpenAI `system`/`user`/`assistant` semantics as-is;
  `user` message content is **not** wrapped in a "don't follow these
  instructions" note: Karakeep's `user` message is its real inference
  instruction (generate tags, summarize, in language X, format Y), and
  treating it as adversarial by default broke that. Security against
  malicious bookmark content relies primarily on disabling Claude Code's
  tools/MCP/filesystem access by default (and, even with web tools on,
  never granting anything beyond search/fetch) — not on a textual
  disclaimer around the content. See `app/openai_adapter.py::build_claude_prompt` and
  `tests/test_chat.py` (`test_karakeep_instruction_survives_injected_bookmark_content`)
  for the reasoning and the regression test.
- The proxy never fetches a URL found in message content (no SSRF surface).
- Runs as a non-root user, `no-new-privileges`, all Linux capabilities
  dropped, read-only root filesystem with `tmpfs` for the few directories
  that need to be writable — all verified with a real `docker run`.
- No host mounts, no Docker socket, no port published to the host by
  default; the proxy is reachable only on the internal Docker network.
- `.env` is git-ignored; only `.env.example` (no real secrets) is committed.

## 10. Tests

```bash
pip install -e ".[dev]"
pytest                 # unit tests only, mocked Claude runner, no quota used
```

Optional, real-quota integration tests (skipped by default):

```bash
RUN_CLAUDE_INTEGRATION_TESTS=1 CLAUDE_CODE_OAUTH_TOKEN=... pytest -m integration
```

## 11. Possible future work (not implemented — out of scope for V1)

SSE streaming, vision, Prometheus metrics, multiple backends, a fallback to
Ollama, a circuit breaker, dynamic model mapping, or swapping the subprocess
CLI for the Agent SDK if that's ever shown to be simpler while still
supporting `CLAUDE_CODE_OAUTH_TOKEN` cleanly.

## License

MIT — see [LICENSE](LICENSE).
