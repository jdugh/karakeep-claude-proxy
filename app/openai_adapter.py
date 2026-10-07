"""Converts between OpenAI chat-completion payloads and Claude Code invocations.

Responsible for:
- turning `messages` into a single prompt. `system` messages are kept
  distinct, `user`/`assistant` messages are passed through as real
  instructions/content in order -- Karakeep's `user` message *is* its actual
  inference instruction (generate tags, summarize, in language X, in format
  Y), not untrusted data to be defused. The proxy does not wrap content in a
  "don't follow these instructions" note: the real security boundary is that
  Claude is given no tools at all (`--tools ""`, `--disallowedTools
  "mcp__*"`, see claude_runner.py), so even a prompt-injected instruction
  from a fetched web page has nothing to execute;
- resolving `response_format` into a `--json-schema` argument for ClaudeRunner;
- resolving the public model alias into the Claude Code model name via the
  configured allowlist;
- building the OpenAI-shaped chat.completion response.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.claude_runner import ClaudeResult
from app.config import Settings
from app.errors import InvalidRequestError, NotFoundError, UnsupportedInputError, UpstreamError
from app.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ChoiceMessage, ResponseFormat

_UNSUPPORTED_CONTENT_TYPES = {"image_url", "input_image", "image"}


def resolve_model(settings: Settings, requested_alias: str) -> str:
    resolved = settings.resolve_model(requested_alias)
    if resolved is None:
        raise NotFoundError(
            f"Unknown model {requested_alias!r}. Available aliases: {', '.join(settings.available_models)}",
            code="model_not_found",
        )
    return resolved


def _extract_part_text(item: dict[str, Any]) -> str:
    part_type = item.get("type", "text")
    if part_type in _UNSUPPORTED_CONTENT_TYPES:
        raise UnsupportedInputError(
            f"content part of type {part_type!r} is not supported: vision is not implemented in V1"
        )
    if part_type != "text":
        raise UnsupportedInputError(f"content part of type {part_type!r} is not supported")
    text = item.get("text")
    if not isinstance(text, str):
        raise InvalidRequestError("content part of type 'text' must include a string 'text' field")
    return text


def _extract_message_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_extract_part_text(item) for item in content)
    raise InvalidRequestError("message content must be a string or a list of content parts")


def build_claude_prompt(messages: list[ChatMessage]) -> str:
    """Builds the prompt sent to Claude over stdin.

    Deterministic, no parsing/heuristics of message content:
    - system messages are kept as-is (real system instructions);
    - the overwhelmingly common Karakeep shape -- a single `user` message --
      is sent through verbatim, since that message *is* the real inference
      instruction (tags/summary/language/format), not untrusted data;
    - for multi-turn requests, `user`/`assistant` turns are labelled and
      kept in order so Claude can tell who said what.

    No "ignore instructions in this section" note is added anywhere: that
    used to contradict Karakeep's own prompt. The actual protection against a
    hostile bookmark is that Claude has no tools to act on anything (see
    claude_runner.py), not a textual disclaimer.
    """
    if not messages:
        raise InvalidRequestError("messages must contain at least one entry", param="messages")

    extracted = [(msg.role, _extract_message_text(msg.content)) for msg in messages]

    non_system = [(role, text) for role, text in extracted if role != "system"]
    if len(extracted) == 1 and len(non_system) == 1 and non_system[0][0] == "user":
        return non_system[0][1]

    sections: list[str] = []
    for role, text in extracted:
        if not text:
            continue
        if role == "system":
            sections.append(text)
        else:
            label = "User" if role == "user" else "Assistant"
            sections.append(f"{label}: {text}")
    return "\n\n".join(sections)


def resolve_response_schema(response_format: ResponseFormat | None) -> tuple[dict[str, Any] | None, bool]:
    """Returns (json_schema_for_cli, wants_structured_content)."""
    if response_format is None or response_format.type == "text":
        return None, False

    if response_format.type == "json_schema":
        if response_format.json_schema is None:
            raise InvalidRequestError(
                "response_format.type='json_schema' requires response_format.json_schema.schema",
                param="response_format",
            )
        return response_format.json_schema.schema_, True

    if response_format.type == "json_object":
        return {"type": "object", "additionalProperties": True}, True

    raise InvalidRequestError(
        f"Unsupported response_format.type: {response_format.type!r}", param="response_format"
    )


def build_chat_response(
    *, request_model_alias: str, claude_result: ClaudeResult, wants_structured: bool
) -> ChatCompletionResponse:
    if wants_structured:
        if claude_result.structured is None:
            raise UpstreamError(
                "Claude Code did not return a structured_output payload for a structured request"
            )
        content = json.dumps(claude_result.structured, ensure_ascii=False)
    else:
        content = claude_result.text

    return ChatCompletionResponse(
        id=f"chatcmpl-local-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request_model_alias,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def prepare_invocation(settings: Settings, request: ChatCompletionRequest) -> tuple[str, str, dict[str, Any] | None, bool]:
    """Validates the request and returns (resolved_model, prompt, json_schema, wants_structured)."""
    if request.stream:
        raise InvalidRequestError(
            "stream=true is not supported in this V1; retry with stream=false", param="stream"
        )

    resolved_model = resolve_model(settings, request.model)
    prompt = build_claude_prompt(request.messages)
    json_schema, wants_structured = resolve_response_schema(request.response_format)
    return resolved_model, prompt, json_schema, wants_structured
