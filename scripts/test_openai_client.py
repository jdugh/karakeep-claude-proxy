#!/usr/bin/env python3
"""Smoke test using the official `openai` Python SDK against a running proxy.

Usage:
    pip install openai
    PROXY_API_KEY=... PROXY_BASE_URL=http://localhost:8080/v1 python scripts/test_openai_client.py
"""
from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("PROXY_BASE_URL", "http://localhost:8080/v1")
API_KEY = os.environ.get("PROXY_API_KEY")
MODEL = os.environ.get("PROXY_TEST_MODEL", "claude-sonnet")


def main() -> int:
    if not API_KEY:
        print("Set PROXY_API_KEY to the proxy's configured key.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("== /v1/models ==")
    models = client.models.list()
    for m in models.data:
        print(f"  {m.id}")

    print("\n== plain text chat completion ==")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(" ", response.choices[0].message.content)

    print("\n== structured output (json_schema) ==")
    structured = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Return three generic tags for a Docker tutorial bookmark."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tags",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["tags"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = structured.choices[0].message.content
    print(" raw:", content)
    print(" parsed:", json.loads(content))

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
