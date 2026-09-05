#!/usr/bin/env python3
"""Thinking-off smoke. Fail if content starts as chain-of-thought or loops."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

THINK_STARTS = (
    "<think>",
    "</think>",
    "thinking",
    "let me think",
    "i need to think",
    "the problem is",
    "the user asked",
    "the user's request",
)


def loops(text: str) -> bool:
    if len(text) < 80:
        return False
    window = 16
    counts: dict[str, int] = {}
    for i in range(0, len(text) - window + 1, window):
        chunk = text[i : i + window]
        counts[chunk] = counts.get(chunk, 0) + 1
        if counts[chunk] >= 5:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly the word PING and nothing else.",
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.load(resp)
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    stripped = content.lstrip()
    lower = stripped.lower()
    leaked = "<think>" in content or "</think>" in content
    starts_cot = any(lower.startswith(s) for s in THINK_STARTS)
    looping = loops(content)
    print(
        json.dumps(
            {
                "content": content,
                "reasoning": reasoning,
                "content_chars": len(stripped),
                "leaked_think": leaked,
                "starts_cot": starts_cot,
                "looping": looping,
                "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
            },
            indent=2,
        )
    )
    if not stripped:
        print("result=fail reason=empty_content", file=sys.stderr)
        return 1
    if leaked or starts_cot:
        print("result=fail reason=chain_of_thought", file=sys.stderr)
        return 1
    if looping:
        print("result=fail reason=looping", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
