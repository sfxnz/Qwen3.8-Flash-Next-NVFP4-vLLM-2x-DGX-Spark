#!/usr/bin/env python3
"""Live OpenAI-compat tool-call smoke against a running serve."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    args = parser.parse_args()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the weather in Wellington? "
                    "Use the get_weather tool. Do not answer from memory."
                ),
            }
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        return 1
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print(
        json.dumps(
            {
                "tool_calls": calls,
                "finish_reason": data["choices"][0].get("finish_reason"),
                "content": msg.get("content"),
            },
            indent=2,
        )
    )
    if not calls:
        print("result=fail reason=no_tool_calls", file=sys.stderr)
        return 1
    name = calls[0].get("function", {}).get("name")
    if name != "get_weather":
        print(f"result=fail reason=wrong_tool name={name!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
