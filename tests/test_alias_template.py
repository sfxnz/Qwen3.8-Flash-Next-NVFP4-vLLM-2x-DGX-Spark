#!/usr/bin/env python3
"""Offline render-diff: chat_template_alias.jinja vs the checkpoint template (F21, C29).

Stock is recovered from the committed alias by inverting the generator and
pinned by sha256, so no HF cache is needed. Rendering needs jinja2.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_alias_template", ROOT / "tools" / "make_alias_template.py"
)
mat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mat)

try:
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:  # pragma: no cover
    jinja2 = None

EFFORTS = (None, "none", "low", "medium", "xhigh")
ALIASES = {"high": "xhigh", "max": "xhigh", "minimal": "low"}
# Server --default-chat-template-kwargs: recipe default (thinking off) and none.
SERVER_DEFAULTS = ({"enable_thinking": False}, {})
USER_THINKING = ("absent", True, False)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
CONVERSATIONS = {
    "user": [{"role": "user", "content": "Count 1 to 5."}],
    "system_user": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
    ],
    "multi_turn": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"},
        {"role": "user", "content": [{"type": "text", "text": "And 3+3?"}]},
    ],
    "tool_round": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Wellington?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Wellington"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "14C, windy"},
    ],
}


def _env() -> "ImmutableSandboxedEnvironment":
    # Same shape as transformers' chat-template environment (which vLLM uses).
    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def resolve_kwargs(server: dict, user_thinking, effort) -> dict:
    """Mirror vLLM v0.30 ChatCompletionRequest.build_chat_params + merge_kwargs.

    protocol.py: extra = {reasoning_effort}; if effort is set and the client
    sent no enable_thinking, enable_thinking = effort != 'none'. merge_kwargs
    drops None values; server defaults sit underneath the request.
    """
    user = {} if user_thinking == "absent" else {"enable_thinking": user_thinking}
    extra = {"add_generation_prompt": True, "reasoning_effort": effort}
    if effort is not None and "enable_thinking" not in user:
        extra["enable_thinking"] = effort != "none"
    req = user | {k: v for k, v in extra.items() if v is not None}
    return dict(server) | req


def render(template, messages, tools, kwargs):
    try:
        return "ok", template.render(messages=messages, tools=tools, **kwargs)
    except jinja2.exceptions.TemplateError as e:
        return "err", str(e)


class GeneratorTest(unittest.TestCase):
    def test_alias_inverts_to_pinned_stock(self) -> None:
        alias = (ROOT / "chat_template_alias.jinja").read_text()
        self.assertEqual(mat.sha256(mat.unmake(alias)), mat.STOCK_SHA256)

    def test_alias_is_fresh_render(self) -> None:
        alias = (ROOT / "chat_template_alias.jinja").read_text()
        self.assertEqual(mat.make(mat.unmake(alias)), alias)

    @unittest.skipUnless(mat.STOCK.exists(), "checkpoint snapshot not in HF cache")
    def test_hf_cache_stock_matches(self) -> None:
        self.assertEqual(mat.sha256(mat.STOCK.read_text()), mat.STOCK_SHA256)


@unittest.skipIf(jinja2 is None, "jinja2 not installed")
class RenderDiffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        alias_text = (ROOT / "chat_template_alias.jinja").read_text()
        env = _env()
        cls.stock = env.from_string(mat.unmake(alias_text))
        cls.alias = env.from_string(alias_text)

    def _cases(self, efforts):
        for server, user_thinking, effort, (conv, msgs), tools in itertools.product(
            SERVER_DEFAULTS,
            USER_THINKING,
            efforts,
            CONVERSATIONS.items(),
            (None, TOOLS),
        ):
            kw = resolve_kwargs(server, user_thinking, effort)
            yield (server, user_thinking, effort, conv, bool(tools)), msgs, tools, kw

    def test_byte_identical_for_supported_efforts(self) -> None:
        n = 0
        for label, msgs, tools, kw in self._cases(EFFORTS):
            with self.subTest(case=label):
                self.assertEqual(
                    render(self.stock, msgs, tools, kw),
                    render(self.alias, msgs, tools, kw),
                )
                n += 1
        self.assertEqual(n, 2 * 3 * len(EFFORTS) * len(CONVERSATIONS) * 2)

    def test_thinking_off_default_prompt(self) -> None:
        # Recipe default: server enable_thinking=false, no effort -> empty think.
        kw = resolve_kwargs({"enable_thinking": False}, "absent", None)
        status, text = render(self.alias, CONVERSATIONS["user"], None, kw)
        self.assertEqual(status, "ok")
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertNotIn("Reasoning effort", text)

    def test_aliases_render_as_target(self) -> None:
        for (label, msgs, tools, kw), (alias_effort, target) in itertools.product(
            self._cases((None,)), ALIASES.items()
        ):
            server, user_thinking = label[0], label[1]
            kw_a = resolve_kwargs(server, user_thinking, alias_effort)
            kw_t = resolve_kwargs(server, user_thinking, target)
            with self.subTest(case=label, effort=alias_effort):
                got = render(self.alias, msgs, tools, kw_a)
                self.assertEqual(got[0], "ok", got[1])
                self.assertEqual(got, render(self.alias, msgs, tools, kw_t))
                stock = render(self.stock, msgs, tools, kw_a)
                if kw_a.get("enable_thinking") is False:
                    # Client forced thinking off: effort is ignored by both.
                    self.assertEqual(stock, got)
                else:
                    self.assertEqual(stock[0], "err")
                    self.assertIn("Unexpected reasoning effort", stock[1])


if __name__ == "__main__":
    unittest.main()
