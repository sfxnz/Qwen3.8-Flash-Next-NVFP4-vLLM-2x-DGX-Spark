"""Shared stdlib helpers for the quality gates: HTTP client, SSE, manifest, stats.

Adapted from the GLM-5.3 harness (quality/common.py). Qwen changes: no
[gMASK]<sop> prefix; every request goes through the served chat template
(/v1/chat/completions), thinking off unless a script says otherwise.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
# Datasets and large runs live outside the repo (AGENTS.md: large artifacts -> data/).
EVALS_DIR = Path(os.environ.get("QWEN38_EVALS_DIR", Path.home() / "projects/data/qwen38-evals"))
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ON = {"chat_template_kwargs": {"enable_thinking": True}}
MAX_WORKERS = 8  # MAX_NUM_SEQS=8; never exceed it from a gate script


class HTTPFailure(RuntimeError):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:500]}")
        self.code = code
        self.body = body


def base_url(url: str) -> str:
    """Accept http://host:port, .../v1 or .../v1/chat/completions."""
    url = url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


class Client:
    def __init__(self, url: str = DEFAULT_URL, model: str | None = None, timeout: float = 3600):
        self.url = base_url(url)
        self.timeout = timeout
        self._model = model

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.get("/v1/models")["data"][0]["id"]
        return self._model

    def get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            body = r.read().decode()
        return body if path == "/metrics" else json.loads(body)

    def _request(self, path: str, body: dict):
        data = json.dumps({"model": self.model, **body}).encode()
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise HTTPFailure(exc.code, exc.read().decode("utf-8", "replace")) from exc

    def post(self, path: str, body: dict) -> dict:
        with self._request(path, body) as r:
            return json.loads(r.read().decode())

    def tokenize(self, text: str) -> list[int]:
        return [int(t) for t in self.post("/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]]

    def tokenize_count(self, text: str) -> int:
        return len(self.tokenize(text))

    def chat(self, messages, *, stream: bool = False, **body) -> dict:
        """One chat completion. Returns a flat dict (same keys for both modes)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"messages": messages, **body}
        t0 = time.time()
        if stream:
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
            with self._request("/v1/chat/completions", body) as r:
                out = parse_sse(r)
        else:
            out = flatten_chat(self.post("/v1/chat/completions", body))
        out["s"] = round(time.time() - t0, 3)
        return out


def _logprob_rows(entries) -> list[dict]:
    """Chat logprobs.content entries -> [{"id", "lp", "top": [(id, lp), ...]}].

    Needs return_tokens_as_token_ids=true so tokens arrive as 'token_id:N'."""
    rows = []
    for e in entries or []:
        rows.append({"id": token_id(e["token"]), "lp": float(e["logprob"]),
                     "top": [(token_id(t["token"]), float(t["logprob"])) for t in e.get("top_logprobs") or []]})
    return rows


def token_id(tok: str) -> int:
    if not tok.startswith("token_id:"):
        raise ValueError(f"expected 'token_id:N' (send return_tokens_as_token_ids=true), got {tok!r}")
    return int(tok[9:])


def flatten_chat(resp: dict) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments") or ""})
    lp = choice.get("logprobs") or {}
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "tool_calls": calls, "finish_reason": choice.get("finish_reason"),
            "token_ids": choice.get("token_ids"), "usage": resp.get("usage") or {},
            "logprobs": _logprob_rows(lp.get("content")) if lp.get("content") else None}


def parse_sse(lines) -> dict:
    """Accumulate an OpenAI chat SSE stream into the flatten_chat shape."""
    content, reasoning, calls = [], [], {}
    finish, usage, token_ids, lps = None, {}, [], []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            content.append(d.get("content") or "")
            reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
            token_ids += ch.get("token_ids") or []
            lps += _logprob_rows((ch.get("logprobs") or {}).get("content"))
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish,
            "token_ids": token_ids or None, "usage": usage, "logprobs": lps or None}


def pmap(fn, items, workers: int = 1) -> list:
    workers = max(1, min(MAX_WORKERS, workers))
    if workers == 1:
        return [fn(x) for x in items]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ files

def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def git_rev() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return ""
    return {"rev": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", str(HERE)))}


def prepare_out(out: str | Path) -> Path:
    p = Path(out).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_manifest(out_dir: Path, client: Client | None, tool: str, args: dict, files=(), extra=None) -> dict:
    """manifest.json: what ran, against which server, over which bytes."""
    models = None
    if client is not None:
        try:
            models = client.get("/v1/models")
        except (OSError, ValueError) as exc:
            models = {"error": str(exc)}
    shas = {str(Path(f).relative_to(ROOT) if str(f).startswith(str(ROOT)) else f): sha256_file(f)
            for f in files if Path(f).exists()}
    shas.update({f"quality/{p.relative_to(HERE)}": sha256_file(p) for p in sorted(HERE.rglob("*.py"))
                 if "__pycache__" not in p.parts})
    man = {"tool": tool, "utc": utc_stamp(), "argv": sys.argv, "args": args, "git": git_rev(),
           "host": platform.node(), "server": client.url if client else None, "models": models,
           "sha256": shas, **(extra or {})}
    write_json(out_dir / "manifest.json", json.loads(json.dumps(man, default=str)))
    return man


def running_requests(client: Client) -> float | None:
    """vllm:num_requests_running now (other traffic co-prefilled with a capture skews it; F01)."""
    try:
        for line in client.get("/metrics").splitlines():
            if line.startswith("vllm:num_requests_running"):
                return float(line.rsplit(" ", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def wait_health(client: Client, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(client.url + "/health", timeout=timeout) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


# ------------------------------------------------------------------ stats

def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired booleans (a = reference, b = candidate)."""
    if len(a) != len(b):
        raise ValueError("paired lists differ in length")
    n01 = sum(1 for x, y in zip(a, b) if x and not y)  # ref right, cand wrong
    n10 = sum(1 for x, y in zip(a, b) if y and not x)  # cand right, ref wrong
    n = n01 + n10
    k = min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    acc_a = sum(a) / len(a) if a else 0.0
    acc_b = sum(b) / len(b) if b else 0.0
    return {"n": len(a), "ref_acc": round(acc_a, 6), "cand_acc": round(acc_b, 6),
            "drop_pp": round(100 * (acc_a - acc_b), 3), "lost": n01, "gained": n10, "p": round(p, 6)}


def threshold_lower(stated: float, floor: float | None) -> float:
    """C25, lower-is-better metrics (KL, flip rate, malformed rate): max(stated, 2 x floor)."""
    return stated if floor is None else max(stated, 2 * floor)


def threshold_agree(stated: float, floor: float | None) -> float:
    """C25, agreement metrics (top-1, first-32 match, ...): min(stated, 1 - 2 x (1 - floor))."""
    return stated if floor is None else min(stated, 1 - 2 * (1 - floor))
