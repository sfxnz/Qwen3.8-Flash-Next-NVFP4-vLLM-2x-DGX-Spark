#!/usr/bin/env python3
"""Vision ruler (plan 0-7; findings G07, G10, G11, G12).

Synthetic inputs, generated with the stdlib only (PNG via zlib; a block grid):

  sq256      256 x 256
  sq1024     1024 x 1024
  photo12mp  4032 x 3024 (phone-photo size; PNG, not JPEG: client decode differs)
  neg16mp    4608 x 3456 negative test. It is served; usage.prompt_tokens shows
             whether a pixel cap resized it (image_tokens < expected_unresized).
  video30    30 frames of 1280 x 720 sent as an OpenAI video_url
             (data:video/jpeg;base64,<frame>,<frame>,... - vLLM loads each frame
             through PIL, which reads PNG). Skipped with a note if the server
             rejects video input.

Each case runs twice: solo, then with one concurrent text stream (ignore_eos)
that is already decoding when the image is sent. Reported: image TTFT, image
tokens (prompt_tokens minus a text-only baseline), and for the concurrent run
the text stream's inter-chunk gaps between the image send and the image's first
token (the decode stall G07 describes).

Every send uses a fresh colour salt, so vLLM's multimodal processor cache never
serves a repeat (a cache hit would hide the CPU preprocessing cost of G11).
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from rulers import common as C

CASES = {
    "sq256": (256, 256),
    "sq1024": (1024, 1024),
    "photo12mp": (4032, 3024),
    "neg16mp": (4608, 3456),
    "video30": (1280, 720),
}
TEXT = "Describe this image in one sentence."
VTEXT = "Describe what changes across this video in one sentence."


def expected_tokens(w: int, h: int) -> int:
    """Unresized Qwen-VL LM tokens: 16 px patches, 2x2 merge -> 32 px per token."""
    return math.ceil(w / 32) * math.ceil(h / 32)


def b64png(w: int, h: int, salt) -> str:
    return base64.b64encode(C.grid_png(w, h, salt=salt)).decode()


def mm_body(args, case: str, salt: str) -> dict:
    w, h = CASES[case]
    if case == "video30":
        frames = ",".join(b64png(w, h, f"{salt}-f{i}") for i in range(args.frames))
        part = {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{frames}"}}
        text = VTEXT
    else:
        part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64png(w, h, salt)}"}}
        text = TEXT
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}, part]}]
    return C.chat_body(args.model, msgs, args.max_tokens)


def baseline_prompt_tokens(args, text: str) -> int:
    r = C.stream_chat(args.url, C.chat_body(args.model, [{"role": "user", "content": [{"type": "text", "text": text}]}], 1))
    return int((r["usage"] or {}).get("prompt_tokens") or 0)


def run_case(args, em, case: str, concurrent: bool, base_tokens: int) -> dict:
    salt = uuid.uuid4().hex[:8]
    body = mm_body(args, case, salt)  # build before timing: client-side PNG work is not server time
    stop = threading.Event()
    t_ref = time.perf_counter()
    text_future = None
    pool = ThreadPoolExecutor(max_workers=2)
    if concurrent:
        tb = C.chat_body(args.model, f"[{salt}] Write a long story about a lighthouse.", 4096, ignore_eos=True)
        text_future = pool.submit(C.stream_chat, args.url, tb, timeout=args.timeout, stop=stop, t_ref=t_ref)
        time.sleep(args.lead)
    r = C.stream_chat(args.url, body, timeout=args.timeout, t_ref=t_ref)
    s = C.step_stats(r)
    time.sleep(0.5 if concurrent else 0)
    stop.set()
    out = {"case": case, "concurrent": concurrent}
    w, h = CASES[case]
    if r["error"]:
        out["error"] = r["error"]
        if case == "video30" and ("video" in r["error"].lower() or "HTTP 400" in r["error"]):
            out["skipped"] = "server rejected video input"
    img_tokens = (s["prompt_tokens"] - base_tokens) if s["prompt_tokens"] else None
    out.update(
        ttft_s=s["ttft_s"], total_s=C.rnd(r["total_s"]), prompt_tokens=s["prompt_tokens"],
        mm_tokens=img_tokens, width=w, height=h,
        answer_head=r["content"][:100],
    )
    if case != "video30":
        exp = expected_tokens(w, h)
        out["expected_unresized"] = exp
        out["resized"] = bool(img_tokens is not None and img_tokens < 0.95 * exp)
    else:
        out["frames"] = args.frames
    if text_future is not None:
        tr = text_future.result()
        pool.shutdown(wait=True)
        a = r["t_send"]
        b = a + (r["times"][0] if r["times"] else r["total_s"])
        g = C.gaps_in_window([tr["t_send"] + t for t in tr["times"]], a, b)
        g_before = C.gaps_in_window([tr["t_send"] + t for t in tr["times"]], 0.0, a)
        out.update(
            text_error=tr["error"],
            text_gap_baseline_p50_ms=C.rnd(C.pct(g_before, 50)),
            text_gap_during_mm_max_ms=C.rnd(max(g) if g else None),
            text_gap_during_mm_p50_ms=C.rnd(C.pct(g, 50)),
            text_chunks_during_mm=len(g),
        )
    else:
        pool.shutdown(wait=True)
    em.row("case", **out)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--solo-only", action="store_true", help="skip the concurrent-text-stream run")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--lead", type=float, default=3.0, help="seconds the text stream decodes before the image")
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()
    C.require_health(args.url)
    em = C.Emitter(__file__, args.out, label=args.label)
    base = {TEXT: baseline_prompt_tokens(args, TEXT), VTEXT: baseline_prompt_tokens(args, VTEXT)}
    results = []
    for case in args.cases:
        bt = base[VTEXT if case == "video30" else TEXT]
        for conc in ([False] if args.solo_only else [False, True]):
            res = run_case(args, em, case, conc, bt)
            print(" ".join(f"{k}={v}" for k, v in res.items() if k != "answer_head"), flush=True)
            results.append(res)
            if res.get("skipped"):
                break
    em.summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
