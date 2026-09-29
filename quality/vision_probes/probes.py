"""Synthetic vision probes (plan 0-8 V, G09): drawn in code, written as PNG with zlib + struct.

Every image is deterministic, and keys.json pins its sha256, so a change to
the drawing code fails loudly instead of silently moving the baseline.

Geometry (fab0aec preprocessor_config.json): patch 16, merge 2, so one image
token covers 32x32 px; sizes here are multiples of 32 so no resize happens.
  grid_wide   2048x512  -> 64x16 = 1024 tokens (H and W mRoPE sections differ)
  grid_tall   512x2048  -> 16x64 = 1024 tokens
  straddle    2048x2048 -> 64x64 = 4096 tokens, after >= 6k text tokens (C29)
"""
from __future__ import annotations

import base64
import hashlib
import math
import re
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from t3_needles import filler, needle_code  # noqa: E402

RGB = {"red": (230, 25, 25), "green": (20, 170, 40), "blue": (25, 60, 230),
       "yellow": (250, 220, 20), "black": (0, 0, 0), "white": (255, 255, 255),
       "grey": (200, 200, 200)}
PX_PER_TOKEN = 32

# 5x7 block glyphs, '#' = ink.
FONT = {
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
}


def png_bytes(width: int, height: int, rgb: bytes) -> bytes:
    """8-bit RGB PNG, filter 0 on every row."""
    if len(rgb) != width * height * 3:
        raise ValueError("pixel buffer size does not match width x height x 3")
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class Canvas:
    def __init__(self, width: int, height: int, color=RGB["white"]):
        self.w, self.h = width, height
        self.px = bytearray(bytes(color) * (width * height))

    def rect(self, x0: int, y0: int, x1: int, y1: int, color) -> None:
        x0, x1 = max(0, x0), min(self.w, x1)
        if x1 <= x0:
            return
        row = bytes(color) * (x1 - x0)
        for y in range(max(0, y0), min(self.h, y1)):
            i = (y * self.w + x0) * 3
            self.px[i:i + len(row)] = row

    def circle(self, cx: int, cy: int, r: int, color) -> None:
        for y in range(cy - r, cy + r + 1):
            dx = int(math.sqrt(max(0, r * r - (y - cy) ** 2)))
            self.rect(cx - dx, y, cx + dx + 1, y + 1, color)

    def text(self, s: str, x: int, y: int, scale: int, color=RGB["black"]) -> None:
        for k, ch in enumerate(s):
            if ch == " ":
                continue
            for row, bits in enumerate(FONT[ch]):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        gx = x + (k * 6 + col) * scale
                        gy = y + row * scale
                        self.rect(gx, gy, gx + scale, gy + scale, color)

    def png(self) -> bytes:
        return png_bytes(self.w, self.h, bytes(self.px))


# ------------------------------------------------------------------ images

def solid(color: str, size: int = 256) -> bytes:
    return Canvas(size, size, RGB[color]).png()


def quadrants(size: int = 512) -> bytes:
    c, h = Canvas(size, size), size // 2
    c.rect(0, 0, h, h, RGB["red"])
    c.rect(h, 0, size, h, RGB["green"])
    c.rect(0, h, h, size, RGB["blue"])
    c.rect(h, h, size, size, RGB["yellow"])
    return c.png()


def grid(cols: int, rows: int, cell: int, marks: dict) -> bytes:
    """cols x rows grid of cell-px squares with 4-px grey lines; marks {(row, col): colour}, 1-based."""
    c = Canvas(cols * cell, rows * cell)
    for (r, k), color in marks.items():
        c.rect((k - 1) * cell + 16, (r - 1) * cell + 16, k * cell - 16, r * cell - 16, RGB[color])
    for k in range(cols + 1):
        c.rect(k * cell - 2, 0, k * cell + 2, rows * cell, RGB["grey"])
    for r in range(rows + 1):
        c.rect(0, r * cell - 2, cols * cell, r * cell + 2, RGB["grey"])
    return c.png()


def grid_wide() -> bytes:
    return grid(8, 2, 256, {(1, 6): "red", (2, 3): "blue"})


def grid_tall() -> bytes:
    return grid(2, 8, 256, {(7, 1): "green"})


def digits(s: str, width: int, height: int, scale: int) -> bytes:
    c = Canvas(width, height)
    tw, th = (len(s) * 6 - 1) * scale, 7 * scale
    c.text(s, (width - tw) // 2, (height - th) // 2, scale)
    return c.png()


BARS = [("red", 300), ("green", 620), ("blue", 450), ("yellow", 180), ("black", 520)]


def bars() -> bytes:
    c = Canvas(1024, 768)
    c.rect(60, 700, 1000, 704, RGB["black"])  # x axis
    for i, (color, h) in enumerate(BARS):
        x = 100 + i * 180
        c.rect(x, 700 - h, x + 120, 700, RGB[color])
    return c.png()


def circles(n: int) -> bytes:
    c = Canvas(768, 512)
    for i in range(n):
        c.circle(96 + i * 144, 256 if i % 2 else 200, 48, RGB["black"])
    return c.png()


STRADDLE_DIGITS = "5831"


def straddle_image() -> bytes:
    return digits(STRADDLE_DIGITS, 2048, 2048, 48)


IMAGES = {
    "solid_red": lambda: solid("red"), "solid_green": lambda: solid("green"),
    "solid_blue": lambda: solid("blue"), "solid_yellow": lambda: solid("yellow"),
    "quadrants": quadrants, "grid_wide": grid_wide, "grid_tall": grid_tall,
    "digits_407218": lambda: digits("407218", 1536, 384, 24),
    "digits_9315": lambda: digits("9315", 1024, 512, 40),
    "bars": bars, "circles_3": lambda: circles(3), "circles_5": lambda: circles(5),
    "straddle": straddle_image,
}


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def image_part(name: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(IMAGES[name]())}}


# ------------------------------------------------------------------ probes

STRADDLE_PARAS = 64  # >= 6k text tokens before the image (recorded live in keys.json)
STRADDLE_SEED = 29   # C29


def straddle_text() -> tuple[str, str, str]:
    name, code = needle_code(STRADDLE_SEED)
    paras = filler(STRADDLE_SEED, STRADDLE_PARAS)
    mid = STRADDLE_PARAS // 2
    note = f"The vault code assigned to {name} is {code}. It was never written down again."
    text = "\n\n".join(["Field notes follow."] + paras[:mid] + [note] + paras[mid:])
    return text, name, code


def _one(q: str) -> str:
    return q + " Reply with only the answer."


def probes() -> list[dict]:
    """Each probe: id, images (names), text, key, match ('word' | 'digits' | 'all')."""
    wide_q = "The image is a grid of 2 rows and 8 columns of squares, numbered from 1 on the left."
    tall_q = "The image is a grid of 8 rows and 2 columns of squares, rows numbered from 1 at the top."
    stext, sname, scode = straddle_text()
    return [
        {"id": "grid_wide.red_col", "images": ["grid_wide"], "match": "digits", "key": "6",
         "text": _one(wide_q + " One square is red. Which column is it in? Answer with the column number.")},
        {"id": "grid_wide.blue_col", "images": ["grid_wide"], "match": "digits", "key": "3",
         "text": _one(wide_q + " One square is blue. Which column is it in? Answer with the column number.")},
        {"id": "grid_wide.blue_row", "images": ["grid_wide"], "match": "word", "key": "bottom",
         "text": _one(wide_q + " One square is blue. Is it in the top row or the bottom row? Answer top or bottom.")},
        {"id": "grid_tall.green_row", "images": ["grid_tall"], "match": "digits", "key": "7",
         "text": _one(tall_q + " One square is green. Which row is it in? Answer with the row number.")},
        {"id": "grid_tall.green_col", "images": ["grid_tall"], "match": "word", "key": "left",
         "text": _one(tall_q + " One square is green. Is it in the left or the right column? Answer left or right.")},
        {"id": "ocr.407218", "images": ["digits_407218"], "match": "digits", "key": "407218",
         "text": _one("What number is written in the image? Answer with the digits only.")},
        {"id": "ocr.9315", "images": ["digits_9315"], "match": "digits", "key": "9315",
         "text": _one("What number is written in the image? Answer with the digits only.")},
        {"id": "chart.tallest", "images": ["bars"], "match": "word", "key": "green",
         "text": _one("This bar chart has five coloured bars. What colour is the tallest bar? One word.")},
        {"id": "chart.shortest", "images": ["bars"], "match": "word", "key": "yellow",
         "text": _one("This bar chart has five coloured bars. What colour is the shortest bar? One word.")},
        {"id": "chart.order", "images": ["bars"], "match": "word", "key": "blue",
         "text": _one("This bar chart has five coloured bars. What colour is the third bar from the left? One word.")},
        {"id": "count.circles3", "images": ["circles_3"], "match": "digits", "key": "3",
         "text": _one("How many black circles are in the image? Answer with a number.")},
        {"id": "count.circles5", "images": ["circles_5"], "match": "digits", "key": "5",
         "text": _one("How many black circles are in the image? Answer with a number.")},
        {"id": "quadrants.bottom_right", "images": ["quadrants"], "match": "word", "key": "yellow",
         "text": _one("The image has four coloured quadrants. What colour is the bottom-right quadrant? One word.")},
        {"id": "multi.blue_second", "images": ["solid_red", "solid_blue"], "match": "word", "key": "second",
         "text": _one("Two images are shown. Which image is blue, the first or the second? One word.")},
        {"id": "multi.blue_first", "images": ["solid_blue", "solid_red"], "match": "word", "key": "first",
         "text": _one("Two images are shown. Which image is blue, the first or the second? One word.")},
        {"id": "straddle.c29", "images": ["straddle"], "match": "all", "key": [STRADDLE_DIGITS, scode],
         "prefix_text": stext,
         "text": ("What four-digit number is written in the image? And what is the vault code assigned to "
                  f"{sname} in the notes above? Reply exactly as: <number>; <code>")},
    ]


def pixel_sha(png: bytes) -> str:
    """sha256 of IHDR + decompressed IDAT: pins the pixels, not the zlib build's byte stream."""
    pos, ihdr, idat = 8, b"", b""
    while pos < len(png):
        (n,) = struct.unpack(">I", png[pos:pos + 4])
        tag, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
        ihdr = data if tag == b"IHDR" else ihdr
        idat += data if tag == b"IDAT" else b""
        pos += 12 + n
    return hashlib.sha256(ihdr + zlib.decompress(idat)).hexdigest()


def image_shas() -> dict:
    return {name: pixel_sha(fn()) for name, fn in sorted(IMAGES.items())}


def messages(probe: dict) -> list[dict]:
    """Text before the image(s) only for the straddle probe, so the image sits after >= 6k tokens."""
    parts = []
    if probe.get("prefix_text"):
        parts.append({"type": "text", "text": probe["prefix_text"] + "\n\n"})
    parts += [image_part(n) for n in probe["images"]]
    parts.append({"type": "text", "text": probe["text"]})
    return [{"role": "user", "content": parts}]


# Competing answers are the other options of the key's own group, so "The second
# image is blue." still passes multi.blue_second.
ANSWER_GROUPS = [set(RGB), {"top", "bottom"}, {"left", "right"}, {"first", "second"}]


def judge(probe: dict, content: str) -> bool:
    """digits: the first line's digits equal the key; word: the key and no competing answer word;
    all: every key string present."""
    text = content.strip()
    if probe["match"] == "digits":
        return re.sub(r"\D", "", text.splitlines()[0] if text else "") == probe["key"]
    if probe["match"] == "word":
        words = set(re.findall(r"[a-z]+", text.lower()))
        rivals = next((g for g in ANSWER_GROUPS if probe["key"] in g), set()) - {probe["key"]}
        return probe["key"] in words and not words & rivals
    up = text.upper()
    return all(k.upper() in up for k in probe["key"])
