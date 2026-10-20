"""Shared registry for the Chapter 2 scripts: number macros, figure saving and output."""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
from bookstyle import plt, save as _save, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: E402,F401
import ch02lib as L  # noqa: E402,F401

SEED = 202
N: dict[str, str] = {}
RAW: dict[str, float] = {}


def save(fig, name):
    return _save(fig, name, chapter="ch02")


def put(name, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    RAW[name] = float(value)
    s = fmt.format(value)
    N[name] = ("\\ensuremath{-}" + s[1:]) if s.startswith("-") else s


def pct(name, value, d=2):
    put(name, 100 * value, "{:.%df}" % d)


def word(i):
    return ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"][i]


def finish(script):
    """Write the numbers registered so far to out/<script>.json and return them."""
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / f"{script}.json").write_text(json.dumps(dict(N=N, RAW=RAW), indent=1))
    return dict(N=dict(N), RAW=dict(RAW))
