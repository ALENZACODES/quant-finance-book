"""Shared registry for the Chapter 12 scripts: number macros, figure saving and output."""
from __future__ import annotations
import json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small matrices: one thread is faster and repeatable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
from bookstyle import plt, save as _save, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: E402,F401
import ch12lib as L  # noqa: E402,F401

SEED = 8317  # an arbitrary label for the chapter, not a date
N: dict[str, str] = {}
RAW: dict = {}


def save(fig, name):
    return _save(fig, name, chapter="ch12")


def put(name, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    RAW[name] = float(value)
    s = fmt.format(value)
    if s.startswith("-") and float(s.replace(",", "")) == 0:
        s = s[1:]
    N[name] = ("\\ensuremath{-}" + s[1:]) if s.startswith("-") else s


def pct(name, value, d=2):
    put(name, 100 * value, "{:.%df}" % d)


def finish(script):
    """Write the numbers registered so far to out/<script>.json and return them."""
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / f"{script}.json").write_text(json.dumps(dict(N=N, RAW=RAW), indent=1))
    return dict(N=dict(N), RAW=dict(RAW))


def rng_for(k):
    """Each script draws from its own stream, so a script run alone reproduces run_all.py."""
    import numpy as np
    return np.random.default_rng([SEED, k])


def sci(name, value, d=1):
    """Register a number in scientific notation, typeset as m \\times 10^{e}."""
    RAW[name] = float(value)
    m, e = f"{value:.{d}e}".split("e")
    N[name] = f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"
