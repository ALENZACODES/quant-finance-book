"""The code listings a chapter prints, read from its LaTeX source, so that a test can run each one
exactly as it appears on the page: the lines in the printed range, in order, and none wider than the
column that the listing style shows without wrapping."""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 88          # characters per line that the book's listing style prints without a break
SUFFIX = {"Python": ".py", "R": ".R", "Octave": ".m"}
RUNNERS = {"Python": [sys.executable], "R": ["Rscript"], "Octave": ["octave-cli", "--no-gui", "-q"]}
PAT = re.compile(r"\\lstinputlisting\[([^\]]*)\]\{([^}]*)\}")


def _ranges(opts: str, n: int):
    m = re.search(r"linerange=\{([^}]*)\}", opts)
    if not m:
        return [(1, n)]
    out = []
    for part in m.group(1).split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def printed(ch: str):
    """Return (path, language, printed text) for every listing in chapters/chNN.tex, merging the
    ranges printed from one file in the order they appear."""
    tex = (ROOT / "chapters" / f"ch{ch}.tex").read_text()
    by_file: dict[str, list] = {}
    for opts, path in PAT.findall(tex):
        lang = re.search(r"language=([A-Za-z]+)", opts).group(1)
        lines = (ROOT / path).read_text().split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        chunk = [l for a, b in _ranges(opts, len(lines)) for l in lines[a - 1:b]]
        by_file.setdefault(path, [lang, []])[1].extend(chunk)
    return [(p, lang, "\n".join(ls) + "\n") for p, (lang, ls) in by_file.items()]


def too_wide(text: str):
    return [l for l in text.split("\n") if len(l.expandtabs(4)) > WIDTH]


def run(lang: str, text: str, suffix: str, where: str = "", timeout: int = 300):
    """Run the printed text as a fresh file from the repository root, with the folder of the listing
    on the Python path so that one listing can import another; return (returncode, stderr), or None
    when the interpreter for the language is not installed."""
    import os
    cmd = RUNNERS[lang]
    env = dict(os.environ, PYTHONPATH=str((ROOT / where).parent) if where else "")
    if shutil.which(cmd[0]) is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / f"printed{suffix}"
        f.write_text(text)
        r = subprocess.run(cmd + [str(f)], cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env=env)
    return r.returncode, r.stderr[-2000:]
