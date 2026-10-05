"""Console conventions: progress bars go to STDOUT, everything else (timestamped
log lines, result tables) goes to STDERR, so ``cmd 2> run.log`` leaves a clean bar
on screen and a full log on disk. Set CHAMFER_QUIET=1 to disable bars (tests do)."""
from __future__ import annotations

import os
import sys
import time

from tqdm import tqdm

_T0 = time.time()


def log(msg: str = "") -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", file=sys.stderr, flush=True)


def table_out(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def bar(total: int, desc: str, leave: bool = False) -> tqdm:
    return tqdm(total=total, desc=desc, file=sys.stdout, leave=leave, dynamic_ncols=True,
                mininterval=0.5, disable=bool(os.environ.get("CHAMFER_QUIET")))
