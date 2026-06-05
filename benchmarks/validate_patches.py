#!/usr/bin/env python3
"""Validate that LATENT patch files are well-formed and report what they touch.

This does NOT apply the patches - that requires `git apply --check` from
inside a llama.cpp checkout. This just inspects the patch headers and
verifies they at least parse as unified diffs.

Usage:
  python benchmarks/validate_patches.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = REPO_ROOT / "patches"

DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def inspect_patch(path: Path) -> dict:
    """Return summary info about a patch file."""
    text = path.read_text(encoding="utf-8")
    files = DIFF_HEADER_RE.findall(text)
    plus = sum(1 for line in text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    minus = sum(1 for line in text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {
        "patch": path.name,
        "size_bytes": path.stat().st_size,
        "files_touched": [a for a, _ in files],
        "lines_added": plus,
        "lines_removed": minus,
    }


def main() -> int:
    if not PATCHES_DIR.is_dir():
        print(f"ERROR: {PATCHES_DIR} not found", file=sys.stderr)
        return 1

    patches = sorted(p for p in PATCHES_DIR.glob("*.patch"))
    if not patches:
        print(f"No patch files in {PATCHES_DIR}")
        return 0

    print("=" * 72)
    print("LATENT Patch Validation")
    print("=" * 72)
    print()

    for p in patches:
        info = inspect_patch(p)
        print(f"--- {info['patch']} ---")
        print(f"  size       : {info['size_bytes']} bytes")
        print(f"  +/-        : +{info['lines_added']} -{info['lines_removed']}")
        print(f"  files      :")
        for f in info["files_touched"]:
            print(f"    {f}")
        print()

    print("To verify they apply cleanly, run from inside a llama.cpp checkout:")
    print("  for p in ../../patches/*.patch; do git apply --check \"$p\"; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
