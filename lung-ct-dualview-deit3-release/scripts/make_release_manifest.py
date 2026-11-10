#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate provenance/RELEASE_FILES.sha256 for this package.

Run this only after the package content is final. The smoke test verifies every
listed file and fails if any file inside the package is not listed.

    python scripts/make_release_manifest.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance" / "RELEASE_FILES.sha256"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = sorted(
        (p for p in ROOT.rglob("*") if p.is_file() and p != OUT),
        key=lambda p: p.relative_to(ROOT).as_posix())
    lines = [f"{sha_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in files]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(lines)} files")
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
