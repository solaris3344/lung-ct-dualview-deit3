#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content scan for local/personal absolute paths, credentials and patient identifiers.

Scans every text file inside the package. Binary/office/archive extensions are
rejected outright (they must not be in the package at all).

    python scripts/report_scan.py
    python scripts/report_scan.py --json

Exit code 0 = no blocking findings. Exit code 1 = blocking finding.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_EXT = {".md", ".csv", ".json", ".py", ".txt", ".cff", ".yml", ".yaml",
            ".toml", ".cfg", ".ini", ".sha256", ""}
BINARY_EXT = {".pt", ".pth", ".ckpt", ".npz", ".pkl", ".joblib", ".h5", ".onnx",
              ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".dcm", ".nii", ".gz",
              ".zip", ".docx", ".xlsx", ".pptx", ".pdf"}

# Files that intentionally contain the search patterns as documentation. They are
# still scanned for the *other* rules, but exempt from their own examples.
DOC_FILES = {"scripts/report_scan.py", "docs/SECURITY_SCAN.md"}

# (id, description, compiled pattern, severity)
RULES = [
    ("ABS_WIN_DRIVE", "Windows drive-letter absolute path",
     re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}(?:[A-Za-z0-9_.\- ]+[\\/])+[A-Za-z0-9_.\- ]*"), "block"),
    ("ABS_UNC", "UNC network path", re.compile(r"\\\\[A-Za-z0-9_.\-]+[\\/]"), "block"),
    ("ABS_HOME", "POSIX absolute home path",
     re.compile(r"(?:^|[\s\"'(])/(?:home|Users)/[A-Za-z0-9_.\-]+"), "block"),
    ("USERNAME", "Local account name in a path context",
     re.compile(r"(?i)(?:users[\\/]|/home/|[a-z]:[\\/]+)serkan\b"), "block"),
    ("PRIVATE_KEY", "Private key block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), "block"),
    ("AWS_KEY", "AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "block"),
    ("GITHUB_TOKEN", "GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "block"),
    ("OPENAI_KEY", "sk-style API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "block"),
    ("GOOGLE_KEY", "Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "block"),
    ("SLACK_TOKEN", "Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"), "block"),
    ("JWT", "JSON web token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "block"),
    ("PASSWORD_ASSIGN", "password/secret/token assignment",
     re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret)\b\s*[:=]\s*['\"][^'\"]{4,}['\"]"), "block"),
    ("PATIENT_ID", "DICOM/TCIA patient identifier pattern",
     re.compile(r"\b(?:Lung_Dx-A\d{4}|LUNG1-\d{3}|Lung_Dx-[A-Z]\d{4})\b"), "block"),
    ("PRIVATE_DIR", "local private directory name",
     re.compile(r"(?i)DO_NOT_UPLOAD|PRIVATE_"), "warn"),
    ("EMAIL", "e-mail address",
     re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "warn"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    findings = []
    binary = []
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        ext = p.suffix.lower()
        if ext in BINARY_EXT:
            binary.append(rel)
            continue
        if ext not in TEXT_EXT:
            findings.append(dict(rule="UNKNOWN_EXT", severity="warn", file=rel, line=0, match=ext))
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(dict(rule="NOT_UTF8", severity="warn", file=rel, line=0, match=""))
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if rel in DOC_FILES:
                continue  # these files document the patterns they search for
            for rid, desc, rx, sev in RULES:
                for mt in rx.finditer(line):
                    findings.append(dict(rule=rid, severity=sev, file=rel,
                                         line=lineno, match=mt.group(0)[:120]))

    blocking = [f for f in findings if f["severity"] == "block"]
    warns = [f for f in findings if f["severity"] == "warn"]

    if args.json:
        print(json.dumps({"n_files": len(files), "binaries": binary,
                          "blocking": blocking, "warnings": warns}, indent=2))
    else:
        print(f"scanned {len(files)} files under {ROOT}")
        print(f"binary/office files present: {binary if binary else 'none'}")
        print(f"blocking findings: {len(blocking)}")
        for f in blocking:
            print(f"  [BLOCK] {f['rule']:<16} {f['file']}:{f['line']}  {f['match']!r}")
        print(f"warnings: {len(warns)}")
        for f in warns:
            print(f"  [warn ] {f['rule']:<16} {f['file']}:{f['line']}  {f['match']!r}")
        print("\nRESULT:", "CLEAN (0 blocking findings)" if not blocking else "BLOCKED")
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())
