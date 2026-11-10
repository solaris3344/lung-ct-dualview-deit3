#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only smoke test for the lung-ct-dualview-deit3 release package.

What it does
------------
1. Re-hashes every released file and compares it with provenance/SOURCE_HASHES.csv.
2. Rebuilds the dataset-relative inventory invariants (846 / 558 / 288 / 508 / 5 folds
   / no group leakage / no development-test content overlap).
3. Recomputes every claim in manifests/claims.json that has a released artifact,
   reading only CSV/JSON files inside this package.
4. Reports claims whose artifacts are absent as UNVERIFIABLE instead of guessing.

Nothing is written. No network access. No torch/numpy needed (standard library only).

Exit code 0 = all released-artifact claims and all integrity checks pass.
Exit code 1 = at least one check failed.
Exit code 2 = package files missing.

Usage
-----
    python scripts/smoke_check.py
    python scripts/smoke_check.py --verbose
    python scripts/smoke_check.py --json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASS_ORDER = ("ADC", "LCC", "NOR", "SCC")
VIEW_NAMES = ("global", "center72", "center90", "upper_left", "upper_right",
              "lower_left", "lower_right")
LABEL_TO_INDEX = {"adenocarcinoma": 0, "large.cell.carcinoma": 1,
                  "normal": 2, "squamous.cell.carcinoma": 3}
N_DEV = 558
N_TEST = 288
N_IMAGES = 846
N_GROUPS = 508
N_FOLDS = 5


# --------------------------------------------------------------------------- io
def read_csv(rel: str) -> list[dict]:
    p = ROOT / rel
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- statistics
def confusion(y_true, y_pred, k=4):
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def macro_f1(m):
    f1s = []
    for c in range(len(m)):
        tp = m[c][c]
        fp = sum(m[r][c] for r in range(len(m))) - tp
        fn = sum(m[c]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def balanced_accuracy(m):
    recs = []
    for c in range(len(m)):
        n = sum(m[c])
        recs.append(m[c][c] / n if n else 0.0)
    return sum(recs) / len(recs)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------- data loading
class Package:
    def __init__(self):
        self.images = read_csv("manifests/images_846.csv")
        self.test_manifest = read_csv("manifests/test_manifest_288.csv")
        self.folds = read_csv("manifests/dev_folds.csv")
        self.dev_list = read_csv("manifests/dev_image_list_v3.csv")
        self.dev_groups = read_csv("manifests/dev_groups.csv")
        self.plan = read_json("manifests/dev_plan_v3.json")
        self.feature_prov = read_json("manifests/feature_cache_provenance.json")
        self.test_pred = read_csv("results/test/test_predictions_v36.csv")
        self.oof = read_csv("results/dev/oof_v34.csv")
        self.test_report = read_json("results/test/test_report_v36.json")
        self.dev_report = read_json("results/dev/report_v34.json")
        self.lock = read_json("results/test/holdout_test_lock.json")
        self.refit = read_json("results/test/refit_complete.json")
        self.audit = read_json("results/test/audit_v36.json")
        self.ext_lungpet = read_json("results/external_dicom/Lung-PET-CT-Dx/report_external.json")
        self.ext_nsclc_ff = read_json(
            "results/external_dicom/NSCLC-Radiomics-FullField-DiagV314/report_external.json")

    # -- derived
    def test_rows(self):
        return [dict(rel=r["relative_path"], y=int(r["true_class"]),
                     g=int(r["global_pred"]), rg=int(r["regional_pred"]),
                     f=int(r["final_pred"])) for r in self.test_pred]

    def oof_rows(self):
        return [dict(rel=r["relative_path"], y=int(r["true_class"]),
                     g=int(r["trained_global_pred"]), rg=int(r["trained_region_pred"]),
                     f=int(r["internal_fixed_15_pred"])) for r in self.oof]


# ------------------------------------------------------------------- metrics
def build_metrics(pkg: Package) -> dict:
    t = pkg.test_rows()
    o = pkg.oof_rows()
    tm = {r["relative_path"]: r for r in pkg.test_manifest}
    cm_final = confusion([r["y"] for r in t], [r["f"] for r in t])
    cm_global = confusion([r["y"] for r in t], [r["g"] for r in t])
    cm_region = confusion([r["y"] for r in t], [r["rg"] for r in t])
    cm_oof = confusion([r["y"] for r in o], [r["f"] for r in o])

    dev = [r for r in pkg.images if r["split"] in ("train", "valid")]
    test = [r for r in pkg.images if r["split"] == "test"]
    groups = {r["relative_path"]: r["group"] for r in pkg.folds}
    fold_of = {r["relative_path"]: int(r["outer_fold"]) for r in pkg.folds}

    leak = 0
    for k in range(N_FOLDS):
        ho = {groups[p] for p in groups if fold_of[p] == k}
        tr = {groups[p] for p in groups if fold_of[p] != k}
        leak += len(ho & tr)

    # fingerprint recipe (see manifests/feature_cache_provenance.json)
    rel = [r["relative_path"].replace("\\", "/").lstrip("./") for r in pkg.dev_list]
    y = [LABEL_TO_INDEX[r["class_folder"].split("_")[0].lower()] for r in pkg.dev_list]
    gmap = {r["rel"].replace("\\", "/").lstrip("./"): r["group"] for r in pkg.dev_groups}
    group = [gmap[p] for p in rel]
    recipe = pkg.feature_prov["recipe"]
    fingerprint = sha_text({
        "signature": recipe["signature"], "rel": rel, "y": y, "group": group,
        "views": VIEW_NAMES, "features_sha256": recipe["features_sha256"]})

    return {
        "n_images": len(pkg.images),
        "n_dev": len(dev),
        "n_test": len(test),
        "n_train_original": sum(1 for r in dev if r["split"] == "train"),
        "n_valid_original": sum(1 for r in dev if r["split"] == "valid"),
        "n_groups": len(set(groups.values())),
        "group_leakage_folds": leak,
        "dev_test_sha_overlap": len({r["sha256"] for r in dev} & {r["sha256"] for r in test}),
        "test_class_counts": [sum(1 for r in test if r["class_short"] == c) for c in CLASS_ORDER],
        "test_correct_final": cm_final[0][0] + cm_final[1][1] + cm_final[2][2] + cm_final[3][3],
        "test_accuracy_final": sum(1 for r in t if r["f"] == r["y"]) / len(t),
        "test_macro_f1_final": macro_f1(cm_final),
        "test_balanced_accuracy_final": balanced_accuracy(cm_final),
        "test_correct_global": sum(1 for r in t if r["g"] == r["y"]),
        "test_correct_region": sum(1 for r in t if r["rg"] == r["y"]),
        "test_confusion_final": cm_final,
        "test_confusion_global": cm_global,
        "test_confusion_region": cm_region,
        "oof_correct_fixed15": sum(1 for r in o if r["f"] == r["y"]),
        "oof_accuracy_fixed15": sum(1 for r in o if r["f"] == r["y"]) / len(o),
        "oof_macro_f1_fixed15": macro_f1(cm_oof),
        "oof_correct_global": sum(1 for r in o if r["g"] == r["y"]),
        "oof_correct_region": sum(1 for r in o if r["rg"] == r["y"]),
        "test_manifest_sha": sha_text([(r["relative_path"], r["sha256"]) for r in pkg.test_manifest]),
        "feature_fingerprint": fingerprint,
        "plan_signature": pkg.plan["signature"],
        "ext_lungpet_patient_auroc": pkg.ext_lungpet["conditional_adc_scc_patient_level"]["auroc"],
        "ext_nsclc_ff_patient_auroc": pkg.ext_nsclc_ff["conditional_adc_scc_patient_level"]["auroc"],
    }


# ---------------------------------------------------------------- comparisons
def close(a, b, tol):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, list) or isinstance(b, list):
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
            return False
        return all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, int) and isinstance(b, int) and tol == 0:
        return a == b
    try:
        return abs(float(a) - float(b)) <= max(tol, 0.0)
    except (TypeError, ValueError):
        return a == b


# ------------------------------------------------------------------ integrity
def check_provenance() -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    rows = read_csv("provenance/SOURCE_HASHES.csv")
    if not rows:
        return False, ["provenance/SOURCE_HASHES.csv is empty"]
    for r in rows:
        p = ROOT / r["release_path"]
        if not p.is_file():
            ok = False
            msgs.append(f"missing released file: {r['release_path']}")
            continue
        got = sha_file(p)
        if got != r["release_sha256"]:
            ok = False
            msgs.append(f"hash mismatch: {r['release_path']} expected {r['release_sha256'][:12]} got {got[:12]}")
    # unmodified copies must equal their source hash byte-for-byte
    for r in rows:
        if r["modified"] == "no" and r["source_sha256"] != r["release_sha256"]:
            ok = False
            msgs.append(f"unmodified copy changed: {r['release_path']}")
    msgs.append(f"{len(rows)} released files re-hashed")
    return ok, msgs


def check_release_manifest() -> tuple[bool, list[str]]:
    """Verify provenance/RELEASE_FILES.sha256 (tamper-evidence for the whole package)."""
    p = ROOT / "provenance" / "RELEASE_FILES.sha256"
    if not p.is_file():
        return True, ["provenance/RELEASE_FILES.sha256 not generated yet (run scripts/make_release_manifest.py)"]
    ok, msgs = True, []
    listed = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        q = ROOT / rel
        if not q.is_file():
            ok = False
            msgs.append(f"listed but missing: {rel}")
        elif sha_file(q) != digest:
            ok = False
            msgs.append(f"hash mismatch: {rel}")
    unlisted = sorted(str(q.relative_to(ROOT)) for q in ROOT.rglob("*")
                      if q.is_file() and str(q.relative_to(ROOT)).replace("\\", "/") not in listed
                      and q.name != "RELEASE_FILES.sha256")
    if unlisted:
        ok = False
        msgs.append(f"files not covered by RELEASE_FILES.sha256: {unlisted}")
    msgs.insert(0, f"{len(listed)} files verified against provenance/RELEASE_FILES.sha256")
    return ok, msgs


def check_consistency(pkg: Package, m: dict) -> tuple[bool, list[str]]:
    """Cross-check recomputed metrics against the stored report JSONs."""
    ok = True
    msgs = []
    rep = pkg.test_report["methods"]["fixed_15"]
    pairs = [
        ("test_correct_final", rep["correct"]),
        ("test_accuracy_final", rep["accuracy"]),
        ("test_macro_f1_final", rep["macro_f1"]),
        ("test_balanced_accuracy_final", rep["balanced_accuracy"]),
        ("test_confusion_final", rep["confusion_true_rows"]),
    ]
    for key, stored in pairs:
        if not close(m[key], stored, 1e-12):
            ok = False
            msgs.append(f"test_report_v36.json {key}: stored {stored!r} != recomputed {m[key]!r}")
    lo, hi = rep["wilson95_accuracy_ci"]
    rlo, rhi = wilson(rep["correct"], rep["n"])
    if not (close(rlo, lo, 1e-9) and close(rhi, hi, 1e-9)):
        ok = False
        msgs.append(f"test_report_v36.json wilson95_accuracy_ci mismatch: stored {(lo, hi)} recomputed {(rlo, rhi)}")
    for name, key in (("global", "test_correct_global"), ("region", "test_correct_region")):
        if pkg.test_report["methods"][name]["correct"] != m[key]:
            ok = False
            msgs.append(f"test_report_v36.json methods.{name}.correct mismatch")
    orep = pkg.dev_report["methods"]["internal_fixed_15"]
    for key, stored in (("oof_correct_fixed15", orep["correct"]),
                        ("oof_accuracy_fixed15", orep["acc"]),
                        ("oof_macro_f1_fixed15", orep["macro_f1"])):
        if not close(m[key], stored, 1e-12):
            ok = False
            msgs.append(f"report_v34.json {key}: stored {stored!r} != recomputed {m[key]!r}")
    for name, key in (("trained_global", "oof_correct_global"), ("trained_region", "oof_correct_region")):
        if pkg.dev_report["methods"][name]["correct"] != m[key]:
            ok = False
            msgs.append(f"report_v34.json methods.{name}.correct mismatch")
    # lock files must agree with each other
    hashes = [pkg.lock["weights_sha256"], pkg.refit["model_weights"],
              pkg.test_report["freeze"]["weights_sha256"]]
    if not (hashes[0] == hashes[1] == hashes[2]):
        ok = False
        msgs.append("checkpoint SHA-256 values disagree between holdout_test_lock / refit_complete / test_report")
    if pkg.lock["signature"] != m["plan_signature"]:
        ok = False
        msgs.append("holdout_test_lock signature differs from CV plan signature")
    if pkg.lock["test_manifest_sha"] != m["test_manifest_sha"]:
        ok = False
        msgs.append("holdout_test_lock test_manifest_sha differs from recomputed manifest digest")
    if pkg.test_report["freeze"]["fingerprint"] != m["feature_fingerprint"]:
        ok = False
        msgs.append("test_report_v36 fingerprint differs from recomputed frozen-feature fingerprint")
    if pkg.audit["n_dev"] != N_DEV or pkg.audit["n_groups"] != N_GROUPS:
        ok = False
        msgs.append("audit_v36.json n_dev/n_groups mismatch")
    if pkg.audit["original_v34_heldout_DEV_only"]["correct"] != m["oof_correct_fixed15"]:
        ok = False
        msgs.append("audit_v36.json original_v34_heldout_DEV_only.correct mismatch")
    # every test prediction row must be a released test image with the manifest label
    tm = {r["relative_path"]: r for r in pkg.test_manifest}
    if len(pkg.test_pred) != N_TEST:
        ok = False
        msgs.append(f"test_predictions_v36.csv has {len(pkg.test_pred)} rows, expected {N_TEST}")
    for r in pkg.test_rows():
        t = tm.get(r["rel"])
        if t is None:
            ok = False
            msgs.append(f"prediction path not in test manifest: {r['rel']}")
            break
        if int(t["class_index"]) != r["y"]:
            ok = False
            msgs.append(f"label mismatch for {r['rel']}")
            break
    if len(pkg.oof) != N_DEV:
        ok = False
        msgs.append(f"oof_v34.csv has {len(pkg.oof)} rows, expected {N_DEV}")
    # probability columns must be valid 4-class distributions
    for r in pkg.test_pred:
        s = sum(float(r[f"final_p{j}"]) for j in range(4))
        if abs(s - 1) > 3e-3:
            ok = False
            msgs.append("final probability row does not sum to 1")
            break
    if not msgs:
        msgs.append("reports, locks and predictions are mutually consistent")
    return ok, msgs


def check_hygiene() -> tuple[bool, list[str], dict]:
    """Reject raw images, model binaries and caches inside the package."""
    bad_ext = {".pt", ".pth", ".ckpt", ".npz", ".pkl", ".joblib", ".h5", ".onnx",
               ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".dcm", ".nii", ".gz", ".zip",
               ".pyc", ".pyo"}
    offenders = []
    counts = {}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
        if ext in bad_ext:
            offenders.append(str(p.relative_to(ROOT)))
    ok = not offenders
    msgs = [f"image/binary/cache files found: {offenders}"] if offenders else [
        "no images, checkpoints, NPZ caches or archives inside the package"]
    return ok, msgs, counts


# --------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args()

    required = ["manifests/images_846.csv", "manifests/claims.json",
                "results/test/test_predictions_v36.csv", "results/dev/oof_v34.csv",
                "provenance/SOURCE_HASHES.csv"]
    missing = [r for r in required if not (ROOT / r).is_file()]
    if missing:
        print("MISSING PACKAGE FILES:", missing)
        return 2

    pkg = Package()
    m = build_metrics(pkg)
    claims = read_json("manifests/claims.json")

    results = []
    for c in claims:
        if c.get("metric") is None:
            status = "UNVERIFIABLE" if c.get("status") == "unverifiable" else "PARTIAL"
            results.append((c["id"], c["arm"], status, c["expected"], None, c["source"] or "artifact not in release"))
            continue
        got = m.get(c["metric"])
        status = "PASS" if close(got, c["expected"], c["tol"]) else "FAIL"
        results.append((c["id"], c["arm"], status, c["expected"], got, c["source"]))

    prov_ok, prov_msgs = check_provenance()
    rel_ok, rel_msgs = check_release_manifest()
    cons_ok, cons_msgs = check_consistency(pkg, m)
    hyg_ok, hyg_msgs, ext_counts = check_hygiene()
    integ_ok, integ_msgs = True, []
    if m["group_leakage_folds"] != 0:
        integ_ok = False
        integ_msgs.append("similarity-group leakage across folds")
    if m["dev_test_sha_overlap"] != 0:
        integ_ok = False
        integ_msgs.append("development/test exact content overlap")
    if len({r["relative_path"] for r in pkg.images}) != N_IMAGES:
        integ_ok = False
        integ_msgs.append("duplicate or missing manifest paths")

    failed = [r for r in results if r[2] == "FAIL"]
    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_unv = sum(1 for r in results if r[2] in ("UNVERIFIABLE", "PARTIAL"))

    if args.json:
        print(json.dumps({
            "package_root": str(ROOT),
            "metrics": m,
            "claims": [{"id": i, "arm": a, "status": s, "expected": e,
                        "recomputed": g, "source": src}
                       for i, a, s, e, g, src in results],
            "provenance_ok": prov_ok, "release_manifest_ok": rel_ok, "consistency_ok": cons_ok,
            "hygiene_ok": hyg_ok, "integrity_ok": integ_ok,
            "n_pass": n_pass, "n_fail": len(failed), "n_unverifiable": n_unv,
        }, indent=2, default=str))
        return 0 if (prov_ok and rel_ok and cons_ok and hyg_ok and integ_ok and not failed) else 1

    w = 96
    print("=" * w)
    print("lung-ct-dualview-deit3 release package - read-only smoke test")
    print("package root:", ROOT)
    print("=" * w)
    print("\n[1] Recomputable claims (each value is recomputed from the stated file)")
    print("-" * w)
    print(f"{'id':<5} {'arm':<6} {'status':<13} {'recomputed':<24} source")
    for i, a, s, e, g, src in results:
        got = "" if g is None else (f"{g:.10g}" if isinstance(g, float) else str(g))
        if s == "FAIL":
            got = f"{got} != {e}"
        print(f"{i:<5} {a:<6} {s:<13} {got[:23]:<24} {src}")
    print("-" * w)
    print(f"PASS {n_pass}   FAIL {len(failed)}   UNVERIFIABLE/PARTIAL {n_unv}")

    print("\n[2] Package integrity")
    print("-" * w)
    for label, ok, msgs in (("provenance hashes", prov_ok, prov_msgs),
                            ("release file manifest", rel_ok, rel_msgs),
                            ("report/prediction consistency", cons_ok, cons_msgs),
                            ("manifest invariants", integ_ok, integ_msgs or ["846/558/288/508, 5 folds, no leakage"]),
                            ("hygiene (no images/weights/caches)", hyg_ok, hyg_msgs)):
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
        for msg in msgs:
            print(f"        - {msg}")

    print("\n[3] Key recomputed numbers")
    print("-" * w)
    print(f"  inventory            : {m['n_images']} images = {m['n_dev']} dev "
          f"({m['n_train_original']} train + {m['n_valid_original']} valid) + {m['n_test']} test")
    print(f"  similarity groups    : {m['n_groups']} (leakage across 5 folds: {m['group_leakage_folds']})")
    print(f"  test class counts    : " + ", ".join(f"{c}={v}" for c, v in zip(CLASS_ORDER, m["test_class_counts"])))
    print(f"  A2 test  (0.85/0.15) : {m['test_correct_final']}/288  acc={m['test_accuracy_final']:.6f}  "
          f"macro-F1={m['test_macro_f1_final']:.6f}  bal-acc={m['test_balanced_accuracy_final']:.6f}")
    print(f"  A2 test  global only : {m['test_correct_global']}/288   region only: {m['test_correct_region']}/288")
    print(f"  A2 OOF   (0.85/0.15) : {m['oof_correct_fixed15']}/558  acc={m['oof_accuracy_fixed15']:.6f}  "
          f"macro-F1={m['oof_macro_f1_fixed15']:.6f}")
    print(f"  A2 OOF   global only : {m['oof_correct_global']}/558   region only: {m['oof_correct_region']}/558")
    print(f"  test manifest digest : {m['test_manifest_sha']}")
    print(f"  feature fingerprint  : {m['feature_fingerprint']}")
    print(f"  CV plan signature    : {m['plan_signature']}")

    print("\n[4] Coverage of the manuscript result arms")
    print("-" * w)
    arms = {}
    for i, a, s, e, g, src in results:
        arms.setdefault(a, []).append(s)
    for a in ("DATA", "LOCK", "A2", "A2-vs-S3", "EXT", "A1", "A3", "A4", "S1", "S2", "S3"):
        if a in arms:
            if any(x == "FAIL" for x in arms[a]):
                st = "NOT verified"
            elif any(x in ("UNVERIFIABLE", "PARTIAL") for x in arms[a]):
                st = "partial / unverifiable"
            else:
                st = "verified"
            print(f"  {a:<10} {st}")
    print("\n  A1/A3/A4 (independent DeiT3, ResNet-34 +/- RPS) and S1/S2/S3 (fixed 72%/90%")
    print("  centre crop, deterministic random crop) have no released predictions, reports,")
    print("  configs or code. Their numbers are reported as UNVERIFIABLE, never recomputed.")
    print("  See docs/MISSING_ARTIFACTS.md.")

    total_ok = prov_ok and rel_ok and cons_ok and hyg_ok and integ_ok and not failed
    print("\n" + "=" * w)
    if total_ok:
        print("SMOKE TEST PASSED for everything that has a released artifact.")
        print("Scope: A2 (dual-view DeiT3), its 558-image OOF, its locked 288-image test, the")
        print("       inventory/split manifests and the external DICOM aggregate reports.")
        print("NOTE: passing this test does NOT make the A1/A3/A4/S1/S2/S3 claims verifiable.")
    else:
        print("SMOKE TEST FAILED - do not mark this package publish-ready.")
    print("=" * w)
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
