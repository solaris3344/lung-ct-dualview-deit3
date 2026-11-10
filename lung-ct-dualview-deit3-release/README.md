# Similarity-aware dual-view DeiT3 for four-class lung CT image classification

Reproducibility release package for the manuscript

> **Similarity-Aware Evaluation of a Shared-Weight Dual-View DeiT3 Model with Crop-Policy Ablation for Lung CT Classification**
> Serkan Keser, Şekip Esat Hayber

**Repository status: scope-limited reproducibility release.**
This package contains the code, the curated dataset-relative manifests, the split/fold
plans, the locked development OOF predictions and the locked 288-image test predictions
of the **shared-weight dual-view DeiT3 system (arm A2)**. It does **not** contain the
artifacts of the A1/A3/A4 baseline arms or the S1/S2/S3 crop-policy arms; those results
are therefore **not reproducible from this repository** and are reported as
unverifiable in `release_audit.md` and `docs/MISSING_ARTIFACTS.md`.

One-command verification (read-only, standard library only, no GPU, no data download):

```bash
python scripts/smoke_check.py
```

---

## 1. What this package contains

| Area | Files | Status |
|---|---|---|
| Training code (A2, 20 epochs, 5 folds x 3 seeds) | `src/experiment_v34.py` | complete, byte-identical to source |
| Training code (A2 predecessor, 15 epochs) | `src/experiment_v33.py` | complete, byte-identical |
| Locked-test / final-refit / external evaluation | `src/validate_v36.py`, `src/v34_reference.py` | complete, byte-identical |
| Similarity grouping + group-disjoint fold plan | `src/devcv.py`, `src/audit_dataset.py`, `src/duplicate_review.py` | complete, byte-identical |
| Curated inventory, groups, folds, class order | `manifests/` | 846 images, 508 groups, 5 folds |
| Development OOF predictions (558 images) | `results/dev/oof_v34.csv` | complete, byte-identical |
| Locked test predictions (288 images) | `results/test/test_predictions_v36.csv` | complete, **absolute local path column removed** |
| Lock hashes (plan, test manifest, feature fingerprint, weights) | `results/test/holdout_test_lock.json`, `refit_complete.json`, `audit_v36.json` | complete |
| External DICOM experiments (separate analysis unit) | `results/external_dicom/` | aggregate reports only |
| Machine-readable claim table | `manifests/claims.json` | 24 verifiable + 7 unverifiable claims |
| Provenance (source path, source SHA-256, release SHA-256) | `provenance/SOURCE_HASHES.csv`, `provenance/RELEASE_FILES.sha256` | complete |

## 2. What this package deliberately does **not** contain

* Raw CT images (843 PNG + 3 JPG). Obtain them from the original provider, see `docs/DATA.md`.
* Model checkpoints (`final_seed42/52/62.pt`, 343 MB each). Only their SHA-256 values are published, in `results/checkpoints/CHECKPOINTS.md`.
* Feature caches (`region_features.npz`, `dev_features.npz`, `holdout_frozen_features.npz`) and the fitted selector (`selector_full_DEV.joblib`). Their SHA-256 values and the frozen-feature array digest are published in `manifests/feature_cache_provenance.json` so that the recorded fingerprint remains checkable.
* Per-image external DICOM predictions: those rows carry a patient identifier column.
* Any artifact for **A1** (independent global-only DeiT3), **A3** (ResNet-34 + Random Pixel Swap), **A4** (ResNet-34), **S1** (fixed 72 % centre crop), **S2** (fixed 90 % centre crop) or **S3** (deterministic random-crop) — see §6.

## 3. Quick start

```bash
python -m pip install -r requirements.txt   # only needed to re-train or re-infer
python scripts/smoke_check.py               # no dependencies, ~2 seconds
python scripts/report_scan.py               # absolute-path / credential / patient-ID scan

# the project's own unit tests (need numpy, scipy, scikit-learn, Pillow)
cd tests/source && PYTHONPATH=../../src python -m unittest -v test_v34 test_v33 test_v36
```

At staging time the smoke test reported `PASS 24 / FAIL 0 / UNVERIFIABLE 7`, the scan
reported 0 blocking findings, and the three original test modules reported
**15 tests, OK** (`docs/ENVIRONMENT.md`).

`scripts/smoke_check.py` re-hashes every released file, rebuilds the inventory
invariants (846 / 558 / 288 / 508 / 5 folds / no group leakage / no dev-test content
overlap), recomputes every claim in `manifests/claims.json` that has an artifact in this
package, and prints which file each number came from. It writes nothing.

Expected tail of the output:

```
PASS 24   FAIL 0   UNVERIFIABLE/PARTIAL 7
...
SMOKE TEST PASSED for everything that has a released artifact.
```

## 4. Verify the provenance locks

`results/test/holdout_test_lock.json` was written **before** test inference. Two of its
fields are reproducible from files in this package:

| Field | Reproduced by |
|---|---|
| `test_manifest_sha256` = `00f29024…b3d3` | `sha256(json.dumps([(relative_path, sha256), …], sort_keys=True))` over the 288 rows of `manifests/test_manifest_288.csv`, in path order |
| `fingerprint` = `5763e693…8946` | plan signature + development path order + class indices + raw group strings + view names + feature-array digest, recipe in `manifests/feature_cache_provenance.json` |

Both checks are part of the smoke test (claims C20 and C21). The three checkpoint
SHA-256 values recorded in `holdout_test_lock.json`, `refit_complete.json` and
`test_report_v36.json` were re-verified against the on-disk weights at staging time;
the weights themselves are not redistributed.

## 5. Reported results that are reproducible here

Class order everywhere: **ADC, LCC, NOR, SCC** (`configs/class_order.json`).
All numbers below are recomputed by `scripts/smoke_check.py` from the two prediction
files; they are not copied from the manuscript.

| Quantity | Value | Recomputed from |
|---|---|---|
| Curated inventory | 846 images = 558 dev (493 + 65) + 288 test | `manifests/images_846.csv` |
| Similarity groups | 508 (largest 7; 32 groups ≥ 2) | `manifests/dev_folds.csv` |
| A2 development OOF (0.85/0.15) | **539/558 = 96.59 %**, macro-F1 0.9454 | `results/dev/oof_v34.csv` |
| A2 OOF global-only / region-only | 536/558 and 527/558 | `results/dev/oof_v34.csv` |
| A2 held-out test (0.85/0.15) | **278/288 = 96.53 %**, macro-F1 **0.9684**, balanced acc 0.9692 | `results/test/test_predictions_v36.csv` |
| A2 test global-only / region-only | 274/288 and 275/288 | `results/test/test_predictions_v36.csv` |
| External Lung-PET-CT-Dx (ADC/SCC, patient level) | AUROC 0.597 | `results/external_dicom/Lung-PET-CT-Dx/report_external.json` |
| External NSCLC-Radiomics full-field (ADC/SCC, patient level) | AUROC 0.531 | `results/external_dicom/NSCLC-Radiomics-FullField-DiagV314/report_external.json` |

The two external cohorts contain **no NORMAL and no LCC** images and are reported as
ADC/SCC conditional analyses, **not** as four-class external validation. Weak transfer
(AUROC 0.597 / 0.531) is a limitation, not a validation success.

## 6. Claims that this package cannot verify

The manuscript's Table 9 (A1, A3, A4) and Table 11 (S1, S2, S3) report numbers for
systems whose code, configs, OOF arrays and test predictions are **absent from the
source project** that this package was staged from. A full-text and filename search of
the source tree, including every JSON/CSV/Markdown file for the tokens `A1`–`A4` and
`S1`–`S3`, found no such runs; no Random Pixel Swap implementation exists anywhere in
the tree. The only fixed-crop comparison present is an earlier **development-only,
frozen-feature** probe (`runs/adaptive_region_v31/report_v31.json`: centre-72 % 470/558,
centre-90 % 479/558), which is a different experiment from the manuscript's retrained
S1/S2/S3 controls and cannot stand in for them.

Consequences, stated explicitly:

* 272/288 (A1), 263/288 (A3), 256/288 (A4), 276/288 (S1), 273/288 (S2) and 278/288 (S3) are **not** reproducible here.
* The claim that A2 beats S3 on the test set must **not** be made; the manuscript itself reports both at 278/288 with an exact McNemar p = 1.000.
* This package must not be cited as a full reproduction of the manuscript's matched benchmark until the missing arms are generated and deposited.

## 7. Licensing in one paragraph

Our code: **no license file is included** — the authors must choose one before the
repository becomes public (see `docs/LICENSING.md`; this is a release blocker, not an
oversight). The DeiT3 weights and the timm library are Apache-2.0. The image dataset is
distributed by its Kaggle author under “Database: Open Database, Contents: © Original
Authors”, i.e. the **contents remain copyright of the original authors**; raw images are
therefore not redistributed here, only filenames, sizes and SHA-256 values.
Third-party notices: `THIRD_PARTY_NOTICES.md`.

## 8. Citation

See `CITATION.cff`. Until the missing arms are deposited, please cite the manuscript and
describe this repository as a partial (A2 + data/manifest) reproducibility package.

## 9. Layout

```
README.md                  this file
CITATION.cff               author metadata taken from the manuscript
requirements.txt           runtime dependencies (training/inference only)
THIRD_PARTY_NOTICES.md     licenses of code, weights and data used
CHANGELOG_STAGING.md       every modification applied to a copied source file
configs/                   class order and protocol constants
docs/                      DATA, PROTOCOL, RELEASE_CHECKLIST, PROVENANCE,
                           MISSING_ARTIFACTS, LICENSING, SECURITY_SCAN, METHOD_FREEZE
manifests/                 inventory, groups, folds, test manifest, claims, feature provenance
results/                   locked OOF/test/external results and checkpoint hashes
src/                       training, selector, inference and split-plan code
scripts/                   smoke test, scan, release-manifest generator
tests/source/              original unit tests shipped with the source project
provenance/                source->release hashes and the release file manifest
```
