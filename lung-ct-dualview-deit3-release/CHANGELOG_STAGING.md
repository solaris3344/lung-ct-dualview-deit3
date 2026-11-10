# Staging change log

Everything that differs between the source project and this package. The source
project was never modified; all changes were made on copies inside the staging
folder. Hashes for every file are in `provenance/SOURCE_HASHES.csv`.

## A. Files copied byte-for-byte (no change)

31 of the 33 source-derived files are byte-identical to the source project; their
`source_sha256` and `release_sha256` are equal and `modified` is `no`. That
includes all training/inference/split code, the original unit tests, the method
freeze note, the inventory/split inputs and every locked report except the two
listed in §B.

Folder move only, file names preserved (content unchanged, so hashes are equal).
Module basenames were deliberately **not** renamed: `tests/source/test_v34.py`
does `from experiment_v34 import …` and `validate_v36.py` does
`from v34_reference import …`, so renaming the modules would have forced edits to
original files and broken those imports.

| Released as | Copied from |
|---|---|
| `src/experiment_v34.py` | `lung_region_v34_20ep/experiment_v34.py` |
| `src/validate_v36.py` | `lung_region_v36_validation/validate_v36.py` |
| `src/v34_reference.py` | `lung_region_v36_validation/v34_reference.py` |
| `src/experiment_v33.py` | `lung_region_v33/experiment_v33.py` |
| `src/devcv.py` | `chest_duplicate_aware_cv_v3/devcv.py` |
| `src/audit_dataset.py` | `chest_duplicate_aware_cv_v3/audit_dataset.py` |
| `src/duplicate_review.py` | `chest_duplicate_aware_cv_v3/duplicate_review.py` |
| `tests/source/test_v33.py` | `lung_region_v33/test_v33.py` |
| `tests/source/test_v34.py` | `lung_region_v34_20ep/test_v34.py` |
| `tests/source/test_v36.py` | `lung_region_v36_validation/test_v36.py` |
| `docs/METHOD_FREEZE.md` | `lung_region_v36_validation/METHOD_FREEZE.md` |
| `manifests/dev_image_list_v3.csv` | `lung_adaptive_fusion_v1/anatomy_review_V3_eslestirilmis.csv` |
| `manifests/dev_groups.csv` | `runs/chest_duplicate_aware_cv_v3/groups.csv` |
| `manifests/dev_plan_v3.json` | `runs/chest_duplicate_aware_cv_v3/plan.json` |
| `manifests/groups_meta_v3.json` | `runs/chest_duplicate_aware_cv_v3/groups_meta.json` |

Because the module names are unchanged, the shipped unit tests can be run from
`tests/source/` with `PYTHONPATH` pointing at `src/`, exactly as in the source
project:

## B. Modified copies (2)

### B1. `results/test/test_predictions_v36.csv`

* source `3d0d0d4925f52b2e…17f140c` (122 574 bytes) → release `b9c972e28e9632d1…9361d026` (89 896 bytes)
* **Change:** removed the leading `image_path` column, which held a full Windows
  path for each of the 288 rows.
* **Not changed:** row order, `relative_path`, `true_class`, `selected_view`,
  `global_pred`, `regional_pred`, `final_pred`, the twelve probability columns
  (kept as the original text, not re-formatted floats).
* **Why:** the release must not carry personal absolute paths, and
  `relative_path` already identifies the image unambiguously against
  `manifests/test_manifest_288.csv`.

### B2. `results/test/audit_v36.json`

* source `906723c0eba800ec…8adceed3` (1 202 bytes) → release `168f68fa3fe8fff5…8c0d5b2c` (1 037 bytes)
* **Change:** `v34_out` and the three `final_seed_weights` values were absolute
  Windows paths and were replaced with repository-relative paths
  (`runs/adaptive_region_v34_20ep_3seed`,
  `runs/adaptive_region_v36_test_locked/final_seed4x.pt`).
* **Not changed:** every count, hash, signature and verdict
  (`saved_fold_weights_count: 0`, `original_v34_heldout_DEV_only.correct: 539`,
  class order, plan signature, feature fingerprint).

## C. Files deliberately not copied

| Not copied | Why |
|---|---|
| `lung_region_v33/README_TR.md`, `lung_region_v34_20ep/README_TR.md`, `lung_region_v36_validation/README_TR.md` | they embed absolute local Windows paths inside their `cd …` run instructions; replaced by `docs/PROTOCOL.md` |
| `lung_region_v3x/BASLAT_*.ps1`, `DURUM_*.ps1` | machine-specific launchers |
| `*.pyc`, `__pycache__` | compiled caches |
| `external_*/external_image_predictions.csv` | contain a `patient_id` column |
| `external_*/` PNG review sheets, `q1_*` diagnostic sheets | image content, and unrelated to the manuscript tables |
| `model_weights.torch` (project root) | 738 MB, belongs to a different manuscript |
| `runs/**/*.pt`, `*.npz`, `*.joblib` | checkpoints/caches; digests published instead |
| `LPE3D_Q1_Manuscript_REVISED.docx`, `figure_*.png` | different manuscript |
| `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv` | clinical table for the external cohort (out of scope, potential personal data) |

## D. Newly written files (not derived from the source project)

`README.md`, `CITATION.cff`, `requirements.txt`, `.gitignore`, `THIRD_PARTY_NOTICES.md`,
`release_audit.md`, `docs/{DATA,PROTOCOL,RELEASE_CHECKLIST,PROVENANCE,MISSING_ARTIFACTS,LICENSING,SECURITY_SCAN,CLAIMS}.md`,
`configs/*.json`, `manifests/{images_846.csv,test_manifest_288.csv,dev_folds.csv,manifest_summary.json,feature_cache_provenance.json,claims.json,README.md}`,
`results/README.md`, `results/checkpoints/*`, `results/external_dicom/README.md`,
`scripts/*.py`, `provenance/RELEASE_FILES.sha256`.

`manifests/images_846.csv`, `test_manifest_288.csv`, `dev_folds.csv` and
`manifest_summary.json` are **computed** from the read-only images and from
`dev_groups.csv` / `dev_plan_v3.json`; they were not hand-edited. The smoke test
re-derives the same numbers from them.

## E. No re-training, no re-inference

Nothing was trained, refit or inferred during staging. The test set was not read
for evaluation. The only reads of image files were the 846 SHA-256 computations
needed to build the inventory.

## F. What was executed during staging

| Action | Result |
|---|---|
| `py_compile` on all source/script/test modules | exit 0 |
| `python -m unittest test_v34 test_v33 test_v36` (from `tests/source/`, `PYTHONPATH=src`) | 15 tests, OK |
| `python scripts/smoke_check.py` | 24 claims PASS, 0 FAIL, 7 unverifiable |
| `python scripts/report_scan.py` | 0 blocking findings |
| SHA-256 of the three `final_seed*.pt` | matches all three lock records |

Details in `docs/ENVIRONMENT.md`. Byte-compilation produced `__pycache__` folders,
which were deleted before the release manifest was generated.
