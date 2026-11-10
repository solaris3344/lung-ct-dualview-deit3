# Provenance and reproducibility record

## 1. Method

The source project was scanned **read-only**. Every file that appears in this
package was copied byte-for-byte unless a sanitisation is listed in §3, and every
copy was hashed before and after:

* `source_sha256` — SHA-256 of the file where it lives in the source project,
* `release_sha256` — SHA-256 of the file inside this package,
* `modified` — `no` when the two hashes are identical.

The full table (33 rows, one per source-derived file) is
`provenance/SOURCE_HASHES.csv`. `provenance/RELEASE_FILES.sha256` lists
**every** file in the package, and `scripts/smoke_check.py` verifies both.

## 2. Released code and where it came from

| Release path | Source path (relative to the project root) | Bytes | SHA-256 |
|---|---|---|---|
| `src/experiment_v34.py` | `lung_region_v34_20ep/experiment_v34.py` | 33465 | `d8447516e96e8b4c…aaf70e1` |
| `src/validate_v36.py` | `lung_region_v36_validation/validate_v36.py` | 34480 | `8cbce71fda31086e…4ff264f6` |
| `src/v34_reference.py` | `lung_region_v36_validation/v34_reference.py` | 33465 | `d8447516e96e8b4c…aaf70e1` |
| `src/experiment_v33.py` | `lung_region_v33/experiment_v33.py` | 30596 | `04439ebf0548a2ec…22d4756e` |
| `src/devcv.py` | `chest_duplicate_aware_cv_v3/devcv.py` | 42103 | `7e756ab92e352360…166105f13` |
| `src/audit_dataset.py` | `chest_duplicate_aware_cv_v3/audit_dataset.py` | 6770 | `6a086460d1263c04…7f26f7b` |
| `src/duplicate_review.py` | `chest_duplicate_aware_cv_v3/duplicate_review.py` | 15184 | `7ceb2a9ad127632a…ef1664fd` |
| `tests/source/test_v34.py` | `lung_region_v34_20ep/test_v34.py` | 4751 | `f33e42199d8d1b70…31afdba` |
| `tests/source/test_v36.py` | `lung_region_v36_validation/test_v36.py` | 2664 | `6edcd76fc3663062…bc9473bc` |
| `tests/source/test_v33.py` | `lung_region_v33/test_v33.py` | 4153 | `97fc41d38de2eb8d…3b454287` |
| `docs/METHOD_FREEZE.md` | `lung_region_v36_validation/METHOD_FREEZE.md` | 2109 | `95857f33d2305996…36fc43de` |

`src/experiment_v34.py` and `src/v34_reference.py` are byte-identical in the
source project as well: `validate_v36.py` imports its constants and helpers from
`v34_reference.py`. Both are shipped so the import works without editing.

## 3. Sanitisations applied to copies (2 files)

### 3.1 `results/test/test_predictions_v36.csv`

* Source SHA-256 `3d0d0d4925f5…17f140c`, release SHA-256 `b9c972e28e96…9361d026`.
* Change: the first column `image_path` was **removed**. It contained an absolute
  Windows path for every row (e.g. a local `…\datasets\chest_ct\…` path).
* Everything else is preserved verbatim: same row order (288 rows), same
  `relative_path`, `true_class`, `selected_view`, `global_pred`, `regional_pred`,
  `final_pred` and all twelve probability columns, as text (no float reformatting).
* `relative_path` is now the image identifier, and it is the same value that
  appears in `manifests/test_manifest_288.csv`.

### 3.2 `results/test/audit_v36.json`

* Source SHA-256 `906723c0eba8…8adceed3`, release SHA-256 `168f68fa3fe8…8c0d5b2c`.
* Change: the two absolute Windows paths (`v34_out` and each
  `final_seed_weights` value) were replaced by repository-relative paths.
  No numeric, hash or count field was touched.

## 4. Excluded source artifacts (with digests, so they can still be checked)

| Artifact | Bytes | SHA-256 | Reason |
|---|---|---|---|
| `final_seed42.pt` | 343332431 | `e5c57946fb60d858…07b73640` | checkpoint; hash-only release |
| `final_seed52.pt` | 343332431 | `84d0cecd00e8c8d2…82e7d3b7` | checkpoint; hash-only release |
| `final_seed62.pt` | 343332431 | `0bd601f7a7fb7a13…1c80de4e` | checkpoint; hash-only release |
| `region_features.npz` | 11097019 | `d0a0ffed8df5ae67…01b07f2a` | binary feature cache |
| `dev_features.npz` | 1595344 | `05ef772096edb758…d74eae` | binary feature cache |
| `holdout_frozen_features.npz` | 5721022 | `1b0ad526c8da0502…bd5c916a` | binary feature cache |
| `selector_full_DEV.joblib` | 188808 | `b4149e87ea6ffeda…6c3c95e0` | fitted binary |
| `train_view_choices.npz` | 905 | `f0f4ed1ff08a089c…b77428c` | cached crop choices |
| `pred_seed*_fold*.npz` (15) | ~5 kB each | — | superseded by `results/dev/oof_v34.csv` |
| `model_weights.torch` (project root, unrelated manuscript) | 738451713 | not hashed | out of scope |

The three checkpoint hashes above are exactly the values recorded in
`results/test/holdout_test_lock.json`, `refit_complete.json` and
`test_report_v36.json`, and they were **re-computed from the on-disk files** at
staging time — the lock files were not merely trusted.

## 5. Verification chain (what proves what)

```
846 image files
   └─ manifests/images_846.csv            per-file SHA-256  (re-verified at staging)
        ├─ manifests/dev_folds.csv        558 dev rows, 508 groups, 5 folds
        │     └─ sha256(json) of dev paths + y + groups + views + feature digest
        │           = feature fingerprint 5763e693…  (in every lock file)
        └─ manifests/test_manifest_288.csv
              └─ sha256(json) of (path, sha256) pairs
                    = test_manifest_sha 00f29024…  (in holdout_test_lock.json)
```

Both arrows are executed by `scripts/smoke_check.py` (claims C20, C21), so the two
lock values that were written **before** test inference are reproducible from
released, path-free files.

## 6. Known reproducibility gaps

1. **Dependency versions.** The source project's `requirements.txt` files are
   unpinned. `requirements.txt` in this package gives compatibility ranges, not a
   bit-exact environment. Re-training will not reproduce the exact weights.
2. **The 15 fold checkpoints were never saved.** `experiment_v34.py` defaults to
   `--keep-final-weights` off and deletes each `resume_seed*_fold*.pt` after its
   fold predictions are written. This is why the test stage uses a separate 3-seed
   **final refit** instead of the 15 fold models, and why the OOF arrays cannot be
   used to score new images. Recorded in `results/test/audit_v36.json`
   (`saved_fold_weights_count: 0`).
3. **The frozen feature cache is not released.** The feature fingerprint is still
   reproducible because the array digest is published
   (`manifests/feature_cache_provenance.json`); re-deriving the cache from images
   requires the backbone and a deterministic GPU, so a regenerated array may differ
   in the last bits (`results/test/feature_extractor_replay.json` reports
   max-abs differences of ~5e-3 between the cache and a live replay).
4. **`plan.features_sha256` is a file hash, not an array hash.** It equals the
   SHA-256 of `dev_features.npz`, whereas the fingerprint uses the digest of the
   float array. Both are documented; do not confuse them.
5. **No figure or paired-statistics code.** See `docs/MISSING_ARTIFACTS.md` M4/M5.
6. **The original project directory is not a git repository** for this study — there
   is no commit SHA to cite. Provenance therefore rests on file hashes only.

## 7. How to re-verify everything in this package

```bash
python scripts/smoke_check.py            # re-hashes 33 source-derived + all released files,
                                         # rebuilds the claims, prints per-claim source
python scripts/report_scan.py            # absolute paths / credentials / patient IDs
python -m json.tool manifests/claims.json > /dev/null   # claim table is valid JSON
```

To re-verify the *source* copies (only possible on the machine that holds the
source project), compare `source_sha256` in `provenance/SOURCE_HASHES.csv` against
`sha256sum` of the files at the recorded `source_path`.
