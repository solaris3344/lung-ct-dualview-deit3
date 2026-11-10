# Results

Two prediction families are released, plus one separate external analysis unit.
They answer different questions and must not be pooled.

## 1. Development OOF — `results/dev/`

| File | Content |
|---|---|
| `oof_v34.csv` | 558 rows, one per development image. `relative_path`, `true_class`, `group`, `outer_fold`, `selected_region`, then for each of four probability sets a `*_pred` column and four `*_p0…*_p3` columns. |
| `report_v34.json` | Metrics recomputed by the training script for the four probability sets, with confusion matrices and per-class recalls. |
| `preflight_v34.json` | Per-outer-fold selected-crop counts (the selector's actual behaviour). |
| `training_config_v34.json` | Frozen run configuration and the plan signature / feature fingerprint. |

The four probability sets in `oof_v34.csv` are:

| Column prefix | Meaning | Correct / 558 |
|---|---|---|
| `trained_global` | global view only | 536 |
| `trained_region` | selected regional view only | 527 |
| `internal_fixed_15` | **0.85 · global + 0.15 · region — the published A2 OOF number** | **539 (96.59 %)** |
| `internal_guarded` | confidence-gated variant, not used for any headline number | 537 |

Aggregation over the three seeds happens before these columns are written; the
file is the pooled three-seed OOF, not a single seed. `selected_region` records
which of the seven views the cross-fitted selector chose for that image.

## 2. Locked test — `results/test/`

| File | Content |
|---|---|
| `test_predictions_v36.csv` | 288 rows, one per held-out test image: `relative_path`, `true_class`, `selected_view`, `global_pred`, `regional_pred`, `final_pred`, and the three probability sets (`global_p*`, `region_p*`, `final_p*`). The original absolute `image_path` column was removed (see `CHANGELOG_STAGING.md`). |
| `test_report_v36.json` | The locked report: `global` 274/288, `region` 275/288, `fixed_15` **278/288**, Wilson intervals, macro-F1, balanced accuracy, confusion matrices, per-class recalls, plus the freeze block. |
| `holdout_test_lock.json` | Written **before** inference: `n_expected: 288`, method string, test-manifest digest, feature fingerprint, three checkpoint SHA-256 values. |
| `refit_complete.json` | Proof that the 3-seed final refit finished and that no test data was read during it. |
| `audit_v36.json` | Why the test stage could not reuse the 15 fold models (`saved_fold_weights_count: 0`), the original development result (539/558), and the count/class/plan identity block. |
| `holdout_overlap_audit.json` | 0 exact duplicates, 16 dHash-flagged near-duplicate candidates with their minimum Hamming distances. Heuristic, not patient linkage. |
| `feature_extractor_replay.json` | Replay error of the frozen extractor against the cached features (max abs ≈ 5e-3), i.e. the cache is reproducible only approximately. |
| `final_refit_config.json` | The refit configuration (identical architecture to V3.4, trained on all 558 development images). |

Reading rules:

* `final_pred` is the published A2 decision (argmax of `final_p*`).
* `true_class` and `*_pred` are **indices** into `["ADC","LCC","NOR","SCC"]`.
* `selected_view` is the crop the selector chose for that image; it is a
  rectangle label, not a lesion location.
* 278/288 is a same-source, publicly distributed test set that had been explored
  in earlier project stages. It is an exploratory benchmark result.

## 3. External DICOM — `results/external_dicom/`

**Separate analysis unit.** Different cohorts, different imaging chain, different
label space (ADC/SCC only), patient-level primary metric. See
`results/external_dicom/README.md`. Never merge these numbers with §1 or §2.

## 4. Checkpoints — `results/checkpoints/`

Hashes only; the 343 MB binaries are not redistributed. See
`results/checkpoints/CHECKPOINTS.md`.

## 5. What is not here

`A1`, `A3`, `A4`, `S1`, `S2`, `S3` have no result files in this package. No
placeholder, no synthetic array and no "reconstructed" prediction file was
created for them. `docs/MISSING_ARTIFACTS.md` records the search that established
this, and `manifests/claims.json` marks the corresponding claims `UNVERIFIABLE`.
