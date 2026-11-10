# Executable protocol (what the released code actually does)

This document describes the protocol **as implemented in the released code and
configs**, not as summarised in prose in the manuscript. Where a number exists in
a released artifact, the artifact is named.

| Item | Value | Source of record |
|---|---|---|
| Classes | ADC, LCC, NOR, SCC (index 0–3, fixed) | `configs/class_order.json` |
| Curated images | 846 | `manifests/images_846.csv` |
| Development pool | 558 (493 provider-train + 65 provider-valid) | `manifests/images_846.csv` |
| Held-out test | 288 provider-test images, deduplicated | `manifests/test_manifest_288.csv` |
| Similarity grouping | 508 dHash groups on the 558 development images (largest 7; 32 groups ≥ 2) | `manifests/dev_folds.csv`, `manifests/groups_meta_v3.json` |
| Outer CV | 5 group-disjoint folds, fixed, sizes 112/111/112/112/111 | `manifests/dev_plan_v3.json` |
| Inner selector CV | 3 stratified-group folds inside each outer training part, seed `1137 + fold` | `src/experiment_v34.py::select_views` |
| Seeds | 42, 52, 62 (unique, enforced) | `src/experiment_v34.py` argument guard |
| Backbone | `deit3_base_patch16_224.fb_in22k_ft_in1k` | `src/experiment_v34.py`, `results/dev/training_config_v34.json` |
| Input | 224 × 224, bicubic resize, RGB, EXIF orientation corrected, aspect ratio not preserved | `src/experiment_v34.py` |
| Normalization | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | `src/experiment_v34.py` |
| Regional candidates | centre 72 %, centre 90 %, and four 72 % corner crops | `configs/protocol_constants.json` |
| Selector | frozen 768-D DeiT3 features; per-view StandardScaler (clip ±8) + class-balanced logistic regression C = 0.03; pick the crop with the highest maximum class probability | `src/experiment_v34.py::expert_prob`, `::select_views` |
| Cross-fitting | training images get an out-of-fold crop from the inner split; validation images get a crop from experts refitted on the whole outer training part | `src/experiment_v34.py` |
| Training loss | class-weighted CE, weights sqrt(n_train/(4·n_class)); 0.80 global + 0.20 regional | `src/experiment_v34.py` |
| Optimizer | AdamW, weight decay 0.02; LR 3e-5 backbone / 3e-4 head; cosine annealing over 20 epochs | `results/dev/training_config_v34.json` |
| Schedule | 20 epochs, batch 2, accumulation 4, epoch 1 backbone frozen, AMP, grad-clip 2.0, gradient checkpointing | `results/dev/training_config_v34.json` |
| Augmentation | RandomAffine(7°, translate 3 %, scale 0.97–1.03, fill 0) | `src/experiment_v34.py` |
| Final probability | arithmetic mean over seeds, then `0.85 · P_global + 0.15 · P_region` | `results/dev/report_v34.json`, `results/test/final_refit_config.json` |
| Test procedure | one final refit of 3 models on all 558 development images, then **one** locked inference on the 288 test images | `docs/METHOD_FREEZE.md`, `results/test/holdout_test_lock.json` |
| TTA / threshold tuning | none | `docs/METHOD_FREEZE.md` |

## 1. Two metric families — do not mix them

1. **Development OOF (558 images)** — the 5 × 3 = 15 fold models. The reported
   value is `internal_fixed_15`: **539/558 = 96.59 %**, macro-F1 0.9454.
   Recomputed from `results/dev/oof_v34.csv`.
2. **Held-out test (288 images)** — a *separate* 3-seed final refit on all 558
   development images. The reported value is `fixed_15`: **278/288 = 96.53 %**,
   macro-F1 0.9684, balanced accuracy 0.9692. Recomputed from
   `results/test/test_predictions_v36.csv`.

The refit model is **not** the 15-fold ensemble. The OOF arrays cannot be used to
predict the test images, and the two accuracies above are not repeated
measurements of the same estimator. Any statement that compares them must say so.
This asymmetry is why `results/test/audit_v36.json` records
`saved_fold_weights_count: 0`.

## 2. What the 0.85 / 0.15 blend does and does not mean

`0.85 · P_global + 0.15 · P_region` is a **fixed, pre-declared** mixture. It is not
a calibrated confidence model, not a learned gate and not evidence that the
regional view is clinically informative. The released OOF file also contains a
guarded variant (`internal_guarded`), 537/558, which is **not** used for the
headline numbers.

## 3. Region selection is geometric, not anatomical

The six candidates are rectangles. No lung mask, no lesion mask, no segmentation
model and no radiologist annotation is involved. A selected crop must never be
described as a lesion localisation. The word "region" in this repository means
"rectangle of the image", nothing more.

## 4. Similarity grouping is not patient separation

dHash groups are constructed from perceptual hashes plus a manually reviewed
candidate list (79 approved edges, 0 exact-pixel edges,
`manifests/groups_meta_v3.json`). Group-disjoint folds prevent *identified
perceptual duplicates* from crossing folds. They do not prove patient
independence, and neighbouring slices of one patient may land in different
groups. The manuscript's own limitation section says the same.

## 5. Test exploration status

The 288-image test directory is publicly distributed and had been used in earlier
project stages. The comparisons reported here are therefore **exploratory
same-source benchmarks**, not a prospective validation on untouched data. The
lock file (`results/test/holdout_test_lock.json`) proves only that *this
particular* inference run used a frozen manifest and frozen weights — it does not
make the source data pristine.

## 6. External DICOM cohorts — a separate analysis unit

`results/external_dicom/` holds three aggregate reports produced by
`src/validate_v36.py --stage external`:

| Cohort | Images | Patients | Patient-level ADC/SCC AUROC |
|---|---|---|---|
| Lung-PET-CT-Dx | 1211 | 251 | 0.597 |
| NSCLC-Radiomics | 994 | 201 | 0.517 |
| NSCLC-Radiomics-FullField-DiagV314 | 994 | 201 | 0.531 |

These cohorts contain **only ADC and SCC** images, so no four-class external
accuracy can be claimed; the analysis unit is the patient, not the image; and the
imaging chain (DICOM conversion, windowing, slice selection) differs from the
training PNG exports. Transfer is weak. These numbers must not be presented as a
successful external validation and must not be merged with the 846-image
benchmark tables.

## 7. Reproduction entry points

```bash
# 0. verify the package as shipped (no data needed)
python scripts/smoke_check.py

# 1. rebuild the grouping / fold plan (needs the 558 development images)
python src/devcv.py --help

# 2. re-run the development CV (needs the 558 development images + the frozen
#    7-view feature cache; expect 15 folds x 3 seeds of GPU time)
python src/experiment_v34.py --stage check   --root <PROJECT_ROOT>
python src/experiment_v34.py --stage train   --root <PROJECT_ROOT> --seeds 42 52 62 --epochs 20
python src/experiment_v34.py --stage evaluate --root <PROJECT_ROOT> --seeds 42 52 62

# 3. final refit + the single locked test inference (needs the 288 test images)
python src/validate_v36.py --stage audit --root <PROJECT_ROOT>
python src/validate_v36.py --stage refit --root <PROJECT_ROOT>
python src/validate_v36.py --stage test  --root <PROJECT_ROOT>
```

`<PROJECT_ROOT>` must contain the `datasets/`, `runs/`, `manifests/`-style tree
described in `docs/DATA.md`, `docs/PROVENANCE.md` and the defaults in
`src/experiment_v34.py` (`--csv`, `--plan`, `--groups`, `--features`,
`--data-root`). Every path argument is explicit or repository-relative; no code
file contains an absolute local path.

**Do not re-run the test stage to hunt for a better number.** `validate_v36.py`
refuses to overwrite an existing `test_report_v36.json`; that guard is part of the
protocol, not an inconvenience.
