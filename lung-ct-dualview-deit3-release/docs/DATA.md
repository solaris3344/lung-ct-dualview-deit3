# Data access, licensing and derived artifacts

## 1. Where the images come from

* Dataset: **Chest CT-Scan images Dataset**, published by Mohamed Hany on Kaggle.
* URL: <https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images>
* Retrieved licence string (public dataset metadata API, 2026-11-10):
  `Database: Open Database, Contents: © Original Authors`.
* Practical meaning: the *database* is offered under an open-database style
  licence, but **copyright of the individual images stays with the original
  authors**. This repository therefore redistributes **no images**, only
  filenames, byte sizes and SHA-256 digests.
* The provider's own description states the collection was assembled from
  several upstream resources; upstream image provenance is not documented.
  Treat the labels as provider-assigned, not pathology-confirmed.

The original archive ships an *original* `train` / `valid` / `test` directory
layout whose proportions the provider describes as 70 / 10 / 20 %. This study
does **not** reuse that split for evaluation: the provider test directory is
used as the held-out same-source test only after exact-file deduplication, and
the provider train+valid directories are merged into a 558-image development
pool that is re-split into five similarity-group-disjoint folds.

## 2. Expected local layout

Reproduce this directory tree from the provider archive:

```
<DATA_ROOT>/
  train/
    adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/                 194 files
    large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/              114 files
    normal/                                                      33 files
    squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/           152 files
  valid/
    adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/                  23 files
    large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/               21 files
    normal/                                                       6 files
    squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/            15 files
  test/
    adenocarcinoma/                                             120 files
    large.cell.carcinoma/                                        51 files
    normal/                                                      27 files
    squamous.cell.carcinoma/                                     90 files
```

Total **846 files** (843 `.png`, 3 `.jpg`): 558 development (493 train + 65 valid)
and 288 test.

The class of a development image is taken from the **prefix of its parent folder
name** (`adenocarcinoma…`, `large.cell.carcinoma…`, `normal…`,
`squamous.cell.carcinoma…`), which is why the long surgical folder names are
preserved. The class of a test image is taken from its parent folder name.

## 3. Matching the released manifests to your copy

`manifests/images_846.csv` is the authoritative inventory. Columns:

| Column | Meaning |
|---|---|
| `relative_path` | path relative to `<DATA_ROOT>`, forward slashes, e.g. `test/normal/21.png`. This is the **image identifier** used in every prediction file. |
| `split` | original provider split: `train`, `valid` or `test` |
| `class_folder` | parent folder name as shipped |
| `class_short` | `ADC`, `LCC`, `NOR`, `SCC` |
| `class_index` | 0–3 in the order ADC, LCC, NOR, SCC (`configs/class_order.json`) |
| `group` | dHash similarity group `G000`–`G507`; empty for test rows |
| `outer_fold` | 0–4; empty for test rows |
| `bytes` | file size |
| `sha256` | SHA-256 of the file content |

Verify your copy before running anything:

```bash
python - <<'PY'
import csv, hashlib, pathlib
root = pathlib.Path("<DATA_ROOT>")
bad = 0
for r in csv.DictReader(open("manifests/images_846.csv", encoding="utf-8")):
    p = root / r["relative_path"]
    if not p.is_file():
        print("MISSING", r["relative_path"]); bad += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != r["sha256"]:
        print("HASH MISMATCH", r["relative_path"]); bad += 1
print("checked 846 images, problems:", bad)
PY
```

`manifests/manifest_summary.json` records the aggregate counts, the exact
class/fold composition and a single `manifest_digest_sha256` over the ordered
`(relative_path, sha256)` pairs, so one number is enough to confirm that a
rebuilt inventory is identical.

## 4. What "curated" means here

The curated inventory was produced by exact-file deduplication (SHA-256) of a
1,000-file provider directory inventory that contained cross-folder copies; 846
distinct contents survived. The released manifest confirms:

* 846 unique `relative_path` values and 846 unique `sha256` values,
* **0** shared SHA-256 values between the 558 development and 288 test images,
* **0** dHash similarity group crossing the training/validation side of any fold.

`curated` does **not** mean patient-disjoint. dHash groups are a *similarity
proxy*, not patient identity. The test set was compared to development images
with a 64-bit dHash; 16 test files were flagged at Hamming distance ≤ 3 and none
were exact duplicates (`results/test/holdout_overlap_audit.json`). Those 16
flags are a sensitivity note, not proof of subject independence.

## 5. Derived artifacts that are released here

| Artifact | File | Licence status |
|---|---|---|
| Inventory with digests | `manifests/images_846.csv` | factual metadata; no image content |
| Test manifest | `manifests/test_manifest_288.csv` | as above |
| dHash groups and folds | `manifests/dev_groups.csv`, `manifests/dev_folds.csv`, `manifests/dev_plan_v3.json` | derived from image content by the authors' code |
| Per-image probabilities | `results/dev/oof_v34.csv`, `results/test/test_predictions_v36.csv` | authors' output; contains no pixel data |
| External aggregate reports | `results/external_dicom/*/report_external.json` | aggregate statistics only |

## 6. Derived artifacts that are **not** released

| Artifact | Why | Published substitute |
|---|---|---|
| 846 raw images | provider copyright; size | SHA-256 per file |
| `final_seed42/52/62.pt` (343 MB each) | size; derived from ImageNet-pretrained weights | SHA-256 in `results/checkpoints/CHECKPOINTS.md` |
| `region_features.npz`, `dev_features.npz`, `holdout_frozen_features.npz` | binary caches | file digest + feature-array digest in `manifests/feature_cache_provenance.json` |
| `selector_full_DEV.joblib` | fitted binary | digest only |
| External per-image predictions | contain a patient-identifier column | patient-level aggregate reports |
| `pred_seed*_fold*.npz` (15 files) | binary caches already aggregated in `oof_v34.csv` | the CSV |

## 7. Filename safety

Filenames were checked: they contain no patient names, no hospital accession
numbers, no dates of birth, no e-mail addresses and no absolute paths. Names
such as `000119 (6).png`, `ad13.png`, `n6 (2) - Copy.jpg` and
`003828_02_01_174.png` are provider-assigned export names. Section 4 of
`docs/SECURITY_SCAN.md` records the scan that was run over the released files.
