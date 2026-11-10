# Artifacts that are missing from the source project

This document exists because the task of releasing the project required locating
the training, inference, selector, split, statistics and figure code for the
manuscript arms **A1, A2, A3, A4** and **S1, S2, S3**. A2 was found and is
released. The other six arms were **not found**, and this file records exactly
what was searched, what was found instead, and what would have to be produced to
close the gap.

## Search performed

Read-only, over the whole source project tree (`wavecv_net`, ~7,400 run files,
7.4 GB) plus every manuscript/report file on the same machine:

| Search | Result |
|---|---|
| Directory and file names matching `deit`, `dualview`, `dual_view`, `selector`, `lung_ct` | only the A2 lineage (`lung_region_v33`, `lung_region_v34_20ep`, `lung_region_v35_30ep`, `lung_region_v36_validation`) and unrelated chest-CT experiments |
| File names matching `A1`, `A2`, `A3`, `A4`, `S1`, `S2`, `S3` | no arm-named artifact; the only `equiv_A1` / `equiv_A2` hits are an unrelated equivalence-test harness (`runs/equiv_*`) |
| Tokens `A1`–`A4`, `S1`–`S3` inside every `*.md` in the source tree | no matches at all |
| Tokens `center72`, `center90`, `crop policy`, `crop-policy`, `random crop`, `random-crop` in `*.json` | only selector view counts and one **development-only frozen-feature** probe (see M4 below) |
| `random pixel swap`, `rps`, `resnet34`, `resnet_34` in `*.py` | no matches anywhere in the source tree |
| Directories named `*baseline*`, `*matched*` | none that contain a matched-baseline run |

The names A1–A4 and S1–S3 come from the **manuscript text**, not from the file
tree: the manuscript's own arm definitions exist only in prose and tables.

## M1 — A1: independent global-only DeiT3-Base

* Manuscript: OOF 94.80 % (529/558), test 94.44 % (272/288).
* Needed: 5 folds × 3 seeds of a global-only trainer, its OOF array and its 288-row
  test prediction file with the same class order and image identifiers.
* Found instead: `chest_deit3_original/` (`runs/chest_deit3_original/test_predictions.csv`)
  belongs to the **earlier 613/72/315 provider split**, not to the 846-image cleaned
  benchmark, and `GRIP_Fuse_DEV_Only/train_AERS/experiment_global_only.py` is a
  development-only pilot for a different experiment family.
* Consequence: 272/288 is **not reproducible** and is marked `UNVERIFIABLE` (claim C25).

## M2 — A3 / A4: ResNet-34 with and without Random Pixel Swap

* Manuscript: A3 91.32 % (263/288), A4 88.89 % (256/288).
* Needed: the RPS implementation with its pinned upstream commit, the ResNet-34
  trainer, 5 folds × 3 seeds per arm, and two 288-row prediction files.
* Found instead: nothing. There is **no RPS code anywhere in the source tree**, no
  ResNet-34 trainer, and the existing ResNet artifacts are ResNet-**50** runs on the
  old 613/72/315 split (`runs/chest_resnet50*`).
* Consequence: 263/288 and 256/288 are **not reproducible** (claims C26, C27), and the
  third-party RPS attribution/licence cannot be audited (release blocker B4).

## M3 — S1 / S2 / S3: matched crop-policy controls

* Manuscript: S1 fixed 72 % centre crop OOF 531/558, test 276/288; S2 fixed 90 %
  centre crop OOF 535/558, test 273/288; S3 deterministic random crop OOF 535/558,
  test 278/288. The manuscript states each policy used 15 fold models plus three
  final refits.
* Needed: three trained families (45 fold models + 9 refits), their configs, their
  OOF arrays, their 288-row test prediction files, and the crop-assignment seed/rule
  for S3.
* Found instead: **nothing matching the description.** The only fixed-crop
  comparison in the tree is `runs/adaptive_region_v31/report_v31.json`, an
  exploratory **development-only** probe using *frozen ImageNet features* rather than
  retrained dual-view models:

  | v31 probe (frozen features, no retraining) | OOF correct |
  |---|---|
  | global only | 462/558 |
  | centre-72 fixed | 470/558 |
  | centre-90 fixed | 479/558 |
  | mean over six crops | 476/558 |

  These numbers are **not** S1/S2/S3: different estimator (frozen features vs
  retrained DeiT3), no test evaluation, no seeds, no fold models. Using them to
  populate Table 11 would be a fabrication.
* Consequence: S1/S2/S3 are **not reproducible** (claims C28–C31). In particular the
  manuscript's statement that A2 and S3 both reach 278/288 cannot be verified here,
  and **the repository must not claim that the learned selector is superior to S3**.

## M4 — Figure-generation code

No script in the source tree produces the manuscript's Figures 1–8. The only image
files near the project root (`figure_architecture_rev2.png`, `figure_study_flow.png`)
belong to an unrelated earlier manuscript. Release blocker B5.

## M5 — Statistics code for the paired comparisons

`src/experiment_v34.py` contains a group bootstrap (`compare()`), and the
manuscript's McNemar/Holm analyses are described but no script implementing the
exact McNemar + Holm correction over the eight pairwise comparisons exists in the
source tree. Because the A1/A3/A4/S1/S2/S3 predictions are absent, those comparisons
could not be run anyway. If the manuscript's statistic is to be reproducible, a
`stats/` module should be added and unit-tested against the manuscript's Table 10.

## M6 — Per-image external DICOM predictions

`results/external_dicom/*/external_image_predictions.csv` exist in the source project
but each row contains a `patient_id` column (e.g. a TCIA subject identifier). They are
**deliberately excluded** from this release. Only the aggregate cohort reports are
published. If per-image external predictions are ever required, the patient column
must be replaced by a study-internal pseudonym and the mapping kept private.

## M7 — Exact dependency versions

The source project's `requirements.txt` files list package names without versions.
Bit-exact re-training therefore cannot be promised; see `docs/PROVENANCE.md`.

## Bottom line

Of the seven reported result arms (A1–A4, S1–S3 plus the data/manifest layer), this
package can verify **A2** and the **data/manifest layer**. The remaining arm results
must either be deposited or removed from the manuscript before the repository is
described as a reproduction of the paper.
