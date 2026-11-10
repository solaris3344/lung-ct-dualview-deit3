# Licence audit

Scope: the authors' code in this package, the third-party weights and libraries it
depends on, the image dataset, and the third-party augmentation cited by the
manuscript. Status values are **CLEAR**, **ACTION REQUIRED** or **BLOCKER**.

| id | Item | Finding | Status |
|---|---|---|---|
| L1 | The authors' own code (`src/`, `scripts/`, `tests/`) | No `LICENSE`, `COPYING` or SPDX header anywhere in the source project or in this package. Without a licence, default copyright applies: others may not legally reuse, modify or redistribute the code even if it is public on GitHub. | **BLOCKER** |
| L2 | Image dataset | Kaggle metadata says `Database: Open Database, Contents: © Original Authors`. The database may be shared, but the **image contents stay copyright of the original authors**. The provider also states the images were assembled from several upstream resources without documenting them. Publishing filenames, sizes and SHA-256 is metadata; publishing the pixels would not be covered. | **ACTION REQUIRED** (keep images out; keep attribution) |
| L3 | DeiT III weights and `timm` | `timm` is Apache-2.0. The `deit3_base_patch16_224.fb_in22k_ft_in1k` checkpoint is Apache-2.0 (upstream `facebookresearch/deit` LICENSE is Apache-2.0; the model card carries `license: apache-2.0`). Apache-2.0 requires retaining notices and stating modifications; weights are downloaded by `timm` at run time and are not redistributed here. | **CLEAR** (notice included in `THIRD_PARTY_NOTICES.md`) |
| L4 | Fine-tuned checkpoints (`final_seed*.pt`) | They are derivative works of the Apache-2.0 weights plus the ImageNet pre-training data. They are **not** redistributed; only SHA-256 values are published, which is a factual statement, not a distribution. If they are ever uploaded, carry the Apache-2.0 notice and mark the modification. | **CLEAR (as released)** |
| L5 | Random Pixel Swap (manuscript ref. [25], arm A3) | No RPS implementation, commit hash or licence text exists in the source project, so the third-party attribution cannot be audited and the licence cannot be verified. Nothing from that work is included. | **BLOCKER** if A3 is published as part of this repository |
| L6 | TCIA external cohorts (Lung-PET-CT-Dx, NSCLC-Radiomics) | Only aggregate metrics are published. The cohorts are not redistributed. TCIA collections carry their own CC-BY / TCIA terms; the aggregate numbers are the authors' own results. | **CLEAR (as released)** |
| L7 | Python dependencies | numpy / scipy / scikit-learn / Pillow / joblib BSD-family; torch / torchvision BSD-3-Clause. Used as dependencies, not vendored. | **CLEAR** |
| L8 | Manuscript figures and tables | Figures were produced by an unknown toolchain (no script in the source project). If a figure embeds dataset pixels, its reuse is covered by L2, not by the code licence. | **ACTION REQUIRED** before figures are added |

## Recommended resolution order

1. **L1 first** — add a `LICENSE` file. Apache-2.0 keeps the stack consistent with
   the DeiT3/timm dependency chain; MIT is simpler if compatibility with the
   upstream notice is not needed. This is the authors' decision, not a decision that
   can be made in a staging script.
2. **L2** — keep the "no pixels, metadata only" rule and keep the attribution block
   in `THIRD_PARTY_NOTICES.md` and `docs/DATA.md`. If the provider's terms are
   ambiguous, cite the licence string verbatim and link the dataset page.
3. **L5** — if A3 is ever added, pin the upstream RPS repository commit, copy its
   `LICENSE` into `third_party/rps/`, and state the modifications. If the upstream
   licence is unknown or non-permissive, either re-implement RPS from the published
   description with a citation, or drop the arm.
4. **L8** — record the figure toolchain, or state in the caption that the figure is
   not reproducible from the repository.

## What this package does *not* do

* It does not choose a licence on the authors' behalf.
* It does not re-licence the dataset or the pretrained weights.
* It does not claim that the dataset's upstream image provenance has been resolved.
