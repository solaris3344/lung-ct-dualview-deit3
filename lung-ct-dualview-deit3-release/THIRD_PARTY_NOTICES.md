# Third-party notices and attribution

This package contains original research code plus references to third-party
software, pretrained weights and a third-party image dataset. It does **not**
redistribute any of the items below.

## 1. timm and the DeiT III backbone

* Library: `timm` (pytorch-image-models), Ross Wightman and contributors, **Apache License 2.0**.
* Model id used by every script: `deit3_base_patch16_224.fb_in22k_ft_in1k`.
* Upstream architecture and weights: DeiT III, Touvron et al., *DeiT III: Revenge of the ViT* (arXiv:2204.07118), `facebookresearch/deit`, **Apache License 2.0** (verified against the upstream `LICENSE` file and the model card's `license: apache-2.0` tag).
* The pretrained weights are downloaded by `timm` at run time. They are not
  stored in this package. The fine-tuned checkpoints produced by this work are
  also **not** redistributed; only their SHA-256 values are published.
* ImageNet pre-training data is not redistributed and is subject to the
  ImageNet terms of use.

## 2. Scientific Python stack

`numpy` (BSD-3-Clause), `scipy` (BSD-3-Clause), `scikit-learn` (BSD-3-Clause),
`Pillow` (MIT-CMU), `joblib` (BSD-3-Clause), `torch` / `torchvision` (BSD-3-Clause).
These are dependencies, not vendored code.

## 3. Image dataset

* Source: *Chest CT-Scan images Dataset*, Mohamed Hany, Kaggle,
  <https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images>.
* Kaggle-reported licence string (retrieved 2026-11-10 from the public dataset
  metadata API): **“Database: Open Database, Contents: © Original Authors”**,
  i.e. an ODbL-style database licence where the **contents stay copyright of the
  original authors**.
* Consequence: this package publishes only *file names, byte sizes and SHA-256
  digests* of the curated images, never the images themselves, and it does not
  re-license the imagery. Anyone reusing the images must obtain them from the
  provider and accept the provider's terms.
* The dataset author describes the collection as assembled from multiple
  upstream sources; the upstream image provenance is **not** documented by the
  provider. See `docs/LICENSING.md`, item L2.

## 4. Random Pixel Swap (RPS) — manuscript reference [25]

The manuscript's arm **A3** uses an RPS augmentation attributed to Abe and
Nyathi. **No RPS implementation exists in the staged source tree**, and no
licence text, commit hash or vendored copy could be audited. Nothing from that
work is included here. If A3 is added later, its licence and attribution must be
resolved before publication (`docs/MISSING_ARTIFACTS.md`, item M3).

## 5. External DICOM cohorts

The aggregate external reports refer to Lung-PET-CT-Dx and NSCLC-Radiomics from
The Cancer Imaging Archive (TCIA). Those cohorts are distributed under their own
TCIA/CC licences and are **not** redistributed here; only aggregate metrics and
the manifest hashes recorded by the evaluation code are published.

## 6. Our own code

No software licence has been chosen for the authors' code yet. Until one is
added, default copyright applies and the repository must not be described as
open source (`docs/LICENSING.md`, item L1).
