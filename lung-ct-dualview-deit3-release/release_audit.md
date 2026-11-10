# Release audit — lung-ct-dualview-deit3

Staging folder: `lung-ct-dualview-deit3-release` (local staging directory; its absolute path is intentionally omitted from this package, see §7)
Source project (read-only): `wavecv_net` (local project tree; absolute path intentionally omitted)
Audit date: 2026-11-10
Package: 67 files, ≈1.1 MB, 0 images, 0 checkpoints, 0 secrets.

---

## 0. Yönetici özeti / GO–NO-GO (Türkçe)

**Karar: KISMİ GO — daraltılmış kapsamla yayınlanabilir; makalenin tamamını "yeniden
üretilebilir" olarak sunacak şekilde yayınlanamaz (tam kapsam için NO-GO).**

Neden:

1. **Bulunan ve doğrulanan:** Dual-view DeiT3 sisteminin (A2) eğitim, crop-seçici,
   cross-fitting, inference ve test kilitleme kodlarının tamamı; 846 görüntülük
   dataset-relative envanter (SHA-256'lı), 558/288 bölünmesi, 508 dHash grubu, 5 fold,
   sınıf sırası, 558 satırlık OOF ve 288 satırlık kilitli test tahminleri, dış DICOM
   toplu raporları. Smoke test 24 iddiayı bu dosyalardan **yeniden hesaplayarak**
   doğruluyor: A2 test 278/288 (macro-F1 0.9684), A2 OOF 539/558 (macro-F1 0.9454).
   İki provenance kilidi (test manifest özeti `00f29024…`, frozen-feature parmak izi
   `5763e693…`) de paket içinden yeniden üretilebiliyor.
2. **Bulunamayan:** **A1, A3, A4, S1, S2, S3** kollarının kod, konfig, OOF ve test
   tahmin dosyaları kaynak projede **yok**. Kaynak ağaçta `A1`–`A4`, `S1`–`S3`
   belirteçleri hiçbir `.md` dosyasında geçmiyor; Random Pixel Swap (RPS)
   implementasyonu hiçbir `.py` dosyasında yok. Bu altı kolun makale sayıları
   (272/288, 263/288, 256/288, 276/288, 273/288, 278/288) bu paketten
   **doğrulanamaz**; sentetik dosya üretilmedi, "yeniden kurulmuş" tahmin yazılmadı.
3. **Kritik uyarı:** A2'nin S3'ten istatistiksel olarak üstün olduğu iddiası
   yazılmamalıdır. Makale her ikisini de 278/288 olarak raporluyor (McNemar p = 1.000).
   Paket bu iddiayı hiçbir yerde tekrarlamıyor, aksine `docs/RELEASE_CHECKLIST.md`
   ve `docs/CLAIMS.md` içinde açıkça yasaklıyor.
4. **Kapatılması gereken bloke edici konular:** yazılım lisansının seçilmemiş olması
   (B2), veri seti içerik telifinin orijinal yazarlarda kalması (B3), RPS atıf/lisans
   belirsizliği (B4), figür üretim kodunun olmaması (B5) ve eksik kolların
   yayınlanmaması (B1).

Yani: **A2 + veri/manifest katmanı için GO**, **makalenin Tablo 9 ve Tablo 11'inin
tamamı için NO-GO**.

---

## 1. Scope and method

The source project was scanned **read-only**. Nothing in it was written, moved,
trained or re-inferred. For every candidate artifact the audit recorded the source
path, byte size and SHA-256 before and after copying
(`provenance/SOURCE_HASHES.csv`), then verified that 31 of 33 copies are
byte-identical and that the 2 modifications are exactly the two documented
sanitisations (`CHANGELOG_STAGING.md` §B).

The 846 image files were hashed (read-only) to build the inventory. The 288 held-out
images were not opened for evaluation; no inference was run.

## 2. Arm-by-arm finding

| Arm | Manuscript meaning | Code found? | Result artifact found? | Status in this package |
|---|---|---|---|---|
| **A2** | shared-weight dual-view DeiT3, cross-fitted crop selector | **yes** — `src/experiment_v34.py`, `src/experiment_v33.py`, `src/validate_v36.py`, `src/devcv.py` | **yes** — `results/dev/oof_v34.csv` (558), `results/test/test_predictions_v36.csv` (288), full lock/report set | **complete and verified** |
| A1 | independent global-only DeiT3-Base | no matched-baseline trainer; only an unrelated 613/72/315-era run and an AERS dev pilot | no | **not released, not verifiable** |
| A3 | ResNet-34 + RPS | no RPS code anywhere in the tree | no | **not released, not verifiable** |
| A4 | ResNet-34 without RPS | no ResNet-34 trainer | no (ResNet-50 runs exist for a different split) | **not released, not verifiable** |
| S1 | fixed 72 % centre crop, 15 folds + 3 refits | no | no (only a dev-only frozen-feature probe, 470/558) | **not released, not verifiable** |
| S2 | fixed 90 % centre crop | no | no (same probe: 479/558) | **not released, not verifiable** |
| S3 | deterministic random-candidate crop | no | no | **not released, not verifiable** |

Evidence and search terms: `docs/MISSING_ARTIFACTS.md`.

## 3. Package contents

| Category | Files | Notes |
|---|---|---|
| code (`src/`) | 7 | byte-identical copies, original module basenames kept so imports still work |
| project's own tests (`tests/source/`) | 3 | byte-identical, 15 tests, all pass |
| scripts (`scripts/`) | 3 | new: smoke test, content scan, release-manifest generator |
| manifests (`manifests/`) | 10 | 4 byte-identical inputs, 3 computed inventories, 3 new documentation/provenance files |
| results (`results/`) | 17 | dev OOF + test locks + 3 external cohorts + checkpoint hashes |
| docs (`docs/`) | 10 | 1 original (METHOD_FREEZE.md), 9 new |
| configs | 2 | new, derived from code constants — no invented values |
| provenance | 2 | source→release hashes, whole-package hash manifest |
| top level | 8 | README, CITATION, requirements, .gitignore, notices, changelog, audit |

Full list with hashes: `provenance/RELEASE_FILES.sha256` (64 entries + itself).

### Excluded (with reasons)

Raw images (copyright + size), `final_seed*.pt` (3 × 343 MB), the three feature
caches, `selector_full_DEV.joblib`, `pred_seed*_fold*.npz`, external per-image
predictions (contain a `patient_id` column), the Turkish READMEs that embed absolute
local paths, `model_weights.torch` (738 MB, unrelated manuscript), the NSCLC clinical
CSV, and all `__pycache__`/`.pyc`. Table in `CHANGELOG_STAGING.md` §C and
`docs/PROVENANCE.md` §4.

## 4. Hash and provenance verification

| Check | Result |
|---|---|
| 33 source-derived files re-hashed against `SOURCE_HASHES.csv` | **pass** (31 unchanged, 2 documented modifications) |
| 64 files checked against `provenance/RELEASE_FILES.sha256`, 0 unlisted | **pass** |
| `test_manifest_sha256` = `00f29024…b3d3` recomputed from `manifests/test_manifest_288.csv` | **pass** (exact) |
| feature fingerprint = `5763e693…8946` recomputed from manifests + published array digest | **pass** (exact) |
| CV plan signature = `522710c7…dae2` | **pass**, identical in plan, all lock files and the hard-coded constant in the training code |
| three checkpoint SHA-256 values agree across `holdout_test_lock.json`, `refit_complete.json`, `test_report_v36.json` | **pass** |
| checkpoint SHA-256 values re-computed from the on-disk weights (not redistributed) | **pass** — all three match |

## 5. Licence audit — summary

| Item | Status |
|---|---|
| authors' code | **BLOCKER** — no licence file anywhere; default copyright |
| image dataset | **ACTION REQUIRED** — provider licence keeps contents © original authors; no pixels redistributed, attribution retained |
| DeiT3 weights / timm | **CLEAR** — Apache-2.0, notices included |
| fine-tuned checkpoints | **CLEAR as released** — not distributed, hashes only |
| RPS (A3) | **BLOCKER** if A3 is ever published — no code, no commit, no licence text |
| TCIA external cohorts | **CLEAR as released** — aggregate metrics only |
| dependency libraries | **CLEAR** |

Full table: `docs/LICENSING.md`. Attribution text: `THIRD_PARTY_NOTICES.md`.

## 6. Claim verification

`manifests/claims.json` holds 31 claims; `scripts/smoke_check.py` re-evaluates them
from files inside the package (`docs/CLAIMS.md` is generated from the same source).

```
PASS 24   FAIL 0   UNVERIFIABLE/PARTIAL 7
```

Verified (selection): 846 / 558 (493 + 65) / 288 images, 508 groups, 5 folds,
0 group leakage, 0 dev–test content overlap, class counts 120/51/27/90, A2 test
278/288 with macro-F1 0.9683702 and balanced accuracy 0.9692130, branch results
274 and 275, A2 OOF 539/558 with macro-F1 0.9453955, branches 536 and 527, the two
provenance locks, the plan signature, and the two external patient-level AUROCs
(0.597 and 0.531).

Unverifiable: A1 272/288, A3 263/288, A4 256/288, S1 276/288, S2 273/288, S3 278/288
and therefore the A2-vs-S3 comparison.

## 7. Secret / personal-data scan — the three "0" statements

| Statement | Evidence |
|---|---|
| **0 raw images** | the package contains no `.png/.jpg/.jpeg/.bmp/.webp/.dcm/.nii` file; `scripts/smoke_check.py` rejects those extensions as a hard failure. Only filenames, byte sizes and SHA-256 digests of the images are published. |
| **0 secrets** | `scripts/report_scan.py` searched every text file for private-key blocks, AWS/GitHub/OpenAI/Google/Slack tokens, JWTs, `password=`/`secret=`/`token=` assignments and `.env` content: **0 blocking findings**. No `.env`, `.pem`, `.key` or credential file exists. |
| **0 personal absolute paths** | the same scan searched for `X:\…` drive-letter paths, UNC paths and POSIX home paths: **0 findings**. The two artifacts that originally carried a path (`test_predictions_v36.csv` and `audit_v36.json`) were sanitised and the change is documented and hashed. Manifest paths are dataset-relative (`train/…`, `valid/…`, `test/…`). |

Manual review additionally checked: `manifests/dev_image_list_v3.csv` has an empty
`manual_comment` column for all 558 rows; the external per-image CSVs (which do
contain TCIA patient IDs) were excluded; `dev_folds.csv` and `images_846.csv` carry
no free-text fields; no file name contains a patient name, accession number or date.

One warning is retained deliberately: `CITATION.cff` contains the corresponding
author's institutional e-mail address, as printed in the manuscript. It is public
academic contact information, not a credential.

## 8. Consistency with the manuscript

| Manuscript statement | This package |
|---|---|
| 846 curated images; 558 development (493 + 65); 288 test | **consistent** — recomputed |
| 508 dHash similarity groups; 5 group-disjoint folds | **consistent** — recomputed, zero leakage |
| A2 OOF 96.59 % (539/558) | **consistent** — recomputed |
| A2 test 96.53 % (278/288), macro-F1 0.9684, balanced acc 0.9692 | **consistent** — recomputed, and equal to the pre-inference lock |
| A2 test branches: global 274, region 275 | **consistent** — recomputed |
| 16 near-duplicate test flags, 0 exact duplicates | **consistent** — `holdout_overlap_audit.json` |
| external ADC/SCC patient-level AUROC 0.597 and 0.531 | **consistent** — aggregate reports |
| external cohorts have no NOR/LCC; not four-class validation | **consistent** — stated in `results/external_dicom/README.md` |
| A1 94.44 % (272/288), A3 91.32 % (263/288), A4 88.89 % (256/288) | **not verifiable** — no artifacts |
| S1 276/288, S2 273/288, S3 278/288; S1/S2/S3 OOF 531/535/535 | **not verifiable** — no artifacts |
| "A2 and S3 both reach 278/288; differences within statistical uncertainty" | **cannot be checked** (S3 absent); the package therefore forbids any A2 > S3 wording |
| figures 1–8 | **not reproducible** — no figure code |
| paired McNemar/Holm statistics over eight comparisons | **not reproducible** — no statistics script and no A1/A3/A4/S1/S2/S3 predictions |

Two internal inconsistencies in the source artifacts are worth recording:

1. `plan.json["features_sha256"]` (`05ef77…`) is the SHA-256 of `dev_features.npz`,
   not of the float feature array; the fingerprint uses the array digest
   (`dfb001…`). Both are now documented (`docs/PROVENANCE.md` §6.4).
2. `manifests/dev_image_list_v3.csv` (the training input) carries legacy review
   columns `v3_known_error`, `previous_status`, `review_priority` and
   `V3_error_rank`. They encode knowledge of which images an earlier model got
   wrong. The released code reads **only** `relative_path` and `class_folder`, so
   they do not enter training, but they must never be used as features or for model
   selection. They were kept because they are part of the original input file and
   removing them would have changed the input hash.

## 9. Tests executed

| Test | Command | Result |
|---|---|---|
| package smoke test | `python scripts/smoke_check.py` | **PASS** — 24 claims pass, 0 fail, 7 unverifiable; all integrity blocks OK |
| content / secret scan | `python scripts/report_scan.py` | **CLEAN** — 0 blocking findings, 1 deliberate e-mail warning |
| original unit tests | `cd tests/source && PYTHONPATH=../../src python -m unittest -v test_v34 test_v33 test_v36` | **15 tests, OK** |
| byte compilation | `python -m py_compile` on 13 modules | exit 0 |

Environment used for the tests: Python 3.11.15, numpy 1.26.4, scikit-learn 1.9.1,
scipy 1.10.0, Pillow 13.0.0, timm 1.0.30, torch 2.11.0+cu128. This is the interpreter
found on the staging machine, **not** a proven reconstruction of the training
environment (`docs/ENVIRONMENT.md`).

## 10. Blocking issues before a public "full reproduction" release

| id | Issue | Severity |
|---|---|---|
| B1 | A1/A3/A4/S1/S2/S3 artifacts absent; the manuscript's Table 9/Table 11 cannot be reproduced | **blocker for full scope** |
| B2 | no software licence | **blocker** |
| B3 | dataset content copyright remains with the original authors | action required |
| B4 | RPS attribution/licence unauditable | blocker if A3 is published |
| B5 | no figure-generation code | blocker for figure reproduction |
| M1 | independent third-party re-derivation of 278/288 and 539/558 not yet done | must-do |
| M2 | exact dependency versions never recorded | must-do |
| M3 | checkpoint hosting/download path undocumented | must-do |
| M4 | no CLI helper to rebuild the manifest digest | must-do (cosmetic; the recipe is documented) |
| M5 | exploratory status of the 288-image test must be in the repository description | must-do |
| M6 | final GitHub URL not added to the manuscript | must-do |

## 11. Decision

* **GO** for publishing this package as a **scope-limited A2 + data/manifest**
  release, provided B2 (licence) is resolved and the README's scope statement is
  kept verbatim.
* **NO-GO** for describing the repository as a full reproduction of the manuscript,
  and **NO-GO** for any statement that A2 is superior to S3 or that the 288-image
  test is an independent clinical validation.
* The package must not be pushed anywhere automatically; it is staged only.

## 12. Declaration

No training, refit or inference was performed. No file in the source project was
modified. No repository was created and nothing was pushed. The archive
`lung-ct-dualview-deit3-release.zip` was produced from this folder and verified to
contain 0 raw images, 0 checkpoints, 0 caches, 0 secrets and 0 personal absolute
paths.

---

### Ek: Türkçe kontrol listesi (kısa)

* [x] 846 görüntü, dataset-relative manifest + SHA-256
* [x] 558/288 bölünme, 508 grup, 5 fold, sınıf sırası
* [x] A2 OOF ve kilitli test tahminleri (dosya bazlı yeniden hesaplandı)
* [x] Kilit hash'leri (manifest, parmak izi, checkpoint) doğrulandı
* [x] Dış DICOM raporları ayrı klasörde, dört sınıflı doğrulama olarak sunulmadı
* [x] 0 ham görüntü / 0 sır / 0 kişisel mutlak yol (tarama + manuel inceleme)
* [ ] A1/A3/A4/S1/S2/S3 — kaynakta yok, doğrulanamaz (B1)
* [ ] Lisans seçilmedi (B2)
* [ ] Figür kodları yok (B5)
* [ ] RPS atıf/lisans (B4)
