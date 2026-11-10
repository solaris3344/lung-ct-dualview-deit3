# Claim table

Every row is generated from `manifests/claims.json` and re-evaluated by
`scripts/smoke_check.py`. Run that script to reproduce the table; do not edit
this file by hand.

Summary of the current run: **24 PASS, 0 FAIL, 7 UNVERIFIABLE/PARTIAL**.

`PASS` means the value was recomputed from the file named in the *computed from*
column and matched the stated value (exact for counts, 1e-12 for full-precision floats).

| id | arm | claim | stated | recomputed | status | computed from | manuscript |
|---|---|---|---|---|---|---|---|
| C01 | DATA | Curated exact-cleaned inventory contains 846 unique image files with 846 distinct SHA-256 contents | 846 | 846 | **PASS** | `manifests/images_846.csv` | Sec. 2.1 |
| C02 | DATA | Development partition has 558 images (493 original train + 65 original valid) | 558 | 558 | **PASS** | `manifests/images_846.csv` | Sec. 2.1, Table 1 |
| C03 | DATA | Held-out same-source test has 288 images | 288 | 288 | **PASS** | `manifests/test_manifest_288.csv` | Sec. 2.1, Table 1 |
| C04 | DATA | Development images fall into 508 dHash similarity groups | 508 | 508 | **PASS** | `manifests/dev_folds.csv` | Sec. 2.2 |
| C05 | DATA | No dHash group crosses the train/validation side of any of the 5 folds | 0 | 0 | **PASS** | `manifests/dev_folds.csv` | Sec. 2.2 |
| C06 | DATA | No exact-duplicate file content is shared between development and the 288-image test | 0 | 0 | **PASS** | `manifests/images_846.csv` | Sec. 2.2 |
| C07 | DATA | Test class counts are ADC 120 / LCC 51 / NOR 27 / SCC 90 | [120, 51, 27, 90] | [120, 51, 27, 90] | **PASS** | `manifests/test_manifest_288.csv` | Table 1 |
| C08 | A2 | A2 (0.85 global + 0.15 selected region) is correct on 278/288 test images | 278 | 278 | **PASS** | `results/test/test_predictions_v36.csv` | Table 9, Abstract |
| C09 | A2 | A2 test top-1 accuracy is 0.9652777... (278/288) | 0.9652777778 | 0.9652777778 | **PASS** | `results/test/test_predictions_v36.csv` | Table 9 |
| C10 | A2 | A2 test macro-F1 is 0.9683702... | 0.968370243 | 0.968370243 | **PASS** | `results/test/test_predictions_v36.csv` | Table 9 |
| C11 | A2 | A2 test balanced accuracy is 0.9692129... | 0.969212963 | 0.969212963 | **PASS** | `results/test/test_predictions_v36.csv` | Table 9 |
| C12 | A2 | A2 global-only branch is correct on 274/288 test images | 274 | 274 | **PASS** | `results/test/test_predictions_v36.csv` | results/test/test_report_v36.json (methods.global) |
| C13 | A2 | A2 region-only branch is correct on 275/288 test images | 275 | 275 | **PASS** | `results/test/test_predictions_v36.csv` | results/test/test_report_v36.json (methods.region) |
| C14 | A2 | A2 test confusion matrix (true rows ADC/LCC/NOR/SCC) is [[115,4,0,1],[0,51,0,0],[1,0,26,0],[4,0,0,86]] | [115,4,0,1] / [0,51,0,0] / [1,0,26,0] / [4,0,0,86] | [115,4,0,1] / [0,51,0,0] / [1,0,26,0] / [4,0,0,86] | **PASS** | `results/test/test_predictions_v36.csv` | Table 9 class results |
| C15 | A2 | A2 pooled development OOF is 539/558 correct | 539 | 539 | **PASS** | `results/dev/oof_v34.csv` | Table 9 |
| C16 | A2 | A2 pooled development OOF accuracy is 0.96594982... | 0.9659498208 | 0.9659498208 | **PASS** | `results/dev/oof_v34.csv` | Table 9 |
| C17 | A2 | A2 pooled development OOF macro-F1 is 0.9453955197... | 0.9453955198 | 0.9453955198 | **PASS** | `results/dev/oof_v34.csv` | results/dev/report_v34.json |
| C18 | A2 | A2 OOF global-only branch is correct on 536/558 development images | 536 | 536 | **PASS** | `results/dev/oof_v34.csv` | results/dev/report_v34.json |
| C19 | A2 | A2 OOF region-only branch is correct on 527/558 development images | 527 | 527 | **PASS** | `results/dev/oof_v34.csv` | results/dev/report_v34.json |
| C20 | LOCK | Recorded test-manifest digest 00f29024... is reproduced from the released 288-row manifest | 00f29024bae662c25a73064603b277ad8ad61c3bcc5ce65659ba38b65345b3d3 | 00f29024bae662c25a73064603b277ad8ad61c3bcc5ce65659ba38b65345b3d3 | **PASS** | `manifests/test_manifest_288.csv vs results/test/holdout_test_lock.json` | Sec. 2.11 reproducibility |
| C21 | LOCK | Recorded frozen-feature fingerprint 5763e693... is reproduced from the released manifests plus the published feature-array digest | 5763e693aaf61c928b829cfa2926218c02917358b4e6cdf2ba5673f28b988946 | 5763e693aaf61c928b829cfa2926218c02917358b4e6cdf2ba5673f28b988946 | **PASS** | `manifests/dev_image_list_v3.csv + dev_groups.csv + feature_cache_provenance.json` | METHOD_FREEZE.md |
| C22 | LOCK | Recorded CV-plan signature 522710c7... is reproduced by the grouping/fold manifests | 522710c76a60df73d6776ae8a05e75d46c943dbc006215bf81d045a49ed3dae2 | 522710c76a60df73d6776ae8a05e75d46c943dbc006215bf81d045a49ed3dae2 | **PASS** | `manifests/dev_plan_v3.json` | METHOD_FREEZE.md |
| C23 | EXT | External Lung-PET-CT-Dx patient-level ADC/SCC AUROC is 0.597 (weak transfer) | 0.5973867268 | 0.5973867268 | **PASS** | `results/external_dicom/Lung-PET-CT-Dx/report_external.json` | Sec. 4 limitations / Supplementary S2 |
| C24 | EXT | External NSCLC-Radiomics full-field patient-level ADC/SCC AUROC is 0.531 (weak transfer) | 0.5311111111 | 0.5311111111 | **PASS** | `results/external_dicom/NSCLC-Radiomics-FullField-DiagV314/report_external.json` | Sec. 4 limitations / Supplementary S2 |
| C25 | A1 | Independent global-only DeiT3-Base (A1) test accuracy 94.44% = 272/288 | 272 | — | **UNVERIFIABLE** | `artifact not in release` | Table 9 |
| C26 | A3 | ResNet-34 + Random Pixel Swap (A3) test accuracy 91.32% = 263/288 | 263 | — | **UNVERIFIABLE** | `artifact not in release` | Table 9 |
| C27 | A4 | ResNet-34 without RPS (A4) test accuracy 88.89% = 256/288 | 256 | — | **UNVERIFIABLE** | `artifact not in release` | Table 9 |
| C28 | S1 | Fixed 72% centre crop (S1) OOF 0.9516 (531/558) and test 0.9583 (276/288) | [0.9516129032258065, 0.9583333333333334] | — | **UNVERIFIABLE** | `artifact not in release` | Table 11 |
| C29 | S2 | Fixed 90% centre crop (S2) OOF 0.9588 (535/558) and test 0.9479 (273/288) | [0.9587813620071685, 0.9479166666666666] | — | **UNVERIFIABLE** | `artifact not in release` | Table 11 |
| C30 | S3 | Deterministic random-candidate crop (S3) OOF 0.9588 (535/558) and test 0.9653 (278/288) | [0.9587813620071685, 0.9652777777777778] | — | **UNVERIFIABLE** | `artifact not in release` | Table 11 |
| C31 | A2-vs-S3 | A2 versus S3 test difference is 0/288 images; both are 278/288 and the paired test is non-significant | 0 | — | **PARTIAL** | `artifact not in release` | Table 11 / Sec. 3.7 |

## Reading the UNVERIFIABLE rows

`UNVERIFIABLE` is a statement about this package, not about the manuscript. It means
that no prediction file, report, config or code for that arm exists in the source
project this package was staged from, so the number can be recomputed from nothing —
it can only be quoted. Quoting an unverifiable number in a README, a repository
description or a release note would turn this package into a claim it cannot support.

`PARTIAL` (C31) is the A2-versus-S3 comparison: the A2 side is verified, the S3 side is
absent, so the comparison cannot be evaluated. The manuscript itself states both arms at
278/288 with exact McNemar p = 1.000; no superiority claim is admissible.

## What is fully verified

* The inventory: 846 files, 846 distinct SHA-256 contents, 558 development (493 + 65),
  288 test (C01–C07).
* The partition: 508 dHash groups, five group-disjoint folds, zero development/test content
  overlap, zero group leakage (C04–C07).
* The dual-view system A2: 278/288 on the locked test with macro-F1 0.9684 and balanced
  accuracy 0.9692; 539/558 development OOF with macro-F1 0.9454 (C08–C19).
* The two provenance locks written before test inference (C20, C21) and the CV-plan
  signature (C22).
* The external DICOM ADC/SCC transfer values, as aggregate cohort statistics (C23, C24).
