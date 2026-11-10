# Security, secret and personal-data scan

Tool: `scripts/report_scan.py` (in this package). It is deterministic, offline and
read-only; it prints every match with file and line, and exits non-zero on a
blocking finding.

```bash
python scripts/report_scan.py          # human readable
python scripts/report_scan.py --json   # machine readable
```

## 1. Result

```
scanned 67 files
binary/office files present: none
blocking findings: 0
warnings: 1
  [warn ] EMAIL   CITATION.cff:21  'skeser@ahievran.edu.tr'
RESULT: CLEAN (0 blocking findings)
```

This file and `scripts/report_scan.py` are the two files that legitimately contain
the search patterns themselves; the scanner exempts them from their own examples
(`DOC_FILES` in the script) and still applies every other rule to them.

The single warning is intentional: the corresponding author's institutional e-mail
address, copied from the manuscript, where it is already public. It is not a
credential and not a personal address of a data subject.

## 2. Rules applied

| Rule | Severity | What it catches |
|---|---|---|
| `ABS_WIN_DRIVE` | block | `X:\…\…` drive-letter absolute paths |
| `ABS_UNC` | block | `\\server\share` network paths |
| `ABS_HOME` | block | `/home/<user>`, `/Users/<user>` |
| `USERNAME` | block | the local account name in a path context |
| `PRIVATE_KEY` | block | `-----BEGIN … PRIVATE KEY-----` blocks |
| `AWS_KEY`, `GITHUB_TOKEN`, `OPENAI_KEY`, `GOOGLE_KEY`, `SLACK_TOKEN`, `JWT` | block | credential-looking tokens |
| `PASSWORD_ASSIGN` | block | `password = "…"`, `secret: "…"`, `api_key = "…"` |
| `PATIENT_ID` | block | TCIA/DICOM identifiers such as `Lung_Dx-A0001`, `LUNG1-001` |
| `PRIVATE_DIR` | warn | file names containing `PRIVATE_` / `DO_NOT_UPLOAD` |
| `EMAIL` | warn | e-mail addresses |
| binary/office extension inside the package | block (via smoke test) | `.pt .npz .pkl .joblib .png .jpg .dcm .nii .gz .zip .docx .pdf …` |

`scripts/smoke_check.py` additionally fails the package if **any** file with those
extensions exists, if a released file's hash differs from
`provenance/RELEASE_FILES.sha256`, or if any file in the folder is not listed there.

## 3. Findings that were fixed during staging

| Finding | Where | Fix |
|---|---|---|
| 288 absolute Windows paths | `results/test/test_predictions_v36.csv` (`image_path` column) | column removed; row order and all values preserved (`CHANGELOG_STAGING.md` §B1) |
| 4 absolute Windows paths | `results/test/audit_v36.json` (`v34_out`, three `final_seed_weights`) | replaced by repository-relative paths (§B2) |
| TCIA patient identifiers | `results/external_dicom/*/external_image_predictions.csv` | files **excluded**; only aggregate reports published |
| clinical table with cohort metadata | `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv` (source project root) | **excluded** |
| absolute paths inside run instructions | the three Turkish `README_TR.md` files | **excluded**; replaced by `docs/PROTOCOL.md` |
| compiled caches | `__pycache__/*.pyc` created by `py_compile` | deleted before the release manifest was generated |

## 4. Manual review performed in addition to the automated scan

* All 558 rows of `manifests/dev_image_list_v3.csv`: the free-text column
  `manual_comment` is empty in every row; `manual_anatomy` and `manual_mask_ok` are
  the literal string `UNREVIEWED` in every row. No clinician notes, no names.
* All 846 filenames: provider export names only (`000119 (6).png`, `ad13.png`,
  `n6 (2) - Copy.jpg`, `003828_02_01_174.png`). No patient names, no accession
  numbers, no dates.
* All 23 JSON files: reviewed for `patient_id`, `subject`, `name`, `email`, `path`
  keys. The three external reports contain only cohort-level aggregates and counts.
* `CITATION.cff`: only the public author metadata from the manuscript.
* `docs/*.md`, `README.md`, `release_audit.md`: read for accidental inclusion of a
  local path, a credential, or a claim the package cannot support. None found after
  the fixes in §3.

## 5. What is still not covered

* The scan is pattern based. A human reviewer must re-read the package before every
  public release; `docs/RELEASE_CHECKLIST.md` lists the wording that must never
  appear.
* The excluded per-image external CSVs still exist in the source project. If they are
  ever published, the `patient_id` column must be replaced by a study pseudonym and
  the mapping kept private.
* The scan cannot prove that a dataset filename is not identifying in combination
  with external knowledge; `docs/DATA.md` §7 states the limit.
