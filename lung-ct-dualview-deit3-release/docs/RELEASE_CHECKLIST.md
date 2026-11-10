# Before this repository is made public

Status as of staging: **11 blockers/must-dos open, 9 items done.** The package is a
verified *partial* release; it is not yet publish-ready as a full reproduction of
the manuscript. See `release_audit.md` for the GO/NO-GO decision.

## Done at staging

- [x] Source project scanned read-only; every released file has its source path,
      source SHA-256 and release SHA-256 in `provenance/SOURCE_HASHES.csv`.
- [x] Absolute local paths removed from the two released artifacts that contained
      them (`results/test/test_predictions_v36.csv`, `results/test/audit_v36.json`);
      changes listed in `CHANGELOG_STAGING.md`.
- [x] No tokens, passwords, cookies, `.env`, SSH keys or credentials in the package
      (`docs/SECURITY_SCAN.md`, 0 blocking findings).
- [x] No raw images, no checkpoint binaries, no NPZ caches, no joblib blobs; only
      digests are published.
- [x] Manifest paths are dataset-relative and contain no account names.
- [x] Every *released* result is recomputed by `python scripts/smoke_check.py`
      from the released prediction files.
- [x] Both the learned selector and the fixed/random crop-policy comparison are
      discussed; the missing ones are named as missing rather than omitted silently.
- [x] The original 613/72/315 split, duplicate records and external DICOM transfer
      are described separately and accurately.
- [x] README no longer says "pending"; installation and the one-command smoke test work.

## Open blockers — must be resolved before public release

- [ ] **B1. Missing arms A1/A3/A4 and S1/S2/S3.** Table 9 and Table 11 of the
      manuscript cannot be reproduced from the staged source project. Either deposit
      the OOF/test predictions, configs and code for those arms, or amend the
      manuscript/repository to state that only A2 and the data/manifest layer are
      released. See `docs/MISSING_ARTIFACTS.md`.
- [ ] **B2. No software licence.** Choose a licence (e.g. MIT or Apache-2.0) and add
      `LICENSE`; until then the repository is "all rights reserved" by default and
      must not be called open source.
- [ ] **B3. Dataset content copyright.** The provider licence keeps contents
      © original authors. Confirm that publishing filenames + SHA-256 (no pixels) is
      acceptable, and keep the attribution block from `THIRD_PARTY_NOTICES.md`.
- [ ] **B4. RPS attribution and licence.** The manuscript attributes arm A3's
      augmentation to Abe and Nyathi (reference [25]) but no RPS code, commit hash or
      licence exists in the source tree. If A3 is ever added, pin the exact commit and
      reproduce the upstream notice.
- [ ] **B5. Figures.** No figure-generation script exists in the source tree; the
      manuscript's figures cannot be regenerated from this package. Either add the
      scripts or state that figures are not reproducible.

## Open must-dos — should be resolved before public release

- [ ] **M1. Re-verify the A2 headline numbers independently.** The recomputation in
      this package is self-consistent (predictions → reports). An independent party
      should re-derive 278/288 and 539/558 from the released CSVs before the
      manuscript's Code Availability statement points here.
- [ ] **M2. Record the exact dependency versions.** The source project pinned package
      names only. Publish an `environment.lock.txt` from the environment that produced
      the numbers, or note that bit-exact re-training is not guaranteed.
- [ ] **M3. Publish or document the checkpoint download path.** `final_seed42/52/62.pt`
      are 343 MB each. Either host them with documented SHA-256 (already recorded) or
      publish the exact command to regenerate them.
- [ ] **M4. Add a `--stage fingerprint` self-check** so that a third party can rebuild
      the 288-image manifest digest without reading this documentation.
- [ ] **M5. State the exploratory status in the repository description.** The 288-image
      test directory is public and was used in earlier project stages; the repository
      must not imply a prospective untouched-data validation.
- [ ] **M6. Add the final GitHub URL** to the manuscript's Data and Code Availability
      section once the repository exists (the URL is intentionally absent from
      `CITATION.cff` at staging time).

## Wording that must never appear in this repository

Do not write, in the README, the release notes or any doc:

- that A2 is statistically superior to S3 — the manuscript reports **both at 278/288**
  with an exact McNemar p = 1.000;
- that the 288-image test is an independent hospital or patient-level validation;
- that the crop selector localises lesions or performs segmentation;
- that dHash grouping proves patient independence;
- that the external DICOM results are a successful four-class external validation
  (ADC/SCC patient-level AUROC is 0.597, 0.531 and 0.517 across the three cohorts,
  and those cohorts contain no NOR and no LCC class);
- that 96.59 % OOF is an independent test result (it is a development OOF number).
