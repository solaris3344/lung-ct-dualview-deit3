# Manifests

All paths in this directory are **relative to the dataset root** (`<DATA_ROOT>`,
see `docs/DATA.md`). No absolute path, account name or machine name appears in
any file here.

| File | Rows | What it is |
|---|---|---|
| `images_846.csv` | 846 | authoritative inventory: `relative_path`, `split`, `class_folder`, `class_short`, `class_index`, `group`, `outer_fold`, `bytes`, `sha256` |
| `test_manifest_288.csv` | 288 | test subset: `relative_path`, `class_short`, `class_index`, `bytes`, `sha256`; its `(path, sha256)` digest equals the lock value `00f29024…` |
| `dev_folds.csv` | 558 | development subset with `class_short`, `group` (`G000`–`G507`) and `outer_fold` (0–4) |
| `dev_image_list_v3.csv` | 558 | the development image list actually consumed by `src/experiment_v34.py --csv`, in the exact row order used to build the frozen feature array; also carries the review metadata columns of the original anatomy review |
| `dev_groups.csv` | 558 | original `rel, split, label, group` table produced by the grouping stage (raw `group` values are used by the fingerprint recipe) |
| `dev_plan_v3.json` | 5 folds | the frozen CV plan (`heldout_rel` per fold) plus `signature`, `fingerprint`, `group_digest`, `features_sha256`, `heldout_rel` |
| `groups_meta_v3.json` | 1 | grouping audit counters: 508 groups, largest 7, 79 approved links, 0 exact-pixel edges, 0 mixed-label groups |
| `manifest_summary.json` | 1 | recomputed inventory summary and a single `manifest_digest_sha256` |
| `feature_cache_provenance.json` | 1 | digests of the excluded feature caches and the recipe that reproduces the frozen-feature fingerprint |
| `claims.json` | 31 | machine-readable claim table consumed by `scripts/smoke_check.py` |

## Image identity

The image identifier is `relative_path`, e.g. `test/adenocarcinoma/000108 (3).png`.
It is used unchanged in:

* every row of `results/test/test_predictions_v36.csv`,
* every row of `results/dev/oof_v34.csv`,
* every row of `images_846.csv` and `dev_folds.csv`.

There is no separate numeric image id and no re-indexing anywhere. `true_class`,
`*_pred` and the `*_p0…*_p3` probability columns all follow the class order in
`configs/class_order.json` (**ADC, LCC, NOR, SCC**).

## Fold semantics

`dev_plan_v3.json["heldout_rel"][k]` is the **validation** list of outer fold `k`;
the training side is its complement. Folds were assigned so that a dHash group is
never split between the two sides. `dev_folds.csv` carries the same assignment in
long form (`outer_fold`), and the smoke test re-checks coverage and leakage.

## Do not regenerate these files in place

If you rebuild the manifests (for example after a different provider archive),
write to a new directory and compare `manifest_digest_sha256`. The `signature`
value `522710c7…` is embedded in the training code as a hard-coded constant; a
plan with a different signature is rejected by `src/experiment_v34.py`, which
is the intended behaviour.
