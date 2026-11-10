# Model checkpoints — hashes only

The three final models are **not redistributed** (343 MB each, and they are
derivative works of the Apache-2.0 DeiT3 weights). The SHA-256 values below were
recorded in `results/test/holdout_test_lock.json` **before** test inference and
were re-computed from the on-disk files at staging time; the two sets agree
exactly, so the hash list is a valid identity record even though the weights
themselves are absent.

| Seed | File (as produced by `src/validate_v36.py --stage refit`) | Bytes | SHA-256 |
|---|---|---|---|
| 42 | `runs/adaptive_region_v36_test_locked/final_seed42.pt` | 343332431 | `e5c57946fb60d858543f43125b38f04abe6852875734d351991ec31107b73640` |
| 52 | `runs/adaptive_region_v36_test_locked/final_seed52.pt` | 343332431 | `84d0cecd00e8c8d2f49f3b95830fa5531eec9fa997d4515c921e17b782e7d3b7` |
| 62 | `runs/adaptive_region_v36_test_locked/final_seed62.pt` | 343332431 | `0bd601f7a7fb7a139e8c3e09d0f5eb059fc53c3bdf554048c97536d91c80de4e` |

The same three hashes appear in:

* `results/test/holdout_test_lock.json` → `weights_sha256`
* `results/test/refit_complete.json` → `model_weights`
* `results/test/test_report_v36.json` → `freeze.weights_sha256`

The smoke test asserts that all three files agree with each other.

## How to regenerate the checkpoints

The weights are deterministic only up to GPU non-determinism and library version,
so regenerating them will **not** reproduce these hashes bit-for-bit unless the
exact environment is restored (see `docs/PROVENANCE.md` §6.1). What *is*
reproducible is the procedure:

```bash
# 1. environment
python -m pip install -r requirements.txt

# 2. the single locked test inference must NOT run yet; refit first
python src/validate_v36.py --stage audit --root <PROJECT_ROOT>
python src/validate_v36.py --stage refit --root <PROJECT_ROOT> --batch 2 --accum 4

# 3. only if you intend to re-evaluate the test set (this is a one-shot gate)
python src/validate_v36.py --stage test  --root <PROJECT_ROOT> --batch 2
```

`--stage refit` writes `resume_final_seedXX.pt` after every epoch so an interrupted
run resumes; on completion it produces `final_seed42.pt`, `final_seed52.pt`,
`final_seed62.pt`. The script refuses to overwrite an existing
`test_report_v36.json`, by design.

## What the checkpoints contain

`torch.load(..., map_location="cpu")` of each file gives a checkpoint with the
DeiT3-Base state dict for a 4-class head plus the refit metadata
(`plan_signature`, `feature_fingerprint`, seed, epoch count). They are **not**
pickled sklearn objects and they contain no patient data.

## Verification snippet

```bash
python - <<'PY'
import hashlib, pathlib
expected = {
 "final_seed42.pt": "e5c57946fb60d858543f43125b38f04abe6852875734d351991ec31107b73640",
 "final_seed52.pt": "84d0cecd00e8c8d2f49f3b95830fa5531eec9fa997d4515c921e17b782e7d3b7",
 "final_seed62.pt": "0bd601f7a7fb7a139e8c3e09d0f5eb059fc53c3bdf554048c97536d91c80de4e",
}
d = pathlib.Path("runs/adaptive_region_v36_test_locked")
for name, want in expected.items():
    p = d / name
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
    print(f"{name:18s} {'OK' if got == want else 'MISMATCH'}  {got}")
PY
```
