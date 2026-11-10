# Environment observed at staging time

**Read this as information, not as a promise.** The source project never recorded the
exact versions that produced the published numbers (§ below), so the table is the
interpreter that happened to be on the staging machine — it was used to byte-compile
the code and to run the shipped unit tests, and nothing more.

## 1. Probe

```bash
python -c "import sys,torch,timm,PIL,joblib,numpy,sklearn,scipy; print(sys.version); print(torch.__version__, torch.version.cuda); print(timm.__version__)"
```

## 2. Observed values

| Item | Value |
|---|---|
| Python | 3.11.15 |
| torch | 2.11.0+cu128 (CUDA 12.8 available) |
| timm | 1.0.30 |
| torchvision | not probed separately (installed as a torch companion) |
| Pillow | 13.0.0 |
| joblib | 1.6.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.9.1 |
| scipy | 1.10.0 |

`requirements.txt` intentionally gives ranges instead of these pins, because the
staging interpreter is not proven to be the training interpreter. Publishing a
`environment.lock.txt` from the machine that produced the numbers is checklist item
M2 in `docs/RELEASE_CHECKLIST.md`.

## 3. What was actually executed at staging

| Action | Result |
|---|---|
| `python -m py_compile` on all 7 source modules, 3 scripts and 3 test modules | exit 0, no syntax errors |
| `python -m unittest test_v34 test_v33 test_v36` from `tests/source/` with `PYTHONPATH=src` | **15 tests, OK** |
| `python scripts/smoke_check.py` | 24 claims PASS, 0 FAIL, 7 unverifiable |
| `python scripts/report_scan.py` | 0 blocking findings |

The three unit-test modules are the project's own tests, unmodified. They exercise
frozen-plan loading, group-leakage rejection, the selector's inner-OOF behaviour and
cache-guard, probability validation, the crop geometry, the Wilson interval, the
bootstrap, the exact dev/test collision guard and the two-class external rule.

```bash
cd tests/source
PYTHONPATH=../../src python -m unittest -v test_v34 test_v33 test_v36
```

## 4. Known quirk on locked-down Windows sessions

`TemporaryDirectory` can raise `PermissionError: [WinError 5]` while creating or
deleting its own sub-directory when the process runs under a restricted token
(observed in the staging sandbox). It is not a defect in the code under test.
Pointing `TMPDIR`/`TEMP` at an ordinary writable folder, or granting the process
normal ACL inheritance in that folder, resolves it.

## 5. What was *not* run at staging

No training, no refit, no inference, no evaluation on the image data. The 846 image
files were only read to compute SHA-256 digests. The 288 held-out images were not
opened.
