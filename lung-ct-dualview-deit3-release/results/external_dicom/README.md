# External DICOM experiments — a separate analysis unit

These three reports were produced by `src/validate_v36.py --stage external`
with the frozen three-seed final models. They are kept in their own directory on
purpose: they answer a different question from the 846-image benchmark and must
never be pooled with it, quoted as a four-class result, or presented as external
validation of the proposed system.

## What was measured

| Cohort | Images | Patients | ADC/SCC conditional, patient level | AUROC | Source file |
|---|---|---|---|---|---|
| Lung-PET-CT-Dx | 1211 | 251 | accuracy 0.713, balanced acc 0.515, macro-F1 0.513 | **0.597** | `Lung-PET-CT-Dx/report_external.json` |
| NSCLC-Radiomics | 994 | 201 | accuracy 0.244, balanced acc 0.480, macro-F1 0.196 | **0.517** | `NSCLC-Radiomics/report_external.json` |
| NSCLC-Radiomics-FullField-DiagV314 | 994 | 201 | accuracy 0.368, balanced acc 0.525, macro-F1 0.366 | **0.531** | `NSCLC-Radiomics-FullField-DiagV314/report_external.json` |

Each report also carries the image-level ADC/SCC statistics and the accuracy of
the four-class model's argmax restricted to the ADC/SCC subset (0.011, 0.225 and
0.035 respectively) — that last number exists to show how badly a four-class head
transfers, not as a result.

## Why these numbers are not a success story

* The cohorts contain **only ADC and SCC** images. There is no NORMAL and no LCC
  class, so a four-class accuracy is not defined for them.
* The primary unit is the **patient**, because many slices come from one subject.
* The imaging chain differs: full DICOM chest CT slices, converted to PNG/JPG with
  a frozen window/slice-selection protocol, versus the provider's small exported
  PNG/JPG crops used in training.
* Discrimination is weak: AUROC 0.597 / 0.517 / 0.531 for ADC versus SCC.
  Balanced accuracy near 0.5 means close to chance on the patient level.
* The two NSCLC-Radiomics variants differ only by the input-intensity handling
  (native vs full-field diagnostic v3.1.4 preparation); the pair is reported to
  show how sensitive the transfer result is to that choice, **not** to pick the
  better one.

The manuscript's limitation section (and Supplementary Section S2) states the same;
`docs/PROTOCOL.md` §6 repeats it so that the code and the paper agree.

## What is deliberately excluded

The source project also contains
`external_image_predictions.csv` for each cohort. Those rows carry a
`patient_id` column (a TCIA subject identifier). They are **not** released: see
`docs/MISSING_ARTIFACTS.md` M6. Only the aggregate reports and the protocol locks
are published.

## Provenance

`external_protocol_lock.json` in each folder was written before the corresponding
evaluation and records the cohort name, the frozen method string
(`0.85global+0.15region`), the frozen plan signature, the external manifest hash
and the image/patient counts. The manifest files themselves are external-cohort
derived and are not redistributed.
