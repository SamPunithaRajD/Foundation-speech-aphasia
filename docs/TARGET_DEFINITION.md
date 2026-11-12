# Target definition — WAB-R-derived severity score

Status: **IMPLEMENTED** (definition and code) · data: **NOT_AVAILABLE** · model: **MODEL_NOT_READY**

Code: `backend/app/training/target.py` (single source of truth; covered by tests in `backend/tests/test_training.py`).

## Source instrument
Western Aphasia Battery-Revised (WAB-R), **Aphasia Quotient (AQ)** (Kertesz, 2007).

| Property | Value |
|---|---|
| Range | 0–100 |
| Direction | **higher AQ = better language performance** |
| Availability | AphasiaBank Test Results collection (for many, not all, participants with aphasia; controls usually have no AQ) |

## Transformation

```
severity_score = 100 − AQ          (target_version: wabr-aq-inverted/1.0)
```

- 0 = lowest WAB-R-derived impairment score, 100 = highest (`score_direction: higher_worse`).
- This is valid because AQ is bounded at 0–100 and runs in the opposite direction; checked against the source.
- **Name:** "WAB-R-derived severity score". It is a transformation of a published instrument, **not** a newly validated clinical severity scale.
- The original `wab_aq` is always stored alongside `severity_score`, `target_source` and `target_version`.
- Predictions keep the raw model output. Clipping to [0, 100] happens only at the final output boundary, and `predicted_aq = 100 − severity_score`.

## Severity bands (derived, secondary)
The published WAB AQ severity ranges (Kertesz, 2007) are applied to AQ:

| AQ | Band | Severity score |
|---|---|---|
| 0–25 | Very severe | 75–100 |
| 26–50 | Severe | 50–74 |
| 51–75 | Moderate | 25–49 |
| ≥ 76 | Mild | ≤ 24 |

- **Decimal AQ.** The published ranges are integers. The code uses half-open intervals (`< 26`, `26 ≤ AQ < 51`, `51 ≤ AQ < 76`, `≥ 76`), which reproduces the published assignment for every integer AQ. This convention is an implementation choice.
- **No "no aphasia" band.** The WAB-R AQ aphasia cut-off is reported as 93.7 or 93.8, and it is contested for mild impairment (96.7 has been proposed). No cut-off is applied, and a low score must be read as "WAB-R-derived score low", not "no aphasia".
- **Bands are secondary.** They are derived from the same AQ and are never independent labels.

## Label handling rules
- **Missing AQ:** the record is kept, reported as `missing WAB-R AQ`, and not trainable. Never imputed.
- **Invalid AQ:** non-numeric, non-finite or outside [0, 100]. Reported as `invalid WAB-R AQ (…)` and not trainable.
- **Ambiguous AQ:** several differing rows for one participant/session. Reported and not trainable; no row is picked silently.
- **Controls without AQ:** excluded from regression. They are not assigned 0 or "no aphasia".

## References
- Kertesz, A. (2007). *Western Aphasia Battery–Revised*. Pearson.
- AQ severity ranges as cited in, e.g., PMC7997243 (Prediction of aphasia severity using DTI).
- Revisiting WAB-R in the context of mild impairment, *AJSLP* (2026), doi:10.1044/2026_AJSLP-25-00361.

## Output validity (revised 2026-10-12)
A regression head is unbounded, so its raw output is checked before anything is reported:

| Raw output | Reported as |
|---|---|
| within 0–100 | score = raw (1 decimal) |
| up to 10 points outside (−10…0 or 100…110) | clipped to the range, `clipped: true`, with a note |
| more than 10 points outside, NaN or infinite | **`INVALID_OUTPUT`**: no score, AQ or band; the raw value is kept and shown as invalid |

Separately, if the recording's inputs are far outside the training distribution, the result is
**`OUT_OF_DISTRIBUTION`**: no score is reported and the raw value is kept. The check is the mean
absolute z-score of the inputs against training statistics, with the limit at `FSA_OOD_FACTOR` (default 1.5) ×
the largest value seen in training.

An invalid output is never converted into a valid-looking score. The 10-point tolerance
(`target.OUTPUT_TOLERANCE`) is an implementation choice and is stored in each bundle.

## Other instruments (Quick Aphasia Battery)
QAB overall (0–10) is a different instrument from WAB-R AQ (0–100). They are **not interchangeable**:
- This project does not convert QAB to AQ. Doing so from a small paired sample would manufacture AQ labels.
- A QAB-based model would need its own explicitly named target (e.g. "QAB-derived severity score"), its own
  bands (none are defined here), and its own `target_version`. It could not be reported as WAB-R-derived.
