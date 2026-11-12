# Pilot: text-only prediction of a WAB-AQ-derived severity score (QAB corpus)

**Status: `PILOT_ONLY`** — real data, real result. Audited 2026-10-13; deployed feature set changed to MATTR only on 2026-11-11 (section 2a). Not validated on Whisper transcripts. Not clinically validated.

This document replaces the earlier version, which mislabelled the instrument as WAB-R and reported a p-value that did not account for feature-set selection. Both are corrected below.

## 1. Data and labels (verified)

| Item | Value |
|---|---|
| Corpus | Quick Aphasia Battery Corpus, AphasiaBank `English/NonProtocol/QAB`, doi:10.21415/T8P2-CN50 |
| Citation (required by TalkBank) | Wilson SM, Eriksson DK, Schneck SM, Lucanie JM. *PLOS ONE* 2018; 13(2): e0192773 |
| Label file / sheet / column | `QAB-dataset.xlsx` / `WAB-dataset` / `AQ` (byte-identical to the file on TalkBank) |
| Instrument | **Western Aphasia Battery (Kertesz, 1982)** Aphasia Quotient. **Not the WAB-R**; the paper cites the 1982 edition |
| Label check | `AQ = 2 × (SpontSp + AudVbl Comp + Repetition + Naming)` holds exactly for 16/16 participants, and each section score recomputes from its item scores |
| Not a QAB score | The QAB overall score (0–10) is a separate column in a separate sheet and is not used |
| Target | `severity = 100 − WAB AQ` ("WAB-AQ-derived severity score") |
| Participants | 16, all chronic post-stroke aphasia; AQ 18.9–98.8 |
| Sessions | 3 per participant (48 transcripts), spanning a median of 21 days (maximum 51) |
| Label matching | One AQ per participant, administered at the **first** session, applied to all of that participant's sessions |
| Transcripts | Human transcripts of a sample of clinician-led conversation (transcriber code SMS) |
| Audio / video | Recorded in the study and listed on TalkBank ("Recordings: 48; media type: audio, video"), but **behind TalkBank member login**. Not available to this project yet |

**Eligible cohort:** a transcript needs at least 25 words (the vocabulary measure cannot be computed below that).
This excludes 5 of 48 transcripts and removes 1 of 16 participants entirely, leaving **15 participants and 43 transcripts**
(14 participants with 3 transcripts, 1 with 1).

## 2. Model
- **Features:** `moving_average_ttr` (vocabulary variety, 25-word window) and `clause_marker_rate` (clause markers per 100 words). Both are word-based, so they are defined the same way for a human transcript and a Whisper transcript.
- **Model:** ridge regression, 3 parameters. In practice vocabulary variety carries the prediction.
- **Fitted on:** the 15 eligible participants only (verified by refitting from the source data).

## 2a. Change on 2026-11-11: clause-marker rate removed from the deployed model
**Reason (robustness, not score):** on a real microphone recording, Whisper produced several isolated
function words with near-zero token probability (three single-word "If" segments at 0–1% probability, plus "So"/"Or").
These count as clause markers, which pushed `clause_marker_rate` to 11.4 per 100 words, outside the training range
(0–6.4), so the recording got no score. Human transcripts do not contain such hallucinated words, so this feature is
fragile on Whisper output in exactly the halting, fragmented speech the tool is meant for.

**What was deployed instead:** `mattr_only`, one of the seven candidates that were already in the audited family.
The family-wise p-value already covers choosing any member of that family, so changing the choice does not add
selection that the correction misses. The full audit was rerun (2,000 permutations) with this candidate deployed:

| Deployed model | Participant MAE | 95% CI | R² | Pearson | Spearman | Largest error | p (family-wise) |
|---|---|---|---|---|---|---|---|
| MATTR only (deployed since 2026-11-11) | **13.3** | 10.0–17.3 | 0.47 | 0.69 | 0.65 | 30.8 | **0.011** |
| MATTR + clause-marker rate (2026-10-13 to 2026-11-11) | 13.7 | 10.2–17.6 | 0.45 | 0.67 | 0.55 | 30.0 | 0.014 |

Mean baseline MAE 18.2. The fairest expected error is still the nested (selection-inclusive) **MAE 14.4**.
Bundle audit: 11/11 checks pass. Interval: ±32.7 points (largest held-out single-recording error).
Previous bundle: `backend/artifacts/archive/severity_qab_pilot_audit_20261013T063525Z`.
No personal recording was used for fitting, selection or evaluation.

**Speech rate is still not in the model.** The QAB transcripts are plain text with untimed pause marks and no
time stamps, and the audio is behind TalkBank member login, so words per minute cannot be computed for the
training participants. Speech rate and articulation rate are measured and reported for each recording but cannot
be learned from this data. With the QAB recordings, a rate-aware model could be trained and evaluated the same way.

## 3. Evaluation (corrected)

Leave-one-participant-out: all sessions of a participant are held out together, and the held-out participant takes
no part in standardisation, feature selection, α tuning or fitting. Predictions are averaged per participant.
Significance by 2,000 permutations of the participant→AQ assignment, repeating the whole procedure each time.

### Reproduction
Re-running the original script unchanged returned the originally reported numbers exactly:
MAE 13.7, mean-baseline MAE 18.2, R² 0.45, permutation p = 0.010 (500 permutations).

### What was wrong with the original report
The deployed feature set was defined **after** two other sets had been evaluated and after feature/label correlations
had been printed for all participants (vocabulary variety was visibly the strongest). The original p = 0.010 did not
account for that choice.

### Correction
All candidates are evaluated on the same 15 participants. The family is every set that was examined, the rule-derived
Whisper-compatible pool, and automatic "most-correlated feature" selectors (re-selected inside every training fold).
The family-wise p-value takes the best candidate of the family in each permutation.

| Candidate | MAE | Per-recording MAE | R² | p (unadjusted) | p (family-wise) |
|---|---|---|---|---|---|
| length-robust (7 features) | 13.5 | 12.7 | 0.39 | 0.006 | 0.013 |
| all text (10 features) | 12.8 | 12.5 | 0.45 | 0.003 | 0.006 |
| **deployed: MATTR + clause-marker rate** | **13.7** | **13.0** | **0.45** | **0.004** | **0.014** |
| MATTR only | 13.3 | 12.5 | 0.47 | 0.003 | 0.011 |
| Whisper-compatible pool (5 features) | 14.0 | 13.7 | 0.40 | 0.005 | 0.019 |
| auto top-1 of 11 (chosen per fold) | 15.4 | 14.2 | 0.22 | 0.012 | 0.042 |
| auto top-2 of 11 (chosen per fold) | 15.3 | 14.0 | 0.22 | 0.013 | 0.040 |
| **Nested: candidate and α chosen per fold** | **14.4** | 13.5 | 0.36 | 0.011 | — |

Mean baseline (predict the training participants' mean): MAE **18.2**. Under permutation the baseline averages 18.2
and every candidate averages 18.5–19.2, i.e. no candidate beats the baseline by construction, which is what a
leak-free procedure should show.

**Deployed model, participant level (n = 15):** MAE 13.7 (bootstrap 95% CI 10.2–17.6), RMSE 15.4, R² 0.45,
Pearson 0.67, Spearman 0.55, largest error 30.0 points, no out-of-range outputs.

**How to read this**
- The signal is real after correction: the deployed model's family-wise p is 0.014.
- The fairest single estimate of expected error is the **nested** one, **MAE 14.4** (95% CI 10.5–19.0), because it includes the freedom to pick the feature set.
- The deployed set is **not better** than the others on the same participants. It is the one that can be computed from a Whisper transcript.
- The earlier comparison in this document (16.9 and 17.9 for the first two sets) was on 16 participants and is not comparable; the difference was the one participant who was later excluded.
- The bootstrap interval treats the predictions as fixed. It reflects 15 participants, not 43 transcripts, and does not include model-refitting variability.

## 4. Not validated on Whisper transcripts
The model was fitted and evaluated on **human** transcripts. The application applies it to **Whisper** transcripts.
Whisper omits fillers and repetitions and makes recognition errors, all of which change vocabulary variety.

- Performance on Whisper-generated transcripts **has not been validated**. The numbers above do not transfer to Whisper input.
- The paired recordings exist on TalkBank behind member login. With access, the same 15 held-out participants can be scored from both transcript versions, which is the test that would settle this:

```
(obtain the 48 QAB recordings through TalkBank membership, place them under D:/FoundationSpeech-data/QAB/audio)
python scripts/qab_pilot_audit.py --data D:/FoundationSpeech-data/QAB        # human-transcript evaluation (done)
# then: transcribe each recording with Whisper and repeat the same leave-one-participant-out evaluation
```

## 5. Deployed bundle (audited: 11/11 checks pass)
`python scripts/qab_bundle_audit.py --data D:/FoundationSpeech-data/QAB --bundle artifacts/severity`

| Check | Result |
|---|---|
| Target names the administered instrument, with file/sheet/column provenance | pass |
| Fitted on exactly the 15 eligible participants / 43 transcripts | pass |
| Refit from source data reproduces the stored coefficients | pass |
| Application inference path reproduces training predictions for all 43 transcripts | pass (max difference 5e-4, from rounding) |
| Transcripts under 25 words are refused | pass |
| Stored interval equals the saved procedure and is described as what it is | pass |
| Input beyond the training range gives no score | pass |
| SHAP attributions add up to the actual prediction | pass |
| Synthetic bundles cannot be loaded, promoted or served | pass (test suite) |

The first promoted bundle failed 4 of these (WAB-R mislabel, no provenance, loosely described interval, no range guard) and was replaced.

**Interval.** ±32.0 points: the **largest** held-out error over the 43 single recordings (the largest per participant is 30.0).
With so few participants there is no narrower defensible interval. It is not a probability, and its coverage is unknown for Whisper transcripts.

**Behaviour in the application**

| Situation | Result |
|---|---|
| 25+ transcribed words, both features within the training range (±0.5 SD) | score, predicted AQ, band label, ±32 interval, SHAP/LIME; labelled **PILOT** and **NOT_VALID_FOR_CLINICAL_INFERENCE** |
| Fewer than 25 transcribed words | `INSUFFICIENT_DATA`, no score |
| A feature outside the training range (MATTR 0.42–0.84, clause-marker rate 0–6.4) | `OUT_OF_DISTRIBUTION`, no score (raw value kept) |
| Raw output more than 10 points outside 0–100 | `INVALID_OUTPUT`, no score |

Everyone in the training data has aphasia, so **fluent speech with a varied vocabulary is outside the training range and gets no score**.
This is a change from the first deployment, which clipped such a case to "0.0 / 100".

## 6. Limitations
- **Sample:** 15 participants, one study, cross-validation only; no independent test set.
- **Whisper:** not validated on Whisper transcripts (section 4).
- **Low-output speakers:** untested for people who produce fewer than 25 words; the most severe participant was excluded for this reason.
- **Text only:** no acoustic features and no speech-foundation-model embeddings; this is the language branch, not the proposed multimodal framework.
- **Label timing:** the AQ was measured at the first session and applied to sessions up to 51 days later.
- **Instrument edition:** WAB (1982). The band label uses AQ ranges published for the WAB-R (Kertesz, 2007); they are not separately verified for the 1982 edition.
- **Elicitation:** transcripts are a transcribed sample of clinician-led conversation.
- **Not clinically validated.**

## 7. Controls
- **Switch:** `backend/.env` → `FSA_ALLOW_PILOT_MODEL=true`. Set it to `false` and restart to return to `MODEL_NOT_READY`.
- **Re-run the evaluation:** `python scripts/qab_pilot_audit.py --data D:/FoundationSpeech-data/QAB`
- **Export (only if the decision passed):** `python scripts/qab_pilot_audit.py --data D:/FoundationSpeech-data/QAB --export-from <audit_report.json>`
- **Audit a bundle:** `python scripts/qab_bundle_audit.py --data D:/FoundationSpeech-data/QAB --bundle <dir>`
- **Promote:** `python -m app.training.promote <audit dir> --reviewer "<name>" --pilot --replace`
