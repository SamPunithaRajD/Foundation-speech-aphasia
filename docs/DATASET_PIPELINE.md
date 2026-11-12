# Dataset pipeline

| Component | Status |
|---|---|
| Loader, CHAT reader, matching, validation report | **IMPLEMENTED** (tested on SYNTHETIC_TEST_DATA only) |
| AphasiaBank data | **NOT_AVAILABLE** (requires TalkBank AphasiaBank membership) |
| Real spreadsheet column mapping | **READY_FOR_DATA** (must be filled in `backend/configs/datasets/aphasiabank.yaml`) |

Code: `backend/app/training/dataset.py`, `chat.py`, `splits.py`, `features.py`.

## 1. Obtain data (blocked)
- **Membership:** AphasiaBank membership through a faculty member. Students must be sponsored by their advisor; see <https://talkbank.org/aphasia/>.
- **Downloads:** English media and CHAT transcripts, the Test Results spreadsheet (WAB-R) and Demographics.
- **Storage:** put the data on **D:\\**. C:\\ has about 15 GB free.

## 2. Layout and configuration

```
<APHASIA_DATA_ROOT>/
    audio/          media (.wav .mp3 .mp4 .m4a .flac), any sub-folders
    transcripts/    CHAT .cha, one per session
    scores/         WAB-R spreadsheet (.xlsx/.csv)
    demographics/   optional
    metadata/       optional, provenance only
```

1. Set `APHASIA_DATA_ROOT` (or `root:` in the dataset config).
2. Open the real spreadsheets and set `scores.file`, `scores.sheet`, `scores.participant_column`, `scores.aq_column` (and `scores.session_column` if AQ is per session). **Column names are never guessed**, and the loader refuses to run until they are set.
3. Check `file_id_regex` against real file names. It maps a `.cha` stem to `participant` + `session`, e.g. `adler01a` → `adler01` / `a`.
4. Optionally set `tasks:` (CHAT `@G` gems such as `Cinderella`) to build one record per session and task.

## 3. Canonical record (`DatasetRecord`)
- **Identity:** `record_id`, `participant_id`, `session_id`, `task`, `audio_path`, `transcript_path`.
- **Target:** `wab_aq`, `severity_score`, `severity_band`, `target_source`, `target_version`.
- **Transcript and audio:** `human_transcript` (normalised CHAT text; the original file is untouched), `speaker_segments_ms` (time-aligned participant utterances).
- **Metadata:** `demographics`, `aphasia_metadata` (from `@ID`), `provenance`, `data_origin` (`REAL` / `SYNTHETIC_TEST_DATA`), `issues`, `trainable`.

The Whisper transcript is stored separately in the feature cache (`whisper_transcript`), so ASR can later be compared with the human transcript.

## 4. CHAT transcripts
- **Verbatim original:** `ChatTranscript.original` keeps the file exactly as it is.
- **Separate normalised text:** per-utterance normalised text is produced separately. Each rule is listed in `chat.NORMALIZATION_RULES`:
  - retracings `[/]` keep the spoken words;
  - `&-uh` becomes "uh";
  - `xxx` is counted as unintelligible and dropped;
  - `(be)cause` becomes "because";
  - annotations and bullets are removed.
- **Kept structure:** timing bullets, `@G` task gems, `@ID` fields, `@Media`, and dependent tiers (`%mor`, `%gra`, …) are all kept.

## 5. Matching and validation
- **Matching rule:** a record is trainable only if audio, transcript, participant and a valid AQ all match, with no duplicate and time-aligned participant speech.
- **Validation report:** written to `dataset/validation_report.json`, `records.json` and `validation_summary.txt`. It lists:
  - totals: audio files, transcripts, matched and trainable records, trainable participants;
  - gaps: missing transcripts (audio with no transcript), missing audio, missing / invalid / ambiguous AQ;
  - problems: duplicate participant/session/task, unmatched file IDs, transcript warnings;
  - the derived band distribution.
- **Nothing is discarded silently.** Every record appears in `records.json` with its issues.

Run validation alone:

```
python -m app.training.train --config configs/experiments/model_c_multimodal.yaml --validate-only
```

## 6. Participant-level split
- **Whole participants only:** they are assigned to train, validation or test (default 70/15/15, stratified by participant AQ band, fixed seed).
- **Shared split:** one split per dataset version and seed (`artifacts/splits/<version>_seed<seed>.json`), shared by every experiment so models are compared on identical participants.
- **Cross-validation:** optional participant-level K-fold over train + validation. Test participants are never included.
- **Leakage check:** `assert_no_leakage` raises if any participant is in more than one split. It runs on every split that is created or loaded, and a test fails if it doesn't.

## 7. Privacy guard
- **Protected folders:** the loader refuses any dataset root that overlaps the application's data or upload folders. Microphone and uploaded recordings can therefore never become training, validation or test data.

## 8. Audio used for features
- **`audio_scope: speaker_utterances` (default):** only the time-aligned `PAR` utterances are cut from the session media and concatenated, which removes investigator speech. Sessions without time alignment are reported and not trainable in this mode.
- **Deployment mismatch:** inference on uploaded audio uses the whole file. This mismatch is documented in the model card limitations.

## 9. Dataset availability check (2026-10-12)
- **AphasiaBank:** not present on this machine (no CHAT files, no configured root). Still requires TalkBank membership.
- **`QAB-dataset.xlsx`** (found in Downloads): **scores only, not usable for training a speech model**.
  - Contents: Quick Aphasia Battery item and summary scores for 88 participants (168 sessions; 16 participants have 6 sessions), plus a second sheet with WAB-R scores for 16 participants (AQ 18.9–98.8).
  - It contains **no audio or transcript references**, so there is nothing to extract speech features from.
  - WAB-R AQ exists for only 16 participants, below the 20-participant minimum even if audio existed.
  - QAB and AQ correlate strongly in this file (r = 0.94, n = 16), but that does not justify converting one into the other; see TARGET_DEFINITION.md.
  - To use this corpus, the matching **audio recordings (and ideally transcripts)** would be needed, keyed by the same participant and session IDs, with documented permission to use them. With audio, the 16 WAB-R participants could serve only as a small external check, and a QAB target would need a separate target definition.

## 10. Changes from the 2026-10 audit
- **One `audio_scope`:** the experiment's `features.audio_scope` now decides both trainability and how audio is cut. Previously the dataset config and the feature config could disagree.
- **Empty score cells** are reported as missing AQ, not invalid AQ.
- **Saved splits are verified before reuse.** Training is blocked if participants were added or removed, or if the fractions changed, instead of silently dropping participants.
