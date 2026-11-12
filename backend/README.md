# FoundationSpeech-Aphasia — Research Backend

**FoundationSpeech-Aphasia: A Self-Supervised Multi-Modal Framework for Explainable Aphasia Severity Prediction Using Speech Foundation Models and Clinical Language Analysis**

FastAPI backend and ML pipeline for the FoundationSpeech-Aphasia research prototype. It serves the existing React frontend (`../src`) through its service layer (`src/services/*`).

> **Research prototype — not a medical device.** Nothing produced by this system is a clinical diagnosis. No component has been clinically validated.

---

## 1. Status at a glance

Research-discipline vocabulary: **Implemented** = runs end to end · **Prototype** = runs, not scientifically validated · **Simulated** = demonstration only · **Proposed** = designed, not implemented · **Validated** = supported by experiments (nothing is in this category yet).

| Component | Status | Notes |
|---|---|---|
| Upload, validation (type, size, duration, decodability, silence) | Implemented | WAV, MP3, FLAC, OGG via libsndfile; M4A, WEBM and other formats via PyAV (bundled FFmpeg) |
| Preprocessing: decode → mono → polyphase resample → DC removal → peak normalisation → WebRTC VAD | Implemented | Optional Butterworth high-pass. **Noise reduction is not implemented**, and this is reported in every result |
| ASR (openai-whisper, configurable checkpoint) | Implemented | Word timestamps; per-word token probabilities reported as *uncalibrated* |
| Clinical language / fluency features | Prototype | 15 features with definition, unit, source, provenance and status. Syntactic complexity is **not computed** (no parser) |
| Foundation-model embeddings: Whisper encoder, wav2vec 2.0, HuBERT, WavLM | Implemented | All four installed locally; mean-pooled final layer. The representations are not yet evaluated for this task |
| Multi-model representation | Implemented | `strategy=multi` extracts from every requested model |
| Multimodal fusion (concatenate / weighted concatenate, per-block L2) | Prototype | Cross-attention transformer fusion is **Proposed** |
| Severity model: WAB-R-derived severity score (100 − AQ), regression | **MODEL_NOT_READY** | Training infrastructure READY; no labelled dataset. `prediction_available: false` until a real, test-evaluated, promoted model exists |
| SHAP / LIME | Implemented (inactive) | Run only for a promoted real-data model, on the interpretable feature layer (named features + one grouped input per speech embedding) |
| Async job lifecycle, polling, cancel | Implemented | `QUEUED → … → COMPLETED / ATTENTION_NEEDED / FAILED / CANCELLED` |
| History (SQLite) and retention | Implemented | Opaque IDs; no audio in the database |
| PDF research report (ReportLab) | Implemented | Built from stored backend data only |
| Training / evaluation infrastructure | **READY_FOR_DATA** (tested on SYNTHETIC_TEST_DATA only) | CHAT reader, dataset loader + validation report, participant-level split + leakage check, Models A/B/C, metrics, bundles, model cards, promotion. See `../docs/` |
| Comparisons: Model A (handcrafted), B (wav2vec2), C (multimodal), ablations, base paper | READY_FOR_DATA / base paper NOT_AVAILABLE | Configs in `configs/experiments/`; no results exist |

### Environment this was built and tested on
Windows 11, Intel i7-1260P (CPU only, **no CUDA GPU**), 16 GB RAM, Python 3.12, ~13 GB free disk.

- **Installed weights (2026-09-25):** Whisper `tiny` (72 MB, `~/.cache/whisper`); `facebook/wav2vec2-base`, `facebook/hubert-base-ls960` and `microsoft/wavlm-base-plus` (~380 MB each, Hugging Face cache). These are the pretrained, self-supervised base checkpoints, not ASR-fine-tuned ones. All four produce embeddings on CPU (verified dimensions 768 / 768 / 384 / 768).
- WavLM-base-plus declares no license on its model card; check its terms before redistributing results or weights.
- Measured on this CPU with 30 s chunks and 120 s of input: wav2vec2 ~15 s, HuBERT ~15 s, WavLM ~21 s, peak RAM ~2.5 GB. With 120 s chunks, WavLM peaked at 9.0 GB and took 69 s, which is why the default chunk is 30 s.
- The system-wide PyTorch install was broken (DLL load error), so the backend uses an isolated `.venv` with CPU PyTorch.

---

## 2. Setup

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt        # CPU PyTorch wheels via the extra index in requirements.txt
cp .env.example .env                   # optional; every setting has a default
```

The venv is about 1.5 GB (PyTorch CPU accounts for most of it).

### Model weights (explicit opt-in)
The API **never downloads weights** unless `FSA_ALLOW_MODEL_DOWNLOAD=true`. While downloads are disabled it also sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, because transformers otherwise contacts the Hub in the background when loading `.bin` checkpoints. To install weights deliberately (the script prefers `model.safetensors` and falls back to `pytorch_model.bin`):

```bash
python scripts/download_models.py whisper:tiny          # 72 MB  – ASR + Whisper encoder (default)
python scripts/download_models.py wav2vec2 hubert wavlm # ~380 MB each
```

Model IDs can be changed with `FSA_WAV2VEC2_MODEL_ID` and similar settings. A local directory path also works.

### GPU
Set `FSA_DEVICE=auto` (the default) or `cuda`, and install a CUDA build of PyTorch, for example `pip install torch --index-url https://download.pytorch.org/whl/cu124`. `GET /api/models` reports the detected device, CUDA availability and VRAM. Whisper uses fp16 only on CUDA.

---

## 3. Running

```bash
# from backend/
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# frontend, from the repository root (proxies /api to :8000 via vite.config.js)
npm run dev
```

The frontend header chip shows **LIVE BACKEND** when the backend health check succeeds. Otherwise it shows **SIMULATED DATA** (or **SIMULATED · OFFLINE**). Click the chip to switch modes. Simulated mode is the original demonstration and is labelled *SIMULATED / DEMONSTRATION DATA* on every page.

Interactive API docs: <http://127.0.0.1:8000/docs>.

### Tests

```bash
cd backend
python -m pytest
```

The 108 tests cover the API (upload, validation, status, results, models, history, reports), audio (WAV, stereo/44.1 kHz, FLAC, MP3, M4A, corrupt, empty, silent, too short/long, oversized), the pipeline (preprocessing, ASR parsing, real Whisper ASR and encoder when installed, HF embedding path with a tiny local random-init model, real wav2vec2/HuBERT/WavLM when installed, Hub-offline enforcement, features, unavailable models), research integrity, and the training infrastructure (target transform, AQ validation, CHAT parsing, dataset matching, missing labels, participant leakage, splits, feature dimensions, model forward/checkpoint, synthetic end-to-end training, quarantine of synthetic checkpoints, promotion rules, prediction contract, personal-recording protection).

The integrity tests check:
- no probabilities without a model;
- probabilities equal the estimator's own `predict_proba`;
- no SHAP/LIME without a prediction or background data;
- proposed fusion strategies are refused;
- live results are never flagged simulated;
- interrupted and cancelled states are preserved.

All test audio is generated programmatically. No recordings of people are used.

---

## 4. API

All errors share one shape: `{"error": {"code": "...", "message": "...", "details": {...}}}`.
Codes: `INVALID_AUDIO`, `FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `AUDIO_DECODE_ERROR`, `AUDIO_TOO_SHORT`, `AUDIO_TOO_LONG`, `MODEL_UNAVAILABLE`, `MODEL_NOT_READY`, `INFERENCE_ERROR`, `INFERENCE_FAILED`, `ASR_ERROR`, `FEATURE_EXTRACTION_ERROR`, `DATABASE_ERROR`, `REPORT_GENERATION_ERROR`, `ANALYSIS_NOT_FOUND`, `ANALYSIS_NOT_FINISHED`, `INVALID_REQUEST`, `INTERNAL_ERROR`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analysis` | multipart `file`, `strategy` (`single`/`multi`), `models` (comma list: `whisper_encoder,wav2vec2,hubert,wavlm`) → `202 {analysis_id, status, status_url, result_url}` |
| GET | `/api/analysis/{id}` | Full record: audio, preprocessing, embeddings, transcript, features, fusion, severity, explainability, stages, attention |
| GET | `/api/analysis/{id}/status` | Status, current stage, per-stage records, progress % (used for polling) |
| POST | `/api/analysis/{id}/cancel` | Cancel a queued or running job |
| GET | `/api/analysis/{id}/transcript` · `/features` · `/severity` · `/explainability` | Individual sections, each with its own `state` |
| GET | `/api/models` | Hardware, embedding-model availability (status, reason, device, dimension), ASR, fusion strategies, severity-classifier and XAI status |
| GET | `/api/history` · `/api/history/{id}` | Stored analyses (newest first) and one record |
| GET | `/api/reports/{id}` | PDF report (`409 ANALYSIS_NOT_FINISHED` while running) |
| GET | `/api/health` | Liveness; used by the frontend to choose live vs simulated mode |

`/cancel` and `/health` go beyond the requested contract. They are needed for the `CANCELLED` state and for the frontend's honest mode detection.

### Job lifecycle
`QUEUED → PREPROCESSING → MODEL_PROCESSING → ASR → FEATURE_EXTRACTION → FUSION → PREDICTION → EXPLAINABILITY → COMPLETED`

- **`ATTENTION_NEEDED`**: the job finished, but something needs a look. Examples: a requested embedding model was unavailable or failed, ASR was unavailable or failed, feature extraction failed, or VAD found no speech. In the no-speech case, ASR is skipped so Whisper cannot hallucinate text from silence.
- **`FAILED`**: preprocessing failed, or the server restarted mid-job (reported as such at next startup).
- **`CANCELLED`**: set via the cancel endpoint.
- **`MODEL_NOT_READY` severity alone does not trigger `ATTENTION_NEEDED`.** It is the prototype's known, documented state, and it is shown prominently in the result instead.

### Result section states
`PENDING`, `COMPUTED`, `PARTIAL`, `SKIPPED`, `MODEL_UNAVAILABLE`, `MODEL_NOT_READY`, `INFERENCE_FAILED`, `INSUFFICIENT_DATA`, `NOT_AVAILABLE`.

---

## 5. Pipeline details

**Features** (`app/features/clinical.py`). Each feature returns `name, label, value, unit, definition, source, provenance, status, reason, note, clinically_validated=false`.

| Feature | Source | Definition |
|---|---|---|
| speech_rate | transcript | words / recording minutes (a fluency proxy; no composite fluency score is invented) |
| articulation_rate | asr_timing | words / VAD speech seconds (word-based, not syllable-based) |
| type_token_ratio | transcript | unique / total words (length-sensitive) |
| moving_average_ttr | transcript | MATTR, window `FSA_MATTR_WINDOW` (default 25); `insufficient_data` below the window |
| brunet_index | transcript | Brunét's W |
| mean_sentence_length | transcript | words per sentence, using Whisper-predicted punctuation |
| immediate_repetition_rate, filled_pause_rate, disfluency_rate | transcript | per 100 words |
| clause_marker_density | transcript | lexical proxy, not a parse |
| pause_frequency, mean_pause_duration, pause_time_ratio, speech_ratio | vad | internal VAD gaps ≥ max(pause threshold, merge gap) |
| syntactic_complexity | — | **not computed** (requires a parser) |

Known bias: Whisper often omits fillers and cleans up repetitions, so disfluency features computed from its transcripts likely under-count. This is noted on the affected features.

**Embeddings.** Each model gives a final-layer, mean-over-frames vector. Audio is chunked at `FSA_MAX_EMBEDDING_SECONDS` (default 30 s, the same as the Whisper encoder window), and embeddings are mean-pooled across chunks, so context is limited to each 30 s chunk. The API returns dimension, frame count, L2 norm and timing. Vectors stay in memory and are not stored (they can be speaker-identifying).

**Fusion.** Per-block L2 normalisation followed by concatenation. `fitted` z-score normalisation requires training statistics and returns `MODEL_NOT_READY` without them. The fused vector is built and described, but not used for prediction.

---

## 6. Severity model and dataset (research work remaining)

The severity target, dataset pipeline, training, evaluation and model cards are documented in the repository `docs/` folder:
[TARGET_DEFINITION](../docs/TARGET_DEFINITION.md) · [DATASET_PIPELINE](../docs/DATASET_PIPELINE.md) ·
[TRAINING](../docs/TRAINING.md) · [EVALUATION](../docs/EVALUATION.md) · [MODEL_CARD_TEMPLATE](../docs/MODEL_CARD_TEMPLATE.md).

**Current state:** infrastructure is READY; the dataset is NOT AVAILABLE; training is BLOCKED; production is MODEL_NOT_READY.

Once AphasiaBank data is available (and the column names are set in `configs/datasets/aphasiabank.yaml`):

```
set APHASIA_DATA_ROOT=D:phasiabank
python -m app.training.train --config configs/experiments/model_c_multimodal.yaml --validate-only
python -m app.training.train --config configs/experiments/model_a_handcrafted.yaml
python -m app.training.train --config configs/experiments/model_b_wav2vec2.yaml
python -m app.training.train --config configs/experiments/model_c_multimodal.yaml
python -m app.training.compare artifacts/experiments/*
python -m app.training.train --config configs/experiments/<selected>.yaml --evaluate-test
python -m app.training.promote artifacts/experiments/<experiment_id> --reviewer "<name>"
```

**What the production loader accepts.** It serves only a bundle that is all of the following:
- trained on **REAL** data;
- **evaluated on held-out test participants**;
- **promoted** by a named reviewer.

Synthetic bundles cannot be loaded, promoted or served.

**Predictions on uploads.** Every prediction on an uploaded or microphone recording is marked
`NOT_VALID_FOR_CLINICAL_INFERENCE`, because those recordings are not part of the evaluated dataset.

### Research decisions that are deliberately *not* made in code
These must be decided and documented by the research team:

- **Dataset.** AphasiaBank/TalkBank is a natural candidate, but access requires TalkBank membership. Which recordings, tasks (e.g. Cinderella story, picture description) and sites to use is an open decision.
- **Clinical scale and labels.** Do not assume Mild/Moderate/Severe labels exist. AphasiaBank provides WAB-R AQ for many, but not all, participants. Any mapping from scores to classes (thresholds, citation), the handling of missing scores, and the exclusion criteria must be recorded per row in `label_source`. The adapter excludes unlabelled rows and never imputes them. If the dataset cannot support a three-class target cleanly, report that as a limitation.
- **Split.** Participant-grouped is the default. Cross-validation, site-held-out splits and the size of the test set are open.
- **Classifier architecture.** The proposed transformer severity model is **not implemented**. Only the logistic-regression baseline exists.
- **Fusion architecture.** No fusion method is claimed superior. Comparison requires experiments.
- **Model weights.** Which checkpoints to use (e.g. `wav2vec2-base` vs `-large`, Whisper size), under the disk and CPU constraints above.

### Relation to the base paper
The base paper is *"Conv-transformer-based Jaya Gazelle optimization for speech intelligibility with aphasia"* (Rajendran & Chandrasekar). Its pipeline is handcrafted acoustic features → Conv-Transformer transducer → Jaya Gazelle optimisation → recognition → language model / DRN.

This project explores a **different** direction: foundation-model representations plus clinical language features, multimodal fusion, a transformer severity classifier and SHAP/LIME. **It does not reproduce the base paper.** The "base-paper-inspired baseline" is listed as *proposed* only, and no improvement over any baseline is claimed until experiments are run.

---

## 7. Privacy and security

- **Uploads:** streamed to a per-job temporary workspace (`FSA_TEMP_DIR`) with the size limit enforced, decoded, then **deleted immediately** unless `FSA_RETAIN_AUDIO=true`. Stale workspaces are removed at startup.
- **Filenames:** not stored by default (`speech-sample.<ext>`), because they can contain names.
- **Database:** stores an opaque ID, technical metadata and computed results. Retention is set by `FSA_RETENTION_DAYS` (default 30; purged at startup).
- **Logs and errors:** logs contain analysis IDs and stage names only, never transcript text or audio. Error messages do not expose paths or content.
- **Authentication:** none; the existing project has none. Bind to `127.0.0.1` (the default) and do not expose the API publicly.
- **Transcripts:** they are stored in SQLite and may contain identifying content spoken by the participant. Use non-identifiable samples only.

## 8. Layout

```
backend/
├── app/
│   ├── main.py              FastAPI app, error handlers, lifespan
│   ├── config.py            FSA_* settings
│   ├── api/                 analysis.py, resources.py (models, history, reports, health)
│   ├── schemas/             Pydantic contracts and state enums
│   ├── services/            job lifecycle (analysis_service.py), service container
│   ├── preprocessing/       ingest.py (upload/validate/decode), pipeline.py (DSP + VAD)
│   ├── models/              hardware.py, registry.py (availability, loading, embeddings)
│   ├── pipelines/           asr.py, fusion.py, severity.py
│   ├── features/            clinical.py
│   ├── explainability/      explain.py (SHAP, LIME)
│   ├── database/            repository.py (SQLite)
│   ├── reports/             pdf.py (ReportLab)
│   ├── training/            target, chat, dataset, splits, features, models, evaluation, bundle,
│   │                        model_card, train, promote, compare, synthetic
│   └── utils/               errors.py
├── configs/                 datasets/*.yaml, experiments/*.yaml
├── scripts/                 download_models.py
├── tests/
├── requirements.txt · .env.example · pytest.ini
```

## 9. Frontend integration notes (contract changes)
Views talk only to `src/services/*`, which map backend records to the view models documented in `src/types/contracts.js`.

- **Simulated mode.** It is kept intact. `analysisService.simulatedResult()` reproduces the original demo values.
- **Severity shape.** Changed from flat fields (`severity`, `confidence`, `probabilities`) to a `severity` object with `available/state/message`, so that "no prediction" can be represented without placeholder numbers.
- **Feature and transcript shapes.** Features became objects with a `computed` flag. Transcript `confidence` may be `null`, because only defensible values are shown.
- **`onProgress` callback.** `analyze(file, onProgress, {strategy, models})` now reports real stage records, which drive the existing pipeline card.
- **Page edits.** Targeted edits in `WorkspacePages.jsx` replace hard-coded demo values with view-model fields. Layout and styling are unchanged, apart from a small header mode chip.
- **Dev proxy.** `vite.config.js` was added for the `/api` dev proxy. `VITE_API_BASE_URL` overrides the base URL.
- **Browser recording.** Real microphone capture (`src/hooks/useMicRecorder.js`: getUserMedia + MediaRecorder, with a live input-level meter). Browser noise suppression, auto-gain and echo cancellation are disabled so the raw signal reaches the pipeline. Chrome and Edge record WebM/Opus (usually 48 kHz), which the backend decodes with PyAV and resamples to 16 kHz. The microphone only works on a secure origin: http://127.0.0.1:8000 or localhost qualifies, a LAN IP over http does not.
