# FoundationSpeech-Aphasia

**FoundationSpeech-Aphasia: A Self-Supervised Multi-Modal Framework for Explainable Aphasia Severity Prediction Using Speech Foundation Models and Clinical Language Analysis**

FoundationSpeech-Aphasia is an academic research prototype for demonstrating an explainable, multimodal speech analysis concept. Its short product name is used in compact application chrome; the complete formal research title above is used in project and research-facing content.

## Run locally

**Quickest (Windows):** double-click `start.bat`. It builds the frontend and serves the app and API together at **http://127.0.0.1:8000**.

Do not open `dist/index.html` directly from disk: browsers cannot load its scripts over `file://`, so the page stays blank.

For frontend development with hot reload:

```sh
npm install
npm run dev            # frontend on :5173, proxies /api to the backend
```

Research backend (FastAPI, Python 3.11+), see [backend/README.md](../backend/README.md):

```sh
cd backend
python -m venv .venv && .venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

## Data modes

- **LIVE BACKEND**: real audio validation, preprocessing, Whisper ASR, clinical language features and speech-foundation-model embeddings from the local backend. Severity prediction uses a text-only PILOT model (QAB corpus, see [PILOT_QAB_TEXT.md](PILOT_QAB_TEXT.md)) when `FSA_ALLOW_PILOT_MODEL=true` and a promoted bundle exists in `backend/artifacts/severity`; otherwise it returns `MODEL_NOT_READY`. Model bundles are not stored in git.
- **SIMULATED / DEMONSTRATION DATA**: the original demonstration outputs. It is used automatically when the backend is offline and can be selected from the header chip. Simulated values are never presented as model predictions.

## Prototype scope

Do not upload identifiable patient information. This prototype is not a certified medical device and is not for diagnosis or treatment decisions. Nothing in it has been clinically validated.

Frontend service seams are in `src/services/`; shared research result contracts are in `src/types/`.
