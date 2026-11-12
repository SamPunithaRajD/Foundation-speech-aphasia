![Animated Header](https://capsule-render.vercel.app/api?type=waving&color=0:081B19,50:17443B,100:83C9AD&height=230&section=header&text=FoundationSpeech-Aphasia&fontSize=38&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=Explainable%20AI%20for%20Speech%20and%20Aphasia%20Research&descAlignY=59&descSize=16)

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=18&duration=3000&pause=900&color=83C9AD&center=true&vCenter=true&width=750&lines=Listen.+Analyze.+Understand.;Speech+Foundation+Models+%2B+Clinical+Language+Analysis;Explainable+AI+for+Aphasia+Research;From+Speech+Signals+to+Research+Insights.)](https://git.io/typing-svg)

**An Explainable Multimodal Speech Analysis Research Prototype**

[Research Paper](#-research-publication) · [Architecture](#-system-architecture) · [Features](#-key-features) · [Installation](#-getting-started)

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-74AA9C?style=for-the-badge&logo=openai&logoColor=white)
![Research](https://img.shields.io/badge/Status-Research%20Prototype-D6AE76?style=for-the-badge)

</div>

---

##  Overview

**FoundationSpeech-Aphasia** is an AI-powered research application designed to explore explainable aphasia-focused speech analysis and severity prediction.

The project investigates how speech foundation models, automatic speech recognition, acoustic representations, and clinical language features can be combined to derive meaningful research insights from spoken language.

The application processes speech recordings, generates transcripts using Whisper, extracts linguistic and acoustic features, computes speech representations, and provides an experimental prediction workflow with explainability and automated research reports.

The goal is to explore a more accessible, data-driven approach to speech analysis while maintaining transparency about model behaviour and limitations.

##  Research Publication

This project is associated with an IEEE-published research paper.

**FoundationSpeech-Aphasia: A Self-Supervised Multi-Modal Framework for Explainable Aphasia Severity Prediction Using Speech Foundation Models and Clinical Language Analysis**

**Authors:**
- Sam Punitha Raj D
- Vishvesh Roshan S
- Chandrasekar A
- Raghavi S

**Publisher:** IEEE

---

##  Key Features

<table>
<tr>
<td width="50%">

###  Speech Processing
- Audio upload and processing
- Speech validation and preprocessing
- Voice Activity Detection
- Whisper-powered transcription
- Transcript analysis

</td>
<td width="50%">

###  Foundation Models
- wav2vec 2.0
- HuBERT
- WavLM
- Whisper Encoder
- Speech embedding extraction

</td>
</tr>
<tr>
<td width="50%">

###  Clinical Language Analysis
- Vocabulary diversity
- Lexical richness
- Clause-related linguistic measures
- Transcript-derived features
- Feature provenance

</td>
<td width="50%">

###  Explainable AI
- SHAP-based explanations
- LIME-based explanations
- Feature contribution analysis
- Prediction interpretation
- Model limitations

</td>
</tr>
<tr>
<td width="50%">

###  Research Workspace
- Analysis history
- Live backend integration
- Analysis summaries
- Model information
- Structured research outputs

</td>
<td width="50%">

###  Reporting
- Automated PDF reports
- Prediction summaries
- Speech feature summaries
- Evaluation information
- Research limitations

</td>
</tr>
</table>

##  System Architecture

The application explores a multimodal research architecture while the currently deployed experimental severity predictor uses a transcript-based pilot model.

```mermaid
flowchart TD
    A["Speech Recording"] --> B["Audio Validation & Preprocessing"]
    B --> C["Whisper Speech Recognition"]
    C --> D["Generated Transcript"]

    D --> E["Clinical Language Feature Extraction"]
    B --> F["Acoustic Feature Extraction"]
    B --> G["Speech Foundation Models"]

    G --> H["Speech Embeddings"]
    E --> I["Experimental Text-Based Pilot"]
    I --> J["Research Severity Estimate"]

    E --> K["Multimodal Research Pipeline"]
    F --> K
    H --> K

    J --> L["Explainability & Research Report"]
    K -. "Further model development" .-> L

    style A fill:#173B34,color:#fff
    style J fill:#225B4A,color:#fff
    style L fill:#173B34,color:#fff
```

### Processing Pipeline

| Stage | Description |
|---|---|
| Audio Input | Accepts speech recordings |
| Preprocessing | Audio validation, normalization and voice activity analysis |
| ASR | Whisper generates speech transcripts |
| Feature Extraction | Acoustic and linguistic feature computation |
| Speech Representations | Embeddings from supported speech foundation models |
| Prediction | Experimental transcript-based severity prediction |
| Explainability | SHAP and LIME, when supported by an eligible model |
| Reporting | Research summaries and PDF generation |

**Research distinction:** The currently deployed pilot prediction is text-based. The broader multimodal architecture is under development and should not be interpreted as a fully validated fusion model.

##  Technology Stack

<div align="center">

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python |
| Speech Recognition | OpenAI Whisper |
| Speech Foundation Models | wav2vec 2.0, HuBERT, WavLM, Whisper Encoder |
| Audio Processing | Python audio-processing libraries |
| Machine Learning | PyTorch, Scikit-learn |
| Explainable AI | SHAP, LIME |
| Data Management | SQLite |
| Reporting | PDF generation |
| Testing | Pytest |

</div>

##  Research Model

The research direction investigates three model configurations.

**Model A — Handcrafted Feature Baseline**

Acoustic and linguistic features are used as inputs to a regression model.

**Model B — Speech Foundation Model**

Speech representations extracted using wav2vec 2.0 are passed to a regression head.

**Model C — Proposed Multimodal Framework**

Speech representations, acoustic features, and clinical language features are combined through a fusion architecture.

These configurations are experimental research directions. Their relative performance must be established through appropriate dataset-level evaluation.

##  Severity Prediction

The application explores a WAB-R-derived impairment score:

\[
\text{Severity Score} = 100 - \text{WAB-R AQ}
\]

The transformation is intended to reverse the direction of the WAB-R AQ scale:

- Lower score: lower WAB-R-derived impairment.
- Higher score: higher WAB-R-derived impairment.

The current deployed text-based pilot has a limited evaluation basis. Its evaluation on human transcripts does not establish equivalent performance on Whisper-generated transcripts.

Predictions are research outputs only and are not suitable for diagnosis, treatment decisions, or independent clinical assessment.

##  Application Screens

<!-- Add your actual application screenshots to docs/assets/ and uncomment these sections. -->

<!--
<div align="center">
<img src="docs/assets/dashboard.png" width="90%" alt="Research Dashboard"/>
<br/><em>Research Dashboard</em>
</div>

<div align="center">
<img src="docs/assets/speech-analysis.png" width="90%" alt="Speech Analysis"/>
<br/><em>Speech Analysis and Experimental Prediction</em>
</div>

<div align="center">
<img src="docs/assets/report.png" width="90%" alt="Research Report"/>
<br/><em>Automated Research Report</em>
</div>
-->

##  Getting Started

### Prerequisites

- Python 3.12
- Node.js and npm
- Git
- Sufficient disk space for speech models and dependencies

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FoundationSpeech-Aphasia.git

cd FoundationSpeech-Aphasia
```

### 2. Configure the backend

```bash
cd backend

python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using the provided `.env.example` file.

Start the backend using the command documented in `backend/README.md`.

### 3. Configure the frontend

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

Use the frontend URL displayed by Vite in your terminal.

> Check the backend and frontend README files for project-specific configuration and startup commands.

---

##  Project Structure

```text
FoundationSpeech-Aphasia/
│
├── backend/
│   ├── app/
│   │   ├── features/
│   │   ├── pipelines/
│   │   ├── training/
│   │   ├── explainability/
│   │   ├── services/
│   │   └── resources/
│   │
│   ├── configs/
│   │   ├── datasets/
│   │   └── experiments/
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── docs/
│   ├── TARGET_DEFINITION.md
│   ├── DATASET_PIPELINE.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── MODEL_CARD_TEMPLATE.md
│
├── .gitignore
└── README.md
```

*The structure above describes the main application areas; individual files may vary as development progresses.*

##  Research Integrity & Limitations

This project is an experimental research prototype.

- It is not a certified medical device.
- Its predictions are not a diagnosis.
- The currently deployed pilot uses transcript-derived features.
- Training and evaluation use a small research sample.
- Generalization from human transcripts to Whisper-generated transcripts has not been established.
- Model performance and explainability must be interpreted within their evaluation limitations.
- Predictions should not replace assessment by qualified speech-language professionals.

Patient recordings, transcripts, credentials, and restricted research datasets must not be committed to this repository.

##  Future Research Directions

- Larger and more diverse aphasia datasets.
- Improved validation using paired human and Whisper transcripts.
- Further development of multimodal fusion.
- Robust severity prediction and uncertainty estimation.
- Participant-independent evaluation.
- Improved feature-level explainability.
- Further experimentation with speech foundation models.
- Research collaboration and reproducibility.

##  Contributing

Contributions, research discussions, and suggestions are welcome.

For substantial changes:

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing the modifications.

Please do not contribute patient-identifiable data or unauthorized clinical datasets.

##  Acknowledgements

We acknowledge the researchers and organizations contributing to open speech research, speech foundation models, and aphasia research resources.

##  License

Add the appropriate license before redistributing this project. Until a license is added, all rights remain with the respective copyright holders.

---

<div align="center">

### Building Explainable AI for Speech Research

*From speech signals to interpretable research insights.*

<br/>

![Animated Footer](https://capsule-render.vercel.app/api?type=waving&color=0:83C9AD,50:17443B,100:081B19&height=130&section=footer&animation=fadeIn)

</div>
