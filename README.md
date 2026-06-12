# 🎙️ Deepfake Audio Detector

> An end-to-end deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using a soft-voting ensemble of three neural architectures trained on the Fake-or-Real (FoR) dataset.

**Ensemble Accuracy: 91.1% · EER: 7.8% · F1: 91.6%**

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Feature Extraction](#-feature-extraction)
- [Model Architectures](#-model-architectures)
- [Ensemble Strategy](#-ensemble-strategy)
- [Performance Report](#-performance-report)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web App](#-web-app)
- [Dependencies](#-dependencies)

---

## 🧠 Project Overview

Voice cloning and text-to-speech synthesis have reached a level of realism where even trained human listeners struggle to distinguish AI-generated speech from genuine recordings. This project builds a binary audio classifier to solve that problem automatically.

The system extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** from raw audio and passes the sequence through three independently trained deep learning models. Their probability outputs are averaged (soft voting) to produce a final verdict.

**What it detects:**
- AI-generated speech from TTS systems (e.g. WaveNet, Tacotron, FastSpeech)
- Voice-cloned audio
- Any synthetic speech not recorded from a live human

**What it does not detect:**
- Audio manipulation / splicing of genuine speech
- Background noise or environmental spoofing

---

## 🌐 Live Demo

The app is hosted on Streamlit Community Cloud:

```
https://deepfakeaudiodetector-wav.streamlit.app
```

Upload any `.wav`, `.flac`, or `.mp3` file (≥ 3 seconds recommended) and receive an instant verdict with confidence scores and signal visualizations.

---

## 📁 Repository Structure

```
deepfake-audio-detector/
│
├── app.py                        # Streamlit web app
├── notebook.ipynb                # Full training pipeline (EDA → train → evaluate)
├── test_audio.py                 # CLI script to test new audio samples
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── outputs/
│   ├── ensemble_softvote.keras   # Trained ensemble model
│   └── ensemble_meta.json        # Config + decision threshold
│
└── report/
    ├── performance_report.md     # Accuracy, EER, F1, confusion matrix
    └── confusion_matrix.png      # Confusion matrix figure
```

---

## 📦 Dataset

**Fake-or-Real (FoR) — `for-norm` split**

| Split      | Genuine | Fake   | Total  |
|------------|---------|--------|--------|
| Train      | ~45,000 | ~45,000 | ~90,000 |
| Validation | ~5,000  | ~5,000  | ~10,000 |
| Test       | ~5,000  | ~5,000  | ~10,000 |

- **Genuine samples:** real human speech recordings
- **Fake samples:** TTS-synthesized speech from multiple engines including Google TTS, Amazon Polly, Microsoft Azure, and open-source vocoders
- **Sampling rate:** 16 kHz (all files resampled)
- **Download:** [Fake-or-Real Dataset](https://bil.eecs.yorku.ca/datasets/)

---

## 🔧 Preprocessing Pipeline

Every audio file goes through a fixed, reproducible pipeline before being fed to any model:

```
Raw Audio File
      │
      ▼
 librosa.load()         →  mono, resampled to 16 kHz
      │
      ▼
 Center-crop / Pad      →  fixed length: 48,000 samples (3 seconds)
      │                     if shorter → zero-pad at end
      ▼                     if longer  → crop from center
 MFCC Extraction        →  40 coefficients, n_fft=512, hop_length=160
      │
      ▼
 Per-sequence Normalization  →  (x − mean) / (std + 1e-9)
      │
      ▼
 Shape: (300, 40)        →  300 time frames × 40 MFCC features
      │
      ▼
 Model Input: (1, 300, 40)
```

**Key parameters:**

| Parameter     | Value        | Notes                              |
|---------------|--------------|------------------------------------|
| Sample rate   | 16,000 Hz    | Standard for speech processing     |
| Duration      | 3 seconds    | 48,000 samples; center-crop policy |
| n_mfcc        | 40           | Captures timbre and vocal tract shape |
| n_fft         | 512          | FFT window (~32 ms at 16 kHz)      |
| hop_length    | 160          | 10 ms frame shift                  |
| Time frames T | 300          | After hop, 3 s → ~300 frames       |

---

## 🎛️ Feature Extraction

**Mel-Frequency Cepstral Coefficients (MFCCs)** were chosen as the primary feature representation for several reasons:

1. **Perceptual alignment** — the Mel scale approximates human auditory perception, making MFCCs sensitive to the timbral differences between natural and synthetic voice
2. **Compactness** — 40 coefficients per frame efficiently encode spectral envelope shape
3. **Robustness** — MFCCs are less sensitive to channel noise than raw spectrograms
4. **Proven effectiveness** — widely validated in speaker verification and anti-spoofing literature

Each audio clip is converted to a 2D matrix of shape `(T=300, n_mfcc=40)`, representing the temporal evolution of the MFCC features. This sequence is normalized per-clip to zero mean and unit variance before model input.

---

## 🏗️ Model Architectures

Three models are trained independently on the same MFCC sequences.

---

### Model 1 — 1D-CNN

Captures **local spectro-temporal patterns** in the MFCC sequence using 1D convolutions along the time axis.

```
Input (300, 40)
    │
Conv1D(64, kernel=3, relu) → BatchNorm → MaxPool1D(2)
    │
Conv1D(128, kernel=3, relu) → BatchNorm → MaxPool1D(2)
    │
Conv1D(256, kernel=3, relu) → BatchNorm → GlobalAvgPool1D
    │
Dense(128, relu) → Dropout(0.4)
    │
Dense(1, sigmoid)          → P(fake)
```

**Accuracy: 82.6%**

---

### Model 2 — BiLSTM with Attention

Captures **long-range temporal dependencies** across the full 3-second sequence. The attention mechanism highlights the most discriminative time frames.

```
Input (300, 40)
    │
Bidirectional LSTM(128, return_sequences=True)
    │
Bidirectional LSTM(64, return_sequences=True)
    │
Attention Layer (learned weights over time axis)
    │
SumPool1D → Dense(64, relu) → Dropout(0.4)
    │
Dense(1, sigmoid)            → P(fake)
```

**Accuracy: 80.5%**

---

### Model 3 — CNN-BiLSTM (Hybrid)

Combines local feature extraction (CNN) with sequential modeling (BiLSTM). The CNN first learns local patterns, then the LSTM models temporal dynamics over CNN-extracted features.

```
Input (300, 40)
    │
Conv1D(64, kernel=3, relu) → BatchNorm → MaxPool1D(2)
    │
Conv1D(128, kernel=3, relu) → BatchNorm → MaxPool1D(2)
    │
Bidirectional LSTM(128, return_sequences=False)
    │
Dense(128, relu) → Dropout(0.4)
    │
Dense(1, sigmoid)            → P(fake)
```

**Accuracy: 89.2%**

---

## 🗳️ Ensemble Strategy

A **soft-voting ensemble** averages the raw sigmoid probabilities from all three models before thresholding:

```
P_ensemble(fake) = [ P_cnn(fake) + P_bilstm(fake) + P_cnnbilstm(fake) ] / 3

Prediction = "Deepfake"  if  P_ensemble(fake) ≥ threshold
             "Genuine"   otherwise
```

The decision threshold is **optimized on the validation set** to minimize Equal Error Rate (EER), yielding a default of **0.535** (slightly above 0.5, biased toward reducing false negatives on fakes).

Soft voting was preferred over hard voting because:
- It preserves confidence information from each model
- More numerically stable when one model is uncertain
- Allows threshold tuning post-hoc without retraining

---

## 📊 Performance Report

### Overall Metrics (Test Set)

| Metric              | 1D-CNN | BiLSTM-Attn | CNN-BiLSTM | **Ensemble** |
|---------------------|--------|-------------|------------|--------------|
| Accuracy            | 82.6%  | 80.5%       | 89.2%      | **91.1%**    |
| F1 Score            | 82.1%  | 79.8%       | 88.9%      | **91.6%**    |
| EER                 | 17.4%  | 19.5%       | 10.8%      | **7.8%**     |
| AUC-ROC             | 0.891  | 0.874       | 0.951      | **0.968**    |

### Per-class Report — Ensemble (Test Set)

| Class    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| Genuine  | 90.4%     | 92.1%  | 91.2% | 5,000   |
| Deepfake | 91.9%     | 90.2%  | 91.0% | 5,000   |
| **Avg**  | **91.1%** | **91.1%** | **91.1%** | 10,000 |

### Confusion Matrix (Ensemble, threshold = 0.535)

```
                  Predicted
                Genuine   Fake
Actual Genuine  [ 4,605    395 ]
Actual Fake     [   490  4,510 ]
```

- **False Positive Rate** (genuine labeled fake): 7.9%
- **False Negative Rate** (fake labeled genuine): 9.8%

### Equal Error Rate (EER)

EER is the point on the DET curve where the false acceptance rate equals the false rejection rate. An EER of **7.8%** means the ensemble achieves balanced error in both directions at its optimal threshold.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deepfake-audio-detector.git
cd deepfake-audio-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** replace `tensorflow` in `requirements.txt` with `tensorflow-macos` and add `tensorflow-metal` for GPU acceleration.

---

## 🚀 Usage

### Run the Streamlit web app locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Upload a `.wav`, `.flac`, or `.mp3` file to get a verdict.

---

### Test a single audio file from the command line

```bash
python test_audio.py --file path/to/audio.wav
```

**Example output:**

```
──────────────────────────────────────────
 File       : sample_speech.wav
 Verdict    : GENUINE (Human)
 Confidence : 94.3%
 P(fake)    : 0.0572
 P(real)    : 0.9428
 Threshold  : 0.535
──────────────────────────────────────────
```

**Optional flags:**

```bash
python test_audio.py --file audio.wav --threshold 0.4   # custom threshold
python test_audio.py --file audio.wav --json            # output as JSON
```

---

### Run the training notebook

Open `notebook.ipynb` in Jupyter or Google Colab and run all cells in order:

```
1. Data loading & EDA
2. Preprocessing & MFCC extraction
3. Model definitions (1D-CNN, BiLSTM-Attention, CNN-BiLSTM)
4. Training loop with callbacks
5. Evaluation & threshold tuning
6. Ensemble soft voting
7. Final metrics & confusion matrix
```

Trained models are saved automatically to `./outputs/`.

---

## 🌍 Web App

The Streamlit app (`app.py`) provides:

| Feature | Details |
|---------|---------|
| File upload | WAV, FLAC, MP3 up to 200 MB |
| Audio playback | In-browser player |
| Signal visualization | Waveform + Mel spectrogram |
| Verdict card | GENUINE / DEEPFAKE with scan-line animation |
| Probability bars | P(genuine) and P(fake) with gradient bars |
| Threshold slider | Adjustable from 0.1 → 0.9 (default 0.535) |
| JSON export | Download full result as `.json` |

---

## 📋 Dependencies

| Package        | Version   | Purpose                            |
|----------------|-----------|------------------------------------|
| `streamlit`    | ≥ 1.35.0  | Web application framework          |
| `tensorflow`   | ≥ 2.15.0  | Model training and inference       |
| `keras`        | ≥ 3.0.0   | Model definition and serialization |
| `librosa`      | ≥ 0.10.0  | Audio loading and MFCC extraction  |
| `numpy`        | ≥ 1.24.0  | Numerical operations               |
| `matplotlib`   | ≥ 3.7.0   | Waveform and spectrogram plots     |
| `soundfile`    | ≥ 0.12.1  | Audio file I/O backend for librosa |

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

<div align="center">
  Made with ♥ by <strong>Suhani</strong>
</div>
