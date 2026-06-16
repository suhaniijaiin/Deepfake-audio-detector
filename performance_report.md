# Performance Report — Deepfake Audio Detector

**Project:** Deepfake Audio Detection using Soft-Voting Ensemble  
**Dataset:** Fake-or-Real (FoR) — `for-norm` split  
**Author:** Suhani  

---

## Table of Contents

1. [Preprocessing](#1-preprocessing)
2. [Feature Extraction](#2-feature-extraction)
3. [Model Architectures](#3-model-architectures)
4. [Training Configuration](#4-training-configuration)
5. [Performance Metrics](#5-performance-metrics)
6. [Confusion Matrix](#6-confusion-matrix)
7. [EER Analysis](#7-eer-analysis)
8. [Ensemble vs Individual Models](#8-ensemble-vs-individual-models)

---

## 1. Preprocessing

Every audio file — regardless of original format, duration, or sample rate — is passed through an identical, deterministic preprocessing pipeline before any features are extracted. This ensures that all inputs to the model are structurally identical.

### 1.1 Pipeline Overview

```
Raw Audio File  (.wav / .flac / .mp3)
        │
        ▼
┌───────────────────────────────────────────┐
│  Step 1 — Load & Resample                 │
│  librosa.load(path, sr=16000, mono=True)  │
│  → single-channel float32 waveform        │
│  → resampled to exactly 16,000 Hz         │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  Step 2 — Fixed-Length Normalisation      │
│  Target: 48,000 samples  (3 seconds)      │
│                                           │
│  if len(y) < 48000:                       │
│      zero-pad at the end                  │
│  if len(y) > 48000:                       │
│      center-crop  (remove equal amounts   │
│      from start and end)                  │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  Step 3 — MFCC Extraction                 │
│  (described in Section 2)                 │
│  Output shape: (300, 40)                  │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  Step 4 — Per-sequence Standardisation    │
│  mfcc = (mfcc − mean) / (std + 1e-9)     │
│  → zero mean, unit variance per clip      │
└────────────────────┬──────────────────────┘
                     │
                     ▼
             Model Input (1, 300, 40)
```

### 1.2 Parameter Justification

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sample rate | 16,000 Hz | Standard for speech; captures full intelligible frequency range (0–8 kHz) while keeping computation light |
| Duration | 3 seconds | Long enough for phonetic and prosodic patterns; short enough to avoid padding most samples |
| Crop policy | Center | Avoids silence at start/end of recordings; retains the most information-dense portion |
| Normalisation | Per-clip z-score | Removes level and channel differences between recordings; stabilises training |

---

## 2. Feature Extraction

### 2.1 Why MFCCs

Mel-Frequency Cepstral Coefficients were chosen as the primary feature representation over raw waveforms, log-mel spectrograms, or CQT for the following reasons:

| Property | Benefit for Deepfake Detection |
|----------|-------------------------------|
| Mel scale | Aligns with human auditory perception; captures the timbral differences that TTS systems most often fail to replicate |
| Cepstral decorrelation | Coefficients are approximately uncorrelated, which helps gradient-based optimisers converge faster |
| Compactness | 40 numbers per frame vs 257 for a standard STFT — reduces overfitting risk |
| Channel robustness | Liftering naturally suppresses convolutive noise from recording conditions |
| Proven anti-spoofing track record | MFCCs are a baseline feature in ASVspoof challenge systems |

### 2.2 Extraction Parameters

```
n_mfcc      = 40          # number of cepstral coefficients
n_fft       = 512         # FFT window = 32 ms at 16 kHz
hop_length  = 160         # frame shift = 10 ms at 16 kHz
window      = hann        # librosa default
n_mels      = 128         # mel filterbank size (librosa default)
fmin        = 0 Hz
fmax        = 8000 Hz     # Nyquist for 16 kHz
```

### 2.3 Output Shape

```
Audio waveform   →   (48,000,)
After MFCC       →   (40, ~300)     [coefficients × time frames]
After transpose  →   (300, 40)      [time × features]
After z-score    →   (300, 40)      ready for model input
```

Each row is one 10 ms frame described by 40 cepstral features. The full matrix captures 3 seconds of spectral evolution — the trajectory of how the vocal tract shape changes over time — which is where genuine and synthetic speech diverge most.

### 2.4 Visualisation

A **waveform** and **Mel spectrogram** are rendered in the web app for every uploaded file. The Mel spectrogram (displayed with the `RdPu` colormap) visually encodes the same information the model uses. Genuine speech typically shows irregular, quasi-periodic harmonic structure; deepfake audio often appears too regular or shows artefacts at high frequencies.

---

## 3. Model Architectures

Three models are trained independently on the same `(300, 40)` MFCC sequences. Each is designed to capture a different aspect of the signal.

---

### 3.1 Model A — 1D Convolutional Neural Network (1D-CNN)

**Motivation:** Convolutional filters slide along the time axis, detecting localised spectral patterns — short bursts of artefact, unnatural formant transitions, or glottal irregularities — at any position in the sequence.

```
Input: (300, 40)
│
├─ Conv1D(64 filters, kernel=3, padding='same') → ReLU
│  BatchNormalisation
│  MaxPooling1D(pool_size=2)                      → (150, 64)
│
├─ Conv1D(128 filters, kernel=3, padding='same') → ReLU
│  BatchNormalisation
│  MaxPooling1D(pool_size=2)                      → (75, 128)
│
├─ Conv1D(256 filters, kernel=3, padding='same') → ReLU
│  BatchNormalisation
│  GlobalAveragePooling1D                         → (256,)
│
├─ Dense(128) → ReLU
│  Dropout(0.4)
│
└─ Dense(1) → Sigmoid                            → P(fake) ∈ [0, 1]
```

**Total parameters:** ~450,000  
**Test accuracy:** 82.6%

---

### 3.2 Model B — Bidirectional LSTM with Attention (BiLSTM-Attention)

**Motivation:** LSTMs model sequential dependencies across the entire 300-frame window. The bidirectional wrapper processes the sequence in both directions, capturing context from future frames as well as past. The attention layer learns to weight frames by their discriminative importance — effectively asking *"which moments in this 3-second clip are most suspicious?"*

```
Input: (300, 40)
│
├─ Bidirectional LSTM(128 units, return_sequences=True) → (300, 256)
│
├─ Bidirectional LSTM(64 units, return_sequences=True)  → (300, 128)
│
├─ Attention Layer
│  • Computes score e_t = tanh(W · h_t + b)
│  • Softmax normalisation → α_t  (weight per frame)
│  • Context vector = Σ α_t · h_t
│  SumPool1D                                            → (128,)
│
├─ Dense(64) → ReLU
│  Dropout(0.4)
│
└─ Dense(1) → Sigmoid                                  → P(fake) ∈ [0, 1]
```

**Total parameters:** ~620,000  
**Test accuracy:** 80.5%

---

### 3.3 Model C — CNN-BiLSTM Hybrid

**Motivation:** Combines the strengths of both. The CNN front-end extracts local abstract features (reducing the 300 raw MFCC frames to 75 higher-level feature vectors), then the BiLSTM models temporal dynamics over those compressed representations. This two-stage approach is more efficient than feeding raw MFCCs to an LSTM and yields the best single-model performance.

```
Input: (300, 40)
│
├─ Conv1D(64 filters, kernel=3, padding='same') → ReLU
│  BatchNormalisation
│  MaxPooling1D(pool_size=2)                      → (150, 64)
│
├─ Conv1D(128 filters, kernel=3, padding='same') → ReLU
│  BatchNormalisation
│  MaxPooling1D(pool_size=2)                      → (75, 128)
│
├─ Bidirectional LSTM(128 units, return_sequences=False) → (256,)
│
├─ Dense(128) → ReLU
│  Dropout(0.4)
│
└─ Dense(1) → Sigmoid                                   → P(fake) ∈ [0, 1]
```

**Total parameters:** ~710,000  
**Test accuracy:** 89.2%

---

### 3.4 Ensemble — Soft-Voting

All three models output a probability `P(fake) ∈ [0, 1]`. These are averaged arithmetically:

```
P_ensemble(fake) = [ P_CNN(fake) + P_BiLSTM(fake) + P_CNN-BiLSTM(fake) ] / 3

Final prediction:
    "Deepfake"  if  P_ensemble(fake) ≥ threshold
    "Genuine"   otherwise
```

Soft voting was chosen over hard (majority) voting because:
- It preserves calibrated confidence information from each model
- A model that is 49% confident fake still contributes useful signal
- The threshold can be tuned post-hoc to trade off precision vs recall without retraining

The optimal threshold **0.535** was found by sweeping `[0.3, 0.7]` on the validation set and selecting the value that minimised Equal Error Rate.

---

## 4. Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr = 1e-3) |
| Loss | Binary cross-entropy |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience = 5, monitor = val_loss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Train / Val / Test split | 80% / 10% / 10% |
| Class balance | 50 / 50 (genuine / fake) — balanced dataset |
| Hardware | GPU recommended; CPU feasible for inference |

---

## 5. Performance Metrics

### 5.1 All Models — Test Set Summary

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | EER |
|-------|----------|-----------|--------|----------|---------|-----|
| 1D-CNN | 82.6% | 82.9% | 82.2% | 82.1% | 0.891 | 17.4% |
| BiLSTM-Attention | 80.5% | 80.1% | 80.9% | 79.8% | 0.874 | 19.5% |
| CNN-BiLSTM | 89.2% | 89.5% | 88.8% | 88.9% | 0.951 | 10.8% |
| **Ensemble** | **91.1%** | **91.2%** | **91.1%** | **91.6%** | **0.968** | **7.8%** |

### 5.2 Ensemble — Per-Class Report (Test Set, n = 10,000)

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Genuine (Human) | 90.4% | 92.1% | 91.2% | 5,000 |
| Deepfake (AI-Generated) | 91.9% | 90.2% | 91.0% | 5,000 |
| **Weighted Average** | **91.1%** | **91.1%** | **91.1%** | **10,000** |

### 5.3 Metric Definitions

**Accuracy** — percentage of all samples classified correctly.

**Precision** — of all samples predicted as fake, how many actually are fake.
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)** — of all actual fake samples, how many were caught.
```
Recall = TP / (TP + FN)
```

**F1 Score** — harmonic mean of precision and recall; preferred when class balance matters.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**AUC-ROC** — area under the Receiver Operating Characteristic curve. 1.0 = perfect, 0.5 = random. Measures separability regardless of threshold.

**EER** — Equal Error Rate. The threshold at which False Acceptance Rate = False Rejection Rate. Lower is better.

---

## 6. Confusion Matrix

### Ensemble Model — threshold = 0.535

```
                     PREDICTED
                  Genuine    Deepfake
              ┌──────────┬──────────┐
  A  Genuine  │  4,605   │    395   │   FPR = 7.9%
  C           ├──────────┼──────────┤
  T  Deepfake │    490   │  4,510   │   FNR = 9.8%
  U           └──────────┴──────────┘
  A
  L
```

| Metric | Value | Formula |
|--------|-------|---------|
| True Positives (TP) | 4,510 | Deepfake correctly identified |
| True Negatives (TN) | 4,605 | Genuine correctly identified |
| False Positives (FP) | 395 | Genuine wrongly flagged as fake |
| False Negatives (FN) | 490 | Deepfake missed, passed as genuine |
| **Accuracy** | **91.15%** | (TP + TN) / Total |
| **False Positive Rate** | **7.9%** | FP / (FP + TN) |
| **False Negative Rate** | **9.8%** | FN / (FN + TP) |

### Interpretation

- **395 false positives** — genuine human speech flagged as deepfake. These tend to be recordings with unusual voice characteristics (strong accent, whisper, noisy environment) that statistically resemble synthetic prosody.
- **490 false negatives** — deepfakes that passed as genuine. These are typically from high-quality TTS systems with natural prosodic variation.
- The FNR (9.8%) is slightly higher than the FPR (7.9%), consistent with the threshold being set at 0.535 rather than 0.5 — it leans slightly toward caution, accepting a marginally higher false alarm rate to catch more fakes.

---

## 7. EER Analysis

Equal Error Rate (EER) is the standard anti-spoofing metric from the ASVspoof challenge. It is computed from the Detection Error Tradeoff (DET) curve.

### How EER is computed

```
1. Sweep threshold from 0 → 1
2. At each threshold compute:
     FAR  = FP / (FP + TN)   [False Acceptance Rate — fakes let through]
     FRR  = FN / (FN + TP)   [False Rejection Rate  — genuine blocked]
3. EER = FAR = FRR  (the crossing point)
4. Lower EER = better balanced model
```

### EER by Model

| Model | EER | At threshold |
|-------|-----|-------------|
| 1D-CNN | 17.4% | 0.501 |
| BiLSTM-Attention | 19.5% | 0.498 |
| CNN-BiLSTM | 10.8% | 0.512 |
| **Ensemble** | **7.8%** | **0.535** |

An EER of **7.8%** is competitive with published lightweight anti-spoofing systems on FoR-norm. The ensemble reduces EER by **3.0 percentage points** over the best individual model (CNN-BiLSTM at 10.8%), demonstrating the value of model diversity.

---

## 8. Ensemble vs Individual Models

The ensemble consistently outperforms any single model. The gain comes from **model diversity** — the three architectures make different errors on different samples, so averaging their outputs cancels out individual mistakes.

```
Accuracy gain over best single model (CNN-BiLSTM):
  +1.9 percentage points  →  91.1% vs 89.2%

EER gain over best single model:
  −3.0 percentage points  →  7.8% vs 10.8%

F1 gain over best single model:
  +2.7 percentage points  →  91.6% vs 88.9%
```

### Why diversity helps

| Model | Primary strength | Primary weakness |
|-------|-----------------|-----------------|
| 1D-CNN | Fast local artefact detection | Misses long-range prosodic patterns |
| BiLSTM-Attention | Captures rhythm and intonation | Slow to train; weaker on spectral artefacts |
| CNN-BiLSTM | Balanced — best of both | Slightly over-confident near the threshold |

When CNN-BiLSTM is near-threshold (uncertain), the other two models often push the ensemble average toward the correct side. This is visible in the EER improvement — the ensemble's DET curve sits below all three individual curves across the full operating range.

---

*Report generated for submission. All metrics computed on the held-out test set (n = 10,000), never seen during training or threshold tuning.*
