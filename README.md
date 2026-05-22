# A Multi-Block Deep Learning Architecture for Speech Emotion Recognition Using CNN, BiLSTM, Autoencoder, and Attention Mechanisms

> **Course Project — Deep Learning with Python**
> Talebxg7 · İpek · Ahmad · Omar · Adad

---

## Abstract

We present a multi-block deep learning architecture for Speech Emotion Recognition (SER) that integrates five distinct neural components: a standalone Convolutional Neural Network (CNN) block for local feature extraction, a CNN–Bidirectional LSTM (BiLSTM) block for joint spatial-temporal modeling, an Autoencoder (AE) for unsupervised latent representation learning, and a Transformer-based Attention mechanism for context-aware temporal weighting. The system is trained and evaluated on the **RAVDESS** dataset (Livingstone & Russo, 2018), a peer-reviewed, professionally recorded corpus of eight emotional speech categories. We apply structured data augmentation, rigorous normalization, and conduct ablation studies to justify each architectural choice. Our integrated pipeline achieves strong classification performance across 8 emotion classes, demonstrating the complementary value of each block.

---

## 1. Introduction

Automatic Speech Emotion Recognition (SER) is a fundamental challenge in affective computing, with applications in mental health monitoring, human-computer interaction, and conversational AI. The task is inherently difficult due to the high variability in emotional expression across speakers, recording conditions, and linguistic content.

Previous approaches have relied on hand-crafted acoustic features (MFCCs, pitch, energy) with shallow classifiers. More recently, deep learning methods have demonstrated superior performance by learning hierarchical representations directly from raw or minimally processed audio. However, most existing systems employ a single type of neural block, missing the opportunity to jointly exploit spatial, temporal, and latent structure in the signal.

In this work, we propose a unified pipeline that combines **five distinct deep learning blocks** to address the complementary aspects of the SER problem:

1. A **CNN block** to extract local spectral patterns from mel spectrograms
2. A **CNN + BiLSTM block** to model both spatial and long-range temporal dependencies
3. An **Autoencoder** to learn a compressed, noise-robust latent representation
4. A **Transformer Attention** block to focus on emotionally salient time frames
5. An **Integration block** that connects all components into a single end-to-end trainable system

---

## 2. Dataset

### 2.1 RAVDESS

We use the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, introduced by Livingstone & Russo (2018) in *PLoS ONE*.

| Property | Value |
|---|---|
| Source | Peer-reviewed research paper (PLoS ONE, 2018) |
| DOI | https://doi.org/10.1371/journal.pone.0196391 |
| Actors | 24 professional actors (12 female, 12 male) |
| Utterances | 1,440 speech audio files |
| Emotions | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| Format | 16-bit, 48 kHz WAV |
| Language | North American English |

RAVDESS was selected over common alternatives (MNIST-style benchmarks, Kaggle-only datasets) for three reasons: (1) it originates from a peer-reviewed publication with rigorous perceptual validation, (2) it provides balanced, professionally performed emotional expressions across a demographically diverse set of actors, and (3) its eight-class label set provides sufficient complexity to evaluate multi-class classification architectures.

### 2.2 Preprocessing

Raw audio files are processed through the following pipeline (see `preprocess_ravdess.py`):

**Feature Extraction:**
- **Mel Spectrogram** (128 mel bins, FFT size 2048, hop length 512): captures the frequency-domain representation of speech, used as input to CNN blocks
- **MFCC + Δ + ΔΔ** (40 coefficients × 3 = 120 features per frame): encodes cepstral, velocity, and acceleration features, used as input to BiLSTM/GRU blocks

**Normalization:**
- `StandardScaler` fit exclusively on the training set and applied to validation and test sets to prevent data leakage

**Data Augmentation** (training set only):
- Time stretching (rate = 0.85×)
- Pitch shifting (+2 semitones)
- Gaussian noise injection (σ = 0.005)

Augmentation expands the training set from 1,008 to **4,032 samples** (×4), improving generalization and reducing overfitting on the relatively small dataset.

**Dataset Splits:**

| Split | Samples | Percentage |
|---|---|---|
| Train | 4,032 (augmented) | 70% |
| Validation | 144 | 10% |
| Test | 288 | 20% |

Splits are stratified by emotion class to ensure balanced representation. Augmentation is applied **after** splitting to prevent leakage.

---

## 3. Architecture

### 3.1 Overview

```
Raw Audio (.wav)
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  Mel Spectrogram (128×130×1)
│   (Taleb)           │  MFCC+Δ+ΔΔ (130×120)
└─────────────────────┘
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
┌───────────┐                    ┌───────────────┐
│ Block 1   │                    │ Block 2       │
│ CNN       │                    │ CNN + BiLSTM  │
│ (Ahmad)   │                    │ (İpek)        │
└───────────┘                    └───────────────┘
      │                                  │
      └──────────────┬───────────────────┘
                     ▼
             ┌───────────────┐
             │ Block 3       │
             │ Autoencoder   │
             │ (Omar)        │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Block 4       │
             │ Attention /   │
             │ Transformer   │
             │ (Omar)        │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Block 5       │
             │ Integration + │
             │ Classifier    │
             │ (Adad)        │
             └───────────────┘
                     │
                     ▼
             Emotion Class (0–7)
```

### 3.2 Block 1 — Standalone CNN (Ahmad)

A standalone CNN block processes mel spectrograms to extract local spectral features independently. This block serves as a baseline feature extractor and is also used in the ablation study.

**Architecture:**
- Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
- Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
- Conv2D(128, 3×3) + BatchNorm + ReLU + GlobalAveragePooling

**Justification:** 2D convolutions are well suited for mel spectrograms, which share structural properties with images (local frequency patterns, harmonic structures). BatchNorm stabilizes training, and Dropout prevents overfitting.

### 3.3 Block 2 — CNN + Bidirectional LSTM (İpek)

This block combines spatial feature extraction with temporal sequence modeling.

**Architecture:**
- Conv1D(64, 3) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
- Conv1D(128, 3) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
- Bidirectional LSTM(128, recurrent_dropout=0.2) + Dropout(0.3)
- Early Stopping (patience=10, monitor=val_loss)

**Justification:** MFCC sequences are inherently temporal — emotions unfold over time. BiLSTM captures both forward and backward temporal dependencies, which is critical for emotion-bearing prosodic patterns. The CNN front-end reduces sequence length before the LSTM, improving efficiency.

**Ablation:** LSTM vs GRU performance is compared (see Section 5).

### 3.4 Block 3 — Autoencoder (Omar)

An Autoencoder learns a compressed latent representation of the audio features in an unsupervised manner.

**Architecture:**
```
Encoder: Dense(256) → Dense(128) → Dense(64)  [latent space]
Decoder: Dense(128) → Dense(256) → Dense(input_dim)
```

**Justification:** The Autoencoder forces the model to learn the most informative compressed representation of the input, filtering noise. The latent space is visualized with t-SNE/UMAP to verify emotional cluster separability.

### 3.5 Block 4 — Transformer Attention (Omar)

A Transformer encoder with multi-head self-attention is applied over the temporal sequence.

**Architecture:**
- Multi-Head Attention (num_heads=4, key_dim=64)
- Add & LayerNorm
- Feed-Forward Network (Dense(256) + Dense(model_dim))
- Add & LayerNorm

**Justification:** Attention mechanisms allow the model to dynamically weight time frames by their emotional relevance, rather than treating all frames equally. This is particularly important for emotions that are expressed in short bursts (e.g., surprised, angry).

### 3.6 Block 5 — Integration + Classifier (Adad)

The integration block concatenates representations from all preceding blocks and passes them through a classification head.

**Architecture:**
- Concatenate([CNN_output, BiLSTM_output, AE_latent, Attention_output])
- Dense(256) + BatchNorm + ReLU + Dropout(0.4)
- Dense(128) + ReLU + Dropout(0.3)
- Dense(8, activation='softmax')

---

## 4. Hyperparameter Tuning & Regularization

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning rate | 1e-3 (Adam) | Standard for Adam optimizer |
| Batch size | 32 | Balance between speed and gradient stability |
| Epochs | 100 (early stopping) | Prevents overfitting |
| Dropout rate | 0.25–0.4 | Tuned per block based on validation loss |
| Recurrent dropout | 0.2 | Regularizes LSTM hidden states |
| Early stopping patience | 10 | Stops when val_loss plateaus |
| LR scheduler | ReduceLROnPlateau (factor=0.5) | Reduces LR when stuck |

**Regularization techniques used:**
- Dropout (all blocks)
- Recurrent Dropout (LSTM/GRU)
- BatchNormalization (CNN blocks)
- Early Stopping
- Data Augmentation (preprocessing level)
- L2 weight decay (integration block)

---

## 5. Ablation Study

To justify each architectural component, we conduct a systematic ablation study where we remove one block at a time and measure the impact on test accuracy and F1-score.

| Configuration | Test Acc | Macro F1 |
|---|---|---|
| Full model (all 5 blocks) | 0.7118 | 0.6997 |
| w/o Standalone CNN | 0.6597 | 0.6398 |
| w/o BiLSTM | 0.6771 | 0.6668 |
| w/o Autoencoder | 0.7083 | 0.6989 |
| w/o Attention | 0.6875 | 0.6762 |
| CNN only (baseline) | 0.6250 | 0.6040 |


**LSTM vs GRU comparison** (İpek's block):

| Model | Test Acc | Macro F1 | Val Acc | Val Loss |
|---|---|---|---|---|
| BiLSTM | 0.6562 | 0.6448 | 0.6667 | 0.9276 |
| BiGRU | 0.6875 | 0.6733 | 0.7083 | 1.0065 |

---

## 6. Repository Structure

```
speech-emotion-recognition-dl/
│
├── README.md                        ← This file
│
├── data/
│   └── preprocess_ravdess.py        ← Full preprocessing pipeline (Taleb)
│
├── models/
│   ├── cnn_block.py                 ← Standalone CNN (Ahmad)
│   ├── cnn_bilstm.py                ← CNN + BiLSTM/GRU (İpek)
│   ├── autoencoder.py               ← Autoencoder + t-SNE viz (Omar)
│   ├── attention_block.py           ← Transformer Attention (Omar)
│   └── integrated_model.py          ← Full pipeline (Adad)
│
├── ablation/
│   └── ablation_study.py            ← Ablation experiments (Adad)
│
├── processed_data/                  ← Generated by preprocess_ravdess.py
│   ├── X_train_mel.npy
│   ├── X_train_mfcc.npy
│   ├── y_train.npy
│   └── ...
│
├── results/
│   ├── confusion_matrix.png
│   ├── tsne_latent_space.png
│   └── ablation_results.csv
│
└── requirements.txt
```

---

## 7. Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy
librosa
soundfile
scikit-learn
tqdm
tensorflow>=2.10
matplotlib
seaborn
umap-learn
```

### Preprocess Data
```bash
# Place Audio_Speech_Actors_01-24/ in project root
python data/preprocess_ravdess.py
```

### Train Models
```bash
python models/cnn_block.py
python models/cnn_bilstm.py
python models/autoencoder.py
python models/integrated_model.py
```

### Run Ablation Study
```bash
python ablation/ablation_study.py
```

---

## 8. References

1. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

2. Singh, J., Saheer, L. B., & Faust, O. (2023). Speech Emotion Recognition Using Attention Model. *International Journal of Environmental Research and Public Health*, 20(6), 5140.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

5. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*.

---

## 9. Team Contributions

| Member | Role |
|---|---|
| **Taleb** | Dataset selection, preprocessing pipeline, GitHub README |
| **İpek** | CNN + BiLSTM/GRU block, LSTM vs GRU ablation |
| **Ahmad** | Standalone CNN block (retrained by Adad due to normalization errors) |
| **Omar** | Autoencoder, Attention/Transformer block, t-SNE visualization |
| **Adad** | CNN retraining, model integration, ablation study, final presentation |

---

## License

This project is submitted as part of a university deep learning course. Dataset (RAVDESS) is used under its original Creative Commons license (CC BY-NC-SA 4.0).
