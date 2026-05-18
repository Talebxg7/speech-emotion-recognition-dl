"""
RAVDESS Preprocessing Pipeline
================================
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Paper:   Livingstone & Russo (2018), PLoS ONE 13(5): e0196391
         https://doi.org/10.1371/journal.pone.0196391

RAVDESS filename format:
  03-01-06-01-02-01-12.wav
  │  │  │  │  │  │  └─ Actor (01-24)
  │  │  │  │  │  └──── Repetition (01-02)
  │  │  │  │  └─────── Statement (01-02)
  │  │  │  └────────── Intensity (01=normal, 02=strong)
  │  │  └───────────── Emotion (01=neutral,02=calm,03=happy,04=sad,
  │  │                          05=angry,06=fearful,07=disgust,08=surprised)
  │  └──────────────── Vocal channel (01=speech, 02=song)
  └─────────────────── Modality (01=AV, 02=video, 03=audio)

Outputs (saved to ./processed_data/):
  - X_train_mel.npy    (N, 128, 130, 1)  CNN input
  - X_val_mel.npy
  - X_test_mel.npy
  - X_train_mfcc.npy   (N, T, 120)       LSTM/GRU input
  - X_val_mfcc.npy
  - X_test_mfcc.npy
  - y_train.npy        integer labels 0-7
  - y_val.npy
  - y_test.npy
  - label_map.json
  - scaler_mel.pkl
  - scaler_mfcc.pkl
  - summary.json

Usage:
  python preprocess_ravdess.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAVDESS_DIR = "./Audio_Speech_Actors_01-24"   # folder with Actor_01..Actor_24
OUT_DIR     = "./processed_data"
SAMPLE_RATE = 22050
DURATION    = 3.0          # pad/trim every clip to 3 seconds
N_MFCC      = 40
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
RANDOM_SEED = 42
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10

# RAVDESS emotion codes (3rd segment of filename, 1-indexed)
EMOTION_MAP = {
    "01": (0, "neutral"),
    "02": (1, "calm"),
    "03": (2, "happy"),
    "04": (3, "sad"),
    "05": (4, "angry"),
    "06": (5, "fearful"),
    "07": (6, "disgust"),
    "08": (7, "surprised"),
}
LABEL_MAP = {v[0]: v[1] for v in EMOTION_MAP.values()}  # int → string


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_emotion(filename: str):
    """
    Parse emotion label from RAVDESS filename.
    Returns integer label or None if not a speech file.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("-")
    if len(parts) < 3:
        return None
    modality     = parts[0]   # 03 = audio-only
    vocal_channel = parts[1]  # 01 = speech
    emotion_code  = parts[2]

    # Keep only audio-only speech files
    if modality != "03" or vocal_channel != "01":
        return None

    entry = EMOTION_MAP.get(emotion_code)
    return entry[0] if entry else None


def load_and_pad(filepath: str, sr: int, duration: float):
    """Load audio, resample, pad or trim to fixed length."""
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio


def augment(audio: np.ndarray, sr: int):
    """Return augmented variants of the audio clip."""
    variants = []
    target_len = len(audio)

    # 1. Time stretch (slower)
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.85)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)))
        variants.append(stretched[:target_len])
    except Exception:
        pass

    # 2. Pitch shift up
    try:
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        variants.append(pitched)
    except Exception:
        pass

    # 3. Gaussian noise
    rng = np.random.RandomState(RANDOM_SEED)
    noise = rng.randn(len(audio)).astype(np.float32)
    variants.append(audio + 0.005 * noise)

    return variants


def extract_mel(audio: np.ndarray, sr: int):
    """Log-mel spectrogram → shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_mfcc(audio: np.ndarray, sr: int):
    """MFCC + delta + delta-delta → shape (T, N_MFCC*3)."""
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)
    return combined.T                                          # (T, 120)


# ─────────────────────────────────────────────
# STEP 1 — Collect files
# ─────────────────────────────────────────────

def collect_files(ravdess_dir: str):
    print(f"\n[1/5] Scanning {ravdess_dir} ...")
    if not os.path.isdir(ravdess_dir):
        raise FileNotFoundError(
            f"Directory not found: {ravdess_dir}\n"
            "Make sure Audio_Speech_Actors_01-24 is in the project folder."
        )

    files, labels = [], []
    for actor_folder in sorted(os.listdir(ravdess_dir)):
        actor_path = os.path.join(ravdess_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if not fname.lower().endswith(".wav"):
                continue
            label = parse_emotion(fname)
            if label is None:
                continue
            files.append(os.path.join(actor_path, fname))
            labels.append(label)

    print(f"    Found {len(files)} speech WAV files across {len(LABEL_MAP)} emotions.")
    for idx, name in LABEL_MAP.items():
        count = labels.count(idx)
        print(f"      {idx} {name}: {count}")
    return files, labels


# ─────────────────────────────────────────────
# STEP 2 — Split
# ─────────────────────────────────────────────

def split_files(files, labels):
    print("\n[2/5] Splitting into train / val / test ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        files, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_tr,
        random_state=RANDOM_SEED
    )
    print(f"    Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# ─────────────────────────────────────────────
# STEP 3 — Extract features
# ─────────────────────────────────────────────

def process_split(file_list, label_list, augment_data=False, split_name=""):
    mels, mfccs, ys = [], [], []
    for fpath, label in tqdm(zip(file_list, label_list),
                             total=len(file_list),
                             desc=f"    {split_name}"):
        try:
            audio = load_and_pad(fpath, SAMPLE_RATE, DURATION)
        except Exception as e:
            print(f"      Skipping {fpath}: {e}")
            continue

        mels.append(extract_mel(audio, SAMPLE_RATE))
        mfccs.append(extract_mfcc(audio, SAMPLE_RATE))
        ys.append(label)

        if augment_data:
            for aug in augment(audio, SAMPLE_RATE):
                mels.append(extract_mel(aug, SAMPLE_RATE))
                mfccs.append(extract_mfcc(aug, SAMPLE_RATE))
                ys.append(label)

    return mels, mfccs, ys


def pad_to_uniform(arrays):
    """Pad all 2D arrays to the same time length."""
    max_len = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        pad_width = [(0, max_len - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
        padded.append(np.pad(a, pad_width, mode="constant"))
    return np.stack(padded, axis=0)


# ─────────────────────────────────────────────
# STEP 4 — Normalise
# ─────────────────────────────────────────────

def normalize_mel(tr, va, te):
    N_tr = tr.shape[0]
    H    = tr.shape[1]
    def flat(x): return x.reshape(-1, H)
    scaler = StandardScaler()
    tr2 = scaler.fit_transform(flat(tr)).reshape(tr.shape)
    va2 = scaler.transform(flat(va)).reshape(va.shape)
    te2 = scaler.transform(flat(te)).reshape(te.shape)
    return tr2, va2, te2, scaler


def normalize_mfcc(tr, va, te):
    F = tr.shape[-1]
    def flat(x): return x.reshape(-1, F)
    scaler = StandardScaler()
    tr2 = scaler.fit_transform(flat(tr)).reshape(tr.shape)
    va2 = scaler.transform(flat(va)).reshape(va.shape)
    te2 = scaler.transform(flat(te)).reshape(te.shape)
    return tr2, va2, te2, scaler


# ─────────────────────────────────────────────
# STEP 5 — Save
# ─────────────────────────────────────────────

def save_outputs(out_dir, arrays, scalers):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[5/5] Saving to {out_dir}/ ...")

    for name, arr in arrays.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"    {name}.npy  {arr.shape}  {arr.dtype}")

    for name, scaler in scalers.items():
        path = os.path.join(out_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"    {name}.pkl")

    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    print("    label_map.json")

    summary = {
        "dataset": "RAVDESS",
        "paper": "Livingstone & Russo (2018) — PLoS ONE 13(5): e0196391",
        "doi": "https://doi.org/10.1371/journal.pone.0196391",
        "sample_rate": SAMPLE_RATE,
        "duration_sec": DURATION,
        "n_mels": N_MELS,
        "n_mfcc": N_MFCC,
        "mfcc_features_per_frame": int(arrays["X_train_mfcc"].shape[-1]),
        "splits": {
            "train": int(arrays["y_train"].shape[0]),
            "val":   int(arrays["y_val"].shape[0]),
            "test":  int(arrays["y_test"].shape[0]),
        },
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "label_map": LABEL_MAP,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("    summary.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  RAVDESS Preprocessing Pipeline")
    print("=" * 55)

    # 1. Collect
    files, labels = collect_files(RAVDESS_DIR)

    # 2. Split (before augmentation — no data leakage)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_files(files, labels)

    # 3. Extract + augment
    print("\n[3/5] Extracting features (train augmented ×4) ...")
    tr_mels, tr_mfccs, y_tr = process_split(X_tr, y_tr, augment_data=True,  split_name="train")
    va_mels, va_mfccs, y_va = process_split(X_va, y_va, augment_data=False, split_name="val  ")
    te_mels, te_mfccs, y_te = process_split(X_te, y_te, augment_data=False, split_name="test ")

    # 4. Stack + normalise
    print("\n[4/5] Stacking + normalising ...")

    tr_mel = np.stack(tr_mels)[..., np.newaxis].astype(np.float32)
    va_mel = np.stack(va_mels)[..., np.newaxis].astype(np.float32)
    te_mel = np.stack(te_mels)[..., np.newaxis].astype(np.float32)

    tr_mfcc = pad_to_uniform(tr_mfccs).astype(np.float32)
    va_mfcc = pad_to_uniform(va_mfccs).astype(np.float32)
    te_mfcc = pad_to_uniform(te_mfccs).astype(np.float32)

    tr_mel,  va_mel,  te_mel,  sc_mel  = normalize_mel(tr_mel, va_mel, te_mel)
    tr_mfcc, va_mfcc, te_mfcc, sc_mfcc = normalize_mfcc(tr_mfcc, va_mfcc, te_mfcc)

    y_tr = np.array(y_tr, dtype=np.int32)
    y_va = np.array(y_va, dtype=np.int32)
    y_te = np.array(y_te, dtype=np.int32)

    print(f"    Train mel:  {tr_mel.shape}   mfcc: {tr_mfcc.shape}")
    print(f"    Val   mel:  {va_mel.shape}   mfcc: {va_mfcc.shape}")
    print(f"    Test  mel:  {te_mel.shape}   mfcc: {te_mfcc.shape}")

    # 5. Save
    save_outputs(
        OUT_DIR,
        arrays={
            "X_train_mel":  tr_mel,
            "X_val_mel":    va_mel,
            "X_test_mel":   te_mel,
            "X_train_mfcc": tr_mfcc,
            "X_val_mfcc":   va_mfcc,
            "X_test_mfcc":  te_mfcc,
            "y_train":      y_tr,
            "y_val":        y_va,
            "y_test":       y_te,
        },
        scalers={
            "scaler_mel":  sc_mel,
            "scaler_mfcc": sc_mfcc,
        }
    )

    print("\n✅  Done! Share processed_data/ with your team.")
    print("    Load with:")
    print("      import numpy as np")
    print("      X_train_mel  = np.load('processed_data/X_train_mel.npy')")
    print("      X_train_mfcc = np.load('processed_data/X_train_mfcc.npy')")
    print("      y_train      = np.load('processed_data/y_train.npy')")
    print("=" * 55)


if __name__ == "__main__":
    main()
