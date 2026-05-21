# -*- coding: utf-8 -*-
# =============================================================================
#  BLOCK 3 — Autoencoder + t-SNE / UMAP Visualization   (Author: Omar)
#  Speech Emotion Recognition on RAVDESS  ·  Input: MFCC sequences (130×120)
# =============================================================================
#
#  This file implements Block 3 of the project pipeline:
#     Encoder:  Flatten → Dense(256) → Dense(128) → Dense(64)  [latent space]
#     Decoder:  Dense(128) → Dense(256) → Dense(input_dim)     [reconstruction]
#
#  Outputs saved for Adad's integration block:
#     - autoencoder.keras          (full autoencoder model)
#     - encoder.keras              (encoder only — Adad uses this)
#     - ae_latent_train.npy        (latent vectors for train set)
#     - ae_latent_val.npy          (latent vectors for val set)
#     - ae_latent_test.npy         (latent vectors for test set)
#     - tsne_latent_space.png      (t-SNE visualization)
#     - umap_latent_space.png      (UMAP visualization, if umap-learn installed)
#     - omar_ae_results.json       (metrics for Adad's ablation table)
#
# =============================================================================

# %% [CELL 1] — Libraries and reproducibility
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Reshape, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.manifold import TSNE

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available     :", bool(tf.config.list_physical_devices("GPU")))

# %% [CELL 2] — (Colab) Mount Google Drive
# -----------------------------------------------------------------------------
# If your .npy files are on Drive, run this cell and set DATA_DIR.
# If they are in the same folder as this script, skip and set DATA_DIR = "."
try:
    from google.colab import drive
    drive.mount("/content/drive")
    # ← CHANGE THIS to your actual folder path on Drive:
    DATA_DIR = "/content/drive/MyDrive/dl_project/processed_data"
except Exception:
    DATA_DIR = "."

print("DATA_DIR =", DATA_DIR)

# %% [CELL 3] — Load MFCC data
# -----------------------------------------------------------------------------
# MFCC shapes: (N, 130, 120)
# Labels: (N,) — integers 0..7
# Data is ALREADY normalized (StandardScaler, mean≈0, std≈1).
# DO NOT normalize again.

def _load(name):
    return np.load(os.path.join(DATA_DIR, name))

X_train = _load("X_train_mfcc.npy")   # (4032, 130, 120)
X_val   = _load("X_val_mfcc.npy")     # (144,  130, 120)
X_test  = _load("X_test_mfcc.npy")    # (288,  130, 120)
y_train = _load("y_train.npy")
y_val   = _load("y_val.npy")
y_test  = _load("y_test.npy")

# Flatten each sample from (130, 120) to (15600,) for Dense autoencoder
INPUT_DIM = X_train.shape[1] * X_train.shape[2]   # 130 × 120 = 15600
LATENT_DIM = 64

X_train_flat = X_train.reshape(len(X_train), -1).astype(np.float32)
X_val_flat   = X_val.reshape(len(X_val),   -1).astype(np.float32)
X_test_flat  = X_test.reshape(len(X_test),  -1).astype(np.float32)

print(f"\nmean={X_train_flat.mean():.4f}  std={X_train_flat.std():.4f}"
      "  (≈0 and ≈1 means data is already normalized)")
print("\nShapes after flatten:")
print("  X_train_flat:", X_train_flat.shape)
print("  X_val_flat  :", X_val_flat.shape)
print("  X_test_flat :", X_test_flat.shape)

# Load class names
try:
    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        LABEL_MAP = json.load(f)
    CLASS_NAMES = [LABEL_MAP[str(i)] for i in range(8)]
except Exception:
    CLASS_NAMES = ["neutral", "calm", "happy", "sad",
                   "angry", "fearful", "disgust", "surprised"]

print("\nClasses:", CLASS_NAMES)

# %% [CELL 4] — Build Autoencoder
# -----------------------------------------------------------------------------
# Architecture (from README):
#   Encoder: Dense(256) → Dense(128) → Dense(64)   [latent space]
#   Decoder: Dense(128) → Dense(256) → Dense(input_dim)
#
# Design choices:
#   - BatchNormalization after each Dense for stable training
#   - Dropout(0.2) for regularization
#   - ReLU activations in hidden layers
#   - Linear activation on the final decoder output (regression, not classification)

def build_autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
    """
    Builds a Dense Autoencoder.
    Returns: autoencoder model, encoder model (for latent extraction).
    """
    # ---- ENCODER ----
    enc_input = Input(shape=(input_dim,), name="encoder_input")

    x = Dense(256, name="enc_dense1")(enc_input)
    x = BatchNormalization(name="enc_bn1")(x)
    x = tf.keras.layers.Activation("relu", name="enc_relu1")(x)
    x = Dropout(0.2, name="enc_drop1")(x)

    x = Dense(128, name="enc_dense2")(x)
    x = BatchNormalization(name="enc_bn2")(x)
    x = tf.keras.layers.Activation("relu", name="enc_relu2")(x)
    x = Dropout(0.2, name="enc_drop2")(x)

    latent = Dense(latent_dim, name="latent_space")(x)   # 64-dim bottleneck

    encoder = Model(enc_input, latent, name="encoder")

    # ---- DECODER ----
    dec_input = Input(shape=(latent_dim,), name="decoder_input")

    y = Dense(128, name="dec_dense1")(dec_input)
    y = BatchNormalization(name="dec_bn1")(y)
    y = tf.keras.layers.Activation("relu", name="dec_relu1")(y)
    y = Dropout(0.2, name="dec_drop1")(y)

    y = Dense(256, name="dec_dense2")(y)
    y = BatchNormalization(name="dec_bn2")(y)
    y = tf.keras.layers.Activation("relu", name="dec_relu2")(y)

    # Linear output — reconstruct normalized MFCC values (can be negative)
    reconstruction = Dense(input_dim, activation="linear",
                           name="reconstruction")(y)

    decoder = Model(dec_input, reconstruction, name="decoder")

    # ---- FULL AUTOENCODER ----
    ae_input = Input(shape=(input_dim,), name="ae_input")
    encoded  = encoder(ae_input)
    decoded  = decoder(encoded)
    autoencoder = Model(ae_input, decoded, name="autoencoder")

    return autoencoder, encoder


autoencoder, encoder = build_autoencoder()
autoencoder.summary()
encoder.summary()

# %% [CELL 5] — Compile and Train
# -----------------------------------------------------------------------------
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",          # reconstruction loss: mean squared error
    metrics=["mae"]      # mean absolute error as secondary metric
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

print("\n" + "="*60)
print("  Training Autoencoder")
print("="*60)

history = autoencoder.fit(
    X_train_flat, X_train_flat,        # input = target (reconstruction)
    validation_data=(X_val_flat, X_val_flat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# %% [CELL 6] — Evaluate reconstruction loss
# -----------------------------------------------------------------------------
val_loss, val_mae   = autoencoder.evaluate(X_val_flat,  X_val_flat,  verbose=0)
test_loss, test_mae = autoencoder.evaluate(X_test_flat, X_test_flat, verbose=0)

print(f"\n  Validation  — MSE: {val_loss:.6f}  |  MAE: {val_mae:.6f}")
print(f"  Test        — MSE: {test_loss:.6f}  |  MAE: {test_mae:.6f}")

# %% [CELL 7] — Plot training curves
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Autoencoder — Reconstruction Loss (MSE)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["mae"],     label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("Autoencoder — MAE")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ae_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> ae_training_curves.png saved.")

# %% [CELL 8] — Extract latent vectors (for Adad's integration)
# -----------------------------------------------------------------------------
# Run encoder only (no dropout at inference time)
ae_latent_train = encoder.predict(X_train_flat, verbose=0)   # (4032, 64)
ae_latent_val   = encoder.predict(X_val_flat,   verbose=0)   # (144,  64)
ae_latent_test  = encoder.predict(X_test_flat,  verbose=0)   # (288,  64)

print("\nLatent vector shapes:")
print("  ae_latent_train:", ae_latent_train.shape)
print("  ae_latent_val  :", ae_latent_val.shape)
print("  ae_latent_test :", ae_latent_test.shape)

np.save("ae_latent_train.npy", ae_latent_train)
np.save("ae_latent_val.npy",   ae_latent_val)
np.save("ae_latent_test.npy",  ae_latent_test)
print(">> Latent vectors saved.")

# %% [CELL 9] — Save models
# -----------------------------------------------------------------------------
autoencoder.save("autoencoder.keras")
encoder.save("encoder.keras")
print(">> autoencoder.keras saved.")
print(">> encoder.keras saved.  (share this with Adad)")

# %% [CELL 10] — t-SNE Visualization
# -----------------------------------------------------------------------------
# Visualize whether the 8 emotions form separable clusters in latent space.
# We use the TEST set (288 samples) — small enough for fast t-SNE.

print("\nRunning t-SNE on test latent vectors (this may take ~1 min)...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=SEED,
    verbose=1
)
latent_2d = tsne.fit_transform(ae_latent_test)   # (288, 2)

# Plot
COLORS = plt.cm.get_cmap("tab10", 8)

fig, ax = plt.subplots(figsize=(9, 7))
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = y_test == cls_idx
    ax.scatter(
        latent_2d[mask, 0],
        latent_2d[mask, 1],
        c=[COLORS(cls_idx)],
        label=cls_name,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=60
    )

ax.set_title("t-SNE of Autoencoder Latent Space (Test Set)", fontsize=13)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("tsne_latent_space.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> tsne_latent_space.png saved.")

# %% [CELL 11] — UMAP Visualization (bonus, if umap-learn is installed)
# -----------------------------------------------------------------------------
try:
    import umap
    print("\nRunning UMAP on test latent vectors...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=SEED,
        verbose=True
    )
    latent_umap = reducer.fit_transform(ae_latent_test)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_test == cls_idx
        ax.scatter(
            latent_umap[mask, 0],
            latent_umap[mask, 1],
            c=[COLORS(cls_idx)],
            label=cls_name,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=60
        )

    ax.set_title("UMAP of Autoencoder Latent Space (Test Set)", fontsize=13)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("umap_latent_space.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(">> umap_latent_space.png saved.")

except ImportError:
    print(">> umap-learn not installed. Run:  pip install umap-learn")
    print("   Skipping UMAP — t-SNE plot is sufficient.")

# %% [CELL 12] — Save results JSON (for Adad's ablation table)
# -----------------------------------------------------------------------------
results = {
    "block": "Autoencoder (Omar)",
    "input": "MFCC flattened (130×120 = 15600-dim)",
    "latent_dim": LATENT_DIM,
    "epochs_trained": len(history.history["loss"]),
    "val_mse":  round(float(val_loss),  6),
    "val_mae":  round(float(val_mae),   6),
    "test_mse": round(float(test_loss), 6),
    "test_mae": round(float(test_mae),  6),
    "latent_outputs": {
        "train": "ae_latent_train.npy",
        "val":   "ae_latent_val.npy",
        "test":  "ae_latent_test.npy",
    },
    "saved_models": {
        "autoencoder": "autoencoder.keras",
        "encoder":     "encoder.keras"
    }
}

with open("omar_ae_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n>> omar_ae_results.json saved.")
print("\n" + "="*60)
print("  AUTOENCODER DONE")
print("="*60)
print(f"  Latent dim : {LATENT_DIM}")
print(f"  Val MSE    : {val_loss:.6f}")
print(f"  Test MSE   : {test_loss:.6f}")
print(f"  Epochs     : {len(history.history['loss'])}")
print("\n  Files to share with Adad:")
print("    encoder.keras")
print("    ae_latent_train.npy")
print("    ae_latent_val.npy")
print("    ae_latent_test.npy")
print("    omar_ae_results.json")
