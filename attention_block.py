# -*- coding: utf-8 -*-
# =============================================================================
#  BLOCK 4 — Transformer Attention Encoder   (Author: Omar)
#  Speech Emotion Recognition on RAVDESS  ·  Input: MFCC sequences (130×120)
# =============================================================================
#
#  Architecture (from README):
#     - Multi-Head Attention (num_heads=4, key_dim=64)
#     - Add & LayerNorm
#     - Feed-Forward Network: Dense(256) → Dense(model_dim)
#     - Add & LayerNorm
#     - Global Average Pooling → Dense classifier
#
#  This block is worth +15 bonus points and is required for the
#  5-block count (+15 bonus).
#
#  Outputs saved for Adad's integration block:
#     - attention_block.keras         (full attention model)
#     - attention_encoder.keras       (feature extractor without classifier head)
#     - attn_features_train.npy       (attention output vectors for train set)
#     - attn_features_val.npy
#     - attn_features_test.npy
#     - omar_attention_results.json   (metrics for Adad's ablation table)
#
# =============================================================================

# %% [CELL 1] — Libraries and reproducibility
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available     :", bool(tf.config.list_physical_devices("GPU")))

# %% [CELL 2] — (Colab) Mount Google Drive
# -----------------------------------------------------------------------------
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
# MFCC shape: (N, 130, 120) — 130 time steps, 120 features per step
# Data is ALREADY normalized. DO NOT normalize again.

def _load(name):
    return np.load(os.path.join(DATA_DIR, name))

X_train = _load("X_train_mfcc.npy").astype(np.float32)  # (4032, 130, 120)
X_val   = _load("X_val_mfcc.npy").astype(np.float32)    # (144,  130, 120)
X_test  = _load("X_test_mfcc.npy").astype(np.float32)   # (288,  130, 120)
y_train = _load("y_train.npy")
y_val   = _load("y_val.npy")
y_test  = _load("y_test.npy")

TIMESTEPS  = X_train.shape[1]    # 130
N_FEATURES = X_train.shape[2]    # 120
N_CLASSES  = 8

print(f"\nmean={X_train.mean():.4f}  std={X_train.std():.4f}"
      "  (≈0 and ≈1 means data is already normalized)")
print("\nShapes:")
print("  X_train:", X_train.shape)
print("  X_val  :", X_val.shape)
print("  X_test :", X_test.shape)

# Load class names
try:
    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        LABEL_MAP = json.load(f)
    CLASS_NAMES = [LABEL_MAP[str(i)] for i in range(8)]
except Exception:
    CLASS_NAMES = ["neutral", "calm", "happy", "sad",
                   "angry", "fearful", "disgust", "surprised"]

print("\nClasses:", CLASS_NAMES)

# %% [CELL 4] — Transformer Encoder Block (reusable layer)
# -----------------------------------------------------------------------------
# This implements one standard Transformer encoder layer:
#   1. Multi-Head Self-Attention   (attends to all time steps)
#   2. Add & LayerNorm             (residual connection)
#   3. Feed-Forward Network        (Dense(256) → Dense(d_model))
#   4. Add & LayerNorm             (residual connection)
#
# WHY TRANSFORMER FOR SPEECH EMOTION:
#   Emotions are often expressed in short bursts (a sharp pitch rise for
#   "surprised", a slow fall for "sad"). Self-attention lets the model
#   dynamically weight which time frames carry the most emotional signal,
#   rather than treating all frames equally like a simple average.

class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    One Transformer Encoder layer.

    Args:
        d_model   : dimensionality of the model (must match input feature dim)
        num_heads : number of attention heads
        ffn_units : number of hidden units in the Feed-Forward Network
        dropout   : dropout rate applied after attention and FFN
    """

    def __init__(self, d_model, num_heads=4, ffn_units=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout

        # Multi-Head Self-Attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,   # key_dim per head = 64 // 4 ... etc
            dropout=dropout,
            name="multi_head_attention"
        )
        self.dropout1 = Dropout(dropout)
        self.norm1    = LayerNormalization(epsilon=1e-6, name="layer_norm_1")

        # Feed-Forward Network
        self.ffn_dense1 = Dense(ffn_units, activation="relu",  name="ffn_dense1")
        self.ffn_dense2 = Dense(d_model,   activation="linear", name="ffn_dense2")
        self.dropout2   = Dropout(dropout)
        self.norm2      = LayerNormalization(epsilon=1e-6, name="layer_norm_2")

    def call(self, x, training=False):
        # --- Multi-Head Attention + residual ---
        # query = key = value = x  (self-attention)
        attn_out = self.mha(x, x, x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)          # Add & Norm

        # --- Feed-Forward Network + residual ---
        ffn_out = self.ffn_dense1(x)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)            # Add & Norm

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout": self.dropout_rate,
        })
        return config


# %% [CELL 5] — Build Full Attention Model
# -----------------------------------------------------------------------------
# Input projection: (N, 130, 120) → (N, 130, 128)
#   The Transformer requires d_model to be divisible by num_heads.
#   120 is not cleanly divisible by 4, so we project to 128 first.
#
# After the Transformer layer, we use GlobalAveragePooling1D to
# collapse the time dimension → one vector per sample → classifier.

D_MODEL   = 128    # must be divisible by num_heads
NUM_HEADS = 4
FFN_UNITS = 256
DROPOUT   = 0.1


def build_attention_model(timesteps=TIMESTEPS,
                          n_features=N_FEATURES,
                          n_classes=N_CLASSES,
                          d_model=D_MODEL,
                          num_heads=NUM_HEADS,
                          ffn_units=FFN_UNITS,
                          dropout=DROPOUT):
    """
    Full attention-based classifier.
    Returns: full model, feature_extractor (without classifier head).
    """
    inp = Input(shape=(timesteps, n_features), name="mfcc_input")

    # --- Input projection to d_model ---
    x = Dense(d_model, name="input_projection")(inp)   # (N, 130, 128)

    # --- Transformer Encoder Block ---
    x = TransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        ffn_units=ffn_units,
        dropout=dropout,
        name="transformer_encoder"
    )(x)

    # --- Aggregate over time axis ---
    # GlobalAveragePooling: (N, 130, 128) → (N, 128)
    # This is the attention-weighted summary of the full sequence.
    pooled = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Save this as feature extractor output (Adad uses this vector)
    features = Dropout(0.2, name="feature_drop")(pooled)

    # --- Classification head ---
    y = Dense(64, activation="relu", name="cls_dense1")(features)
    y = Dropout(0.3, name="cls_drop")(y)
    output = Dense(n_classes, activation="softmax", name="output")(y)

    # Full model
    full_model = Model(inp, output, name="attention_classifier")

    # Feature extractor (no classifier head) — for Adad's integration
    feature_extractor = Model(inp, features, name="attention_feature_extractor")

    return full_model, feature_extractor


attention_model, attention_extractor = build_attention_model()
attention_model.summary()

# %% [CELL 6] — Compile and Train
# -----------------------------------------------------------------------------
attention_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=12,
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
print("  Training Attention Block")
print("="*60)

history = attention_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# %% [CELL 7] — Evaluate
# -----------------------------------------------------------------------------
y_pred_probs = attention_model.predict(X_test, verbose=0)
y_pred       = y_pred_probs.argmax(axis=1)

test_acc    = accuracy_score(y_test, y_pred)
test_macro_f1 = f1_score(y_test, y_pred, average="macro")

val_loss, val_acc = attention_model.evaluate(X_val, y_val, verbose=0)

print(f"\n----- Attention Block | TEST results -----")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  Macro-F1 : {test_macro_f1:.4f}")
print("\n  Per-class report:")
print(classification_report(y_test, y_pred,
                             target_names=CLASS_NAMES, digits=3,
                             zero_division=0))

# %% [CELL 8] — Plot training curves
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Attention Block — Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Attention Block — Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attention_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> attention_training_curves.png saved.")

# %% [CELL 9] — Extract features for Adad's integration block
# -----------------------------------------------------------------------------
attn_features_train = attention_extractor.predict(X_train, verbose=0)  # (4032, 128)
attn_features_val   = attention_extractor.predict(X_val,   verbose=0)  # (144,  128)
attn_features_test  = attention_extractor.predict(X_test,  verbose=0)  # (288,  128)

print("\nAttention feature shapes:")
print("  attn_features_train:", attn_features_train.shape)
print("  attn_features_val  :", attn_features_val.shape)
print("  attn_features_test :", attn_features_test.shape)

np.save("attn_features_train.npy", attn_features_train)
np.save("attn_features_val.npy",   attn_features_val)
np.save("attn_features_test.npy",  attn_features_test)
print(">> Attention features saved.")

# %% [CELL 10] — Save models
# -----------------------------------------------------------------------------
attention_model.save("attention_block.keras")
attention_extractor.save("attention_encoder.keras")
print(">> attention_block.keras saved.")
print(">> attention_encoder.keras saved.  (share this with Adad)")

# %% [CELL 11] — Save results JSON (for Adad's ablation table)
# -----------------------------------------------------------------------------
results = {
    "block": "Transformer Attention (Omar)",
    "input": "MFCC sequences (130×120)",
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "ffn_units": FFN_UNITS,
    "epochs_trained": len(history.history["loss"]),
    "val_accuracy": round(float(val_acc), 4),
    "val_loss":     round(float(val_loss), 4),
    "test_accuracy": round(float(test_acc), 4),
    "test_macro_f1": round(float(test_macro_f1), 4),
    "feature_outputs": {
        "train": "attn_features_train.npy",
        "val":   "attn_features_val.npy",
        "test":  "attn_features_test.npy",
    },
    "saved_models": {
        "full_model":         "attention_block.keras",
        "feature_extractor":  "attention_encoder.keras"
    }
}

with open("omar_attention_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n>> omar_attention_results.json saved.")
print("\n" + "="*60)
print("  ATTENTION BLOCK DONE")
print("="*60)
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  Test Macro-F1 : {test_macro_f1:.4f}")
print(f"  Epochs        : {len(history.history['loss'])}")
print("\n  Files to share with Adad:")
print("    attention_encoder.keras")
print("    attn_features_train.npy")
print("    attn_features_val.npy")
print("    attn_features_test.npy")
print("    omar_attention_results.json")
