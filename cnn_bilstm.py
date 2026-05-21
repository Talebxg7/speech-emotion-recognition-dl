# -*- coding: utf-8 -*-
# =============================================================================
#  BLOCK 2 — CNN + Bidirectional LSTM / GRU   (Author: İpek)
#  Speech Emotion Recognition on RAVDESS  ·  Input: MFCC + Δ + ΔΔ sequences
# =============================================================================
#
#  Bu dosya bir Colab notebook gibi "# %% [CELL]" bloklarına ayrılmıştır.
#  Her "# %%" bir Colab hücresine karşılık gelir; sırayla yapıştırıp çalıştır.
#
#  Bu blok ders projesindeki 2. bloğu uygular:
#     Conv1D (yerel zaman örüntüleri) → Bidirectional LSTM/GRU (zamansal modelleme)
#
#  GÖREV KAPSAMI (iş bölümünden):
#     - CNN katmanları (Conv1D + BatchNorm + MaxPool + Dropout)
#     - Bidirectional LSTM VEYA GRU
#     - Düzenlileştirme: Dropout, Recurrent Dropout, Early Stopping
#     - ABLATION: BiLSTM vs BiGRU karşılaştırması (tablo + grafik)
#
#  TASARIM NOTLARI (ekip arkadaşının CNN bloğundaki hatalardan kaçınmak için):
#     1) Veri ZATEN StandardScaler ile normalize (mean≈0, std≈1).
#        => Tekrar normalize ETMİYORUZ. (Ahmet'in /max hatasını yapmıyoruz.)
#     2) Değerlendirmeyi DÜRÜST yapıyoruz: test seti varsa test üzerinde,
#        yoksa açıkça "validation" diyerek val üzerinde. Val'i "test" diye
#        yutturmuyoruz.
#     3) neutral (sınıf 0) yarı sayıda örneğe sahip => accuracy yanıltabilir,
#        bu yüzden MACRO-F1 da raporluyoruz.
# =============================================================================


# %% [CELL 1] — Kütüphaneler ve tekrarlanabilirlik (reproducibility)
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,
    Dropout, Bidirectional, LSTM, GRU, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Sonuçların her çalıştırmada aynı çıkması için tohum (seed) sabitliyoruz.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available     :", bool(tf.config.list_physical_devices("GPU")))


# %% [CELL 2] — (Colab) Google Drive'ı bağla
# -----------------------------------------------------------------------------
# .npy dosyalarını Drive'a koyduysan bu hücreyi çalıştır ve DATA_DIR'i ayarla.
# Dosyalar notebook'la aynı klasördeyse bu hücreyi atlayabilir, DATA_DIR="" yap.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    # ÖRNEK: dosyaların Drive'da şu klasörde olduğunu varsayar — KENDİ yoluna göre düzelt:
    DATA_DIR = "/content/drive/MyDrive/dl_project/processed_data"
except Exception:
    # Colab değilse / lokalde çalışıyorsan
    DATA_DIR = "."

print("DATA_DIR =", DATA_DIR)


# %% [CELL 3] — Veriyi yükle (MFCC) + akıllı test-seti yönetimi
# -----------------------------------------------------------------------------
# Bizim bloğumuz MFCC dizilerini kullanır:
#     X_*_mfcc : (N, 130, 120)  -> 130 zaman adımı, her adımda 120 özellik
#                (40 MFCC + 40 Δ + 40 ΔΔ)
#     y_*      : (N,)           -> 0..7 tamsayı etiketler
#
# NOT: y etiketleri tamsayı olduğu için kayıp fonksiyonu olarak
#      'sparse_categorical_crossentropy' kullanacağız (one-hot'a gerek yok).

def _load(name):
    return np.load(os.path.join(DATA_DIR, name))

X_train = _load("X_train_mfcc.npy")
X_val   = _load("X_val_mfcc.npy")
y_train = _load("y_train.npy")
y_val   = _load("y_val.npy")

# Test seti MFCC dosyası bazen paylaşılmamış olabilir. Varsa kullan, yoksa
# değerlendirmeyi validation üzerinden yap ama bunu AÇIKÇA belirt.
HAS_TEST = os.path.exists(os.path.join(DATA_DIR, "X_test_mfcc.npy"))
if HAS_TEST:
    X_test = _load("X_test_mfcc.npy")
    y_test = _load("y_test.npy")
    EVAL_NAME = "TEST"
    X_eval, y_eval = X_test, y_test
    print(">> X_test_mfcc bulundu — değerlendirme TEST seti üzerinde yapılacak.")
else:
    X_eval, y_eval = X_val, y_val
    EVAL_NAME = "VALIDATION"
    print(">> UYARI: X_test_mfcc.npy YOK. Değerlendirme VALIDATION üzerinde")
    print("   yapılacak. Gerçek test skoru için Taleb'den X_test_mfcc.npy iste.")

# ÖNEMLİ: Veri zaten StandardScaler ile normalize (mean≈0, std≈1).
# Bu yüzden burada TEKRAR normalize ETMİYORUZ.
print("\nmean=%.4f  std=%.4f  (≈0 ve ≈1 ise veri zaten normalize demektir)"
      % (X_train.mean(), X_train.std()))

print("\nShapes:")
print("  X_train:", X_train.shape, " y_train:", y_train.shape)
print("  X_val  :", X_val.shape,   " y_val  :", y_val.shape)
print("  X_eval :", X_eval.shape,  " y_eval :", y_eval.shape, f"({EVAL_NAME})")

# Etiket haritası (grafiklerde sınıf isimleri için)
try:
    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        LABEL_MAP = json.load(f)
    CLASS_NAMES = [LABEL_MAP[str(i)] for i in range(8)]
except Exception:
    CLASS_NAMES = ["neutral", "calm", "happy", "sad",
                   "angry", "fearful", "disgust", "surprised"]
print("\nClasses:", CLASS_NAMES)

# Model girişi için boyutlar
TIMESTEPS = X_train.shape[1]   # 130
N_FEATURES = X_train.shape[2]  # 120
N_CLASSES = 8


# %% [CELL 4] — Model kurucu fonksiyon: CNN ön-yüz + BiRNN
# -----------------------------------------------------------------------------
# Tek bir fonksiyonla hem BiLSTM hem BiGRU kurabilmek için rnn_type parametresi
# kullanıyoruz. Böylece ABLATION'da iki modeli ADİL biçimde (aynı CNN ön-yüzü,
# aynı dropout, aynı boyutlar) karşılaştırabiliyoruz — tek değişen RNN tipi.
#
# MİMARİ GEREKÇESİ:
#   * Conv1D ön-yüz: MFCC dizisinde komşu zaman adımları arasındaki yerel
#     örüntüleri yakalar VE MaxPooling ile diziyi kısaltır (130 -> ~32),
#     bu da ardından gelen RNN'i hem hızlandırır hem de uzun-bağımlılık
#     öğrenmesini kolaylaştırır.
#   * Bidirectional RNN: Duygu, bir cümlenin hem başına hem sonuna yayılır.
#     İleri+geri okuma (bidirectional) her iki yönden bağlamı yakalar.
#   * Düzenlileştirme: Dropout (genel), recurrent_dropout (RNN durumları),
#     L2 (son katman ağırlıkları), BatchNorm (CNN'i kararlı eğitir).

def build_cnn_birnn(rnn_type="lstm",
                    rnn_units=128,
                    conv_dropout=0.3,
                    rnn_dropout=0.3,
                    recurrent_dropout=0.2,
                    l2_lambda=1e-4):
    """CNN(Conv1D) ön-yüzü + Bidirectional LSTM/GRU sınıflandırıcı kurar.

    Args:
        rnn_type: "lstm" veya "gru" — ablation için değişen tek parametre.
        rnn_units: RNN gizli birim sayısı (bidirectional olduğu için çıktı 2x).
        conv_dropout: CNN bloklarından sonraki dropout oranı.
        rnn_dropout: RNN çıktısından sonraki dropout oranı.
        recurrent_dropout: RNN'in tekrarlayan (zamansal) bağlantılarına dropout.
        l2_lambda: Son dense katmanlarda L2 ağırlık cezası.

    Returns:
        Derlenmemiş (uncompiled) tf.keras Model.
    """
    rnn_type = rnn_type.lower()
    assert rnn_type in ("lstm", "gru"), "rnn_type 'lstm' veya 'gru' olmalı"
    RNNLayer = LSTM if rnn_type == "lstm" else GRU

    inp = Input(shape=(TIMESTEPS, N_FEATURES), name="mfcc_input")

    # ---- CNN ön-yüz: 1. blok ----
    # padding='same' => zaman boyutunu Conv adımında korur; küçültmeyi
    # yalnızca MaxPooling yapsın diye. Activation'ı BatchNorm'DAN SONRA
    # koyuyoruz (yaygın ve kararlı sıralama: Conv -> BN -> ReLU).
    x = Conv1D(64, kernel_size=3, padding="same", name="conv1")(inp)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)   # 130 -> 65
    x = Dropout(conv_dropout, name="cdrop1")(x)

    # ---- CNN ön-yüz: 2. blok ----
    x = Conv1D(128, kernel_size=3, padding="same", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)   # 65 -> 32
    x = Dropout(conv_dropout, name="cdrop2")(x)

    # ---- Bidirectional RNN ----
    # return_sequences=False => her diziyi tek bir vektöre özetler (son durum).
    # Bu vektör doğrudan sınıflandırıcıya gider.
    x = Bidirectional(
        RNNLayer(rnn_units,
                 recurrent_dropout=recurrent_dropout,
                 return_sequences=False),
        name=f"bi_{rnn_type}"
    )(x)
    x = Dropout(rnn_dropout, name="rdrop")(x)

    # ---- Sınıflandırıcı kafa ----
    x = Dense(64, activation="relu",
              kernel_regularizer=l2(l2_lambda), name="dense1")(x)
    x = Dropout(0.3, name="head_drop")(x)
    out = Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = Model(inp, out, name=f"cnn_bi{rnn_type}")
    return model


# %% [CELL 5] — Eğitim yardımcı fonksiyonu (her iki model için ortak)
# -----------------------------------------------------------------------------
# Aynı eğitim ayarlarını iki modelde de kullanmak ABLATION'ı adil kılar.

def compile_and_train(model, tag):
    """Modeli derler, eğitir, eğitim geçmişini (history) döndürür."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        # val_loss iyileşmeyi 12 epoch durdurursa eğitimi kes ve EN İYİ
        # ağırlıkları geri yükle (overfit'i önler).
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=1),
        # val_loss platoya girince öğrenme oranını yarıya indir.
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    print(f"\n{'='*60}\n  Eğitim başlıyor: {tag}\n{'='*60}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# %% [CELL 6] — Değerlendirme yardımcı fonksiyonu (dürüst metrikler)
# -----------------------------------------------------------------------------
def evaluate_model(model, tag):
    """Eval seti (test ya da val) üzerinde accuracy + macro-F1 hesaplar.

    Returns: dict(metrikler) ve tahmin edilen sınıflar (confusion matrix için)
    """
    probs = model.predict(X_eval, verbose=0)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_eval, y_pred)
    # macro-F1: her sınıfa eşit ağırlık verir -> dengesiz 'neutral' için adil.
    macro_f1 = f1_score(y_eval, y_pred, average="macro")

    print(f"\n----- {tag} | {EVAL_NAME} sonuçları -----")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print("\n  Sınıf bazında rapor:")
    print(classification_report(y_eval, y_pred,
                                target_names=CLASS_NAMES, digits=3,
                                zero_division=0))

    # En iyi epoch'taki val skorları (history'den değil, modelden okunur)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    return {
        "model": tag,
        "eval_set": EVAL_NAME,
        f"{EVAL_NAME.lower()}_accuracy": round(float(acc), 4),
        f"{EVAL_NAME.lower()}_macro_f1": round(float(macro_f1), 4),
        "val_accuracy": round(float(val_acc), 4),
        "val_loss": round(float(val_loss), 4),
    }, y_pred


# %% [CELL 7] — ABLATION: BiLSTM modelini eğit ve değerlendir
# -----------------------------------------------------------------------------
model_lstm = build_cnn_birnn(rnn_type="lstm")
model_lstm.summary()

hist_lstm = compile_and_train(model_lstm, "CNN+BiLSTM")
res_lstm, pred_lstm = evaluate_model(model_lstm, "CNN+BiLSTM")
model_lstm.save("cnn_bilstm.keras")


# %% [CELL 8] — ABLATION: BiGRU modelini eğit ve değerlendir
# -----------------------------------------------------------------------------
# DİKKAT: Aynı yapı, tek fark RNN tipi. Adil karşılaştırma.
model_gru = build_cnn_birnn(rnn_type="gru")
model_gru.summary()

hist_gru = compile_and_train(model_gru, "CNN+BiGRU")
res_gru, pred_gru = evaluate_model(model_gru, "CNN+BiGRU")
model_gru.save("cnn_bigru.keras")


# %% [CELL 9] — SONUÇ TABLOSU (README'ye yapıştırılmaya hazır Markdown)
# -----------------------------------------------------------------------------
# Bu hücre, README'deki BOŞ "LSTM vs GRU" tablosunu dolduracak metni üretir.

def epochs_trained(history):
    return len(history.history["loss"])

rows = [
    ("CNN+BiLSTM", res_lstm, hist_lstm),
    ("CNN+BiGRU",  res_gru,  hist_gru),
]

eval_lc = EVAL_NAME.lower()
print("\n\n=== README için Markdown tablosu (kopyala-yapıştır) ===\n")
print(f"| Model | {EVAL_NAME} Acc | Macro-F1 | Val Acc | Val Loss | Epochs |")
print("|---|---|---|---|---|---|")
for name, res, hist in rows:
    print(f"| {name} | {res[f'{eval_lc}_accuracy']:.4f} | "
          f"{res[f'{eval_lc}_macro_f1']:.4f} | {res['val_accuracy']:.4f} | "
          f"{res['val_loss']:.4f} | {epochs_trained(hist)} |")

# JSON olarak da kaydet (Adad'ın ablation tablosuna eklemesi için)
ablation_out = {
    "block": "CNN + BiLSTM/GRU (İpek)",
    "eval_set": EVAL_NAME,
    "note": ("Değerlendirme TEST üzerinde." if HAS_TEST
             else "X_test_mfcc paylaşılmadığı için VALIDATION üzerinde."),
    "results": [res_lstm, res_gru],
}
with open("ipek_ablation_results.json", "w") as f:
    json.dump(ablation_out, f, indent=2, ensure_ascii=False)
print("\n>> ipek_ablation_results.json kaydedildi.")


# %% [CELL 10] — GRAFİK 1: Eğitim eğrileri (LSTM vs GRU yan yana)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: doğruluk (accuracy)
axes[0].plot(hist_lstm.history["val_accuracy"], label="BiLSTM val_acc")
axes[0].plot(hist_gru.history["val_accuracy"],  label="BiGRU val_acc")
axes[0].set_title("Validation Accuracy: BiLSTM vs BiGRU")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Sağ: kayıp (loss)
axes[1].plot(hist_lstm.history["val_loss"], label="BiLSTM val_loss")
axes[1].plot(hist_gru.history["val_loss"],  label="BiGRU val_loss")
axes[1].set_title("Validation Loss: BiLSTM vs BiGRU")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_lstm_vs_gru_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(">> ablation_lstm_vs_gru_curves.png kaydedildi.")


# %% [CELL 11] — GRAFİK 2: Confusion matrix (daha iyi olan model için)
# -----------------------------------------------------------------------------
# Hangi model daha iyiyse onun karışıklık matrisini çiziyoruz.
best_name, best_pred = ("CNN+BiLSTM", pred_lstm)
if res_gru[f"{eval_lc}_macro_f1"] > res_lstm[f"{eval_lc}_macro_f1"]:
    best_name, best_pred = ("CNN+BiGRU", pred_gru)

cm = confusion_matrix(y_eval, best_pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name} ({EVAL_NAME})")
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
# Hücrelere sayıları yaz
thresh = cm.max() / 2.0
for i in range(8):
    for j in range(8):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix_best_birnn.png", dpi=150, bbox_inches="tight")
plt.show()
print(f">> En iyi model: {best_name}")
print(">> confusion_matrix_best_birnn.png kaydedildi.")


# %% [CELL 12] — KISA YORUM (rapora yazılabilecek özet)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("  ÖZET (rapora yazılabilir)")
print("="*60)
print(f"""
Bu blok, MFCC+Δ+ΔΔ dizilerini Conv1D ön-yüzünden geçirip Bidirectional
{best_name.split('Bi')[-1]} ile zamansal olarak modelledi. {EVAL_NAME} setinde en iyi model
{best_name} oldu (macro-F1 = {max(res_lstm[f'{eval_lc}_macro_f1'], res_gru[f'{eval_lc}_macro_f1']):.4f}).

LSTM vs GRU: GRU daha az parametreyle benzer/yakın sonuç verme eğilimindedir;
LSTM ise uzun bağımlılıklarda bazen hafif avantaj sağlar. Tablodaki gerçek
sayılar hangisinin bu veri setinde öne çıktığını gösterir.

NOT: {'TEST seti kullanıldı.' if HAS_TEST else 'X_test_mfcc paylaşılmadığı için VALIDATION raporlandı; Talebden test dosyasını iste.'}
""")
