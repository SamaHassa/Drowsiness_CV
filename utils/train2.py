import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report


IMG_SIZE   = (224, 224)
BATCH_SIZE = 16 
EPOCHS     = 30
SEED       = 42

YAWN_DIR   = Path("dataset/yawn")
NOYAWN_DIR = Path("dataset/no yawn")   
TMP_DIR    = Path("yawn_split")       
MODEL_DIR  = Path("model")
MODEL_PATH = MODEL_DIR / "yawn_model.keras"



def build_split():

    all_paths, all_labels = [], []

    for folder, label in [(NOYAWN_DIR, "no_yawn"), (YAWN_DIR, "yawn")]:
        if not folder.exists():
            raise FileNotFoundError(
                f"Not found: {folder}\n"
                f"'no yawn' has a space — check exact folder name."
            )
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        print(f"  {label:10s}: {len(files):>5} images")
        for f in files:
            all_paths.append(f)
            all_labels.append(label)

    print(f"  Total : {len(all_paths)}")

    # Shuffle before split
    combined = list(zip(all_paths, all_labels))
    random.seed(SEED)
    random.shuffle(combined)

    n      = len(combined)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)

    splits = {
        "train": combined[:i_val],
        "val":   combined[i_val:i_test],
        "test":  combined[i_test:],
    }
    print(f"  Train : {len(splits['train'])} | "
          f"Val : {len(splits['val'])} | "
          f"Test : {len(splits['test'])}")

    for split, data in splits.items():
        for cls in ["yawn", "no_yawn"]:
            (TMP_DIR / split / cls).mkdir(parents=True, exist_ok=True)
        for path, label in data:
            dst = TMP_DIR / split / label / path.name
            if not dst.exists():
                shutil.copy(str(path), str(dst))

    print("  Split folders ready.\n")



def get_generators(batch_size):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    ).flow_from_directory(
        TMP_DIR / "train",
        target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="binary", shuffle=True, seed=SEED,
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(
        TMP_DIR / "val",
        target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="binary", shuffle=False,
    )

    test_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(
        TMP_DIR / "test",
        target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="binary", shuffle=False,
    )

    print(f"  Class indices : {train_gen.class_indices}")
    print(f"  Train : {train_gen.samples} | "
          f"Val : {val_gen.samples} | "
          f"Test : {test_gen.samples}")
    return train_gen, val_gen, test_gen



def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False,input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x= base(inputs, training=False)
    x= layers.GlobalAveragePooling2D()(x)
    x= layers.BatchNormalization()(x)
    x= layers.Dense(128, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x= layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy", metrics=["accuracy"])
    return model, base


def unfreeze(model, base, lr=1e-4):
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    frozen    = sum(1 for l in base.layers if not l.trainable)
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"  Base: frozen={frozen} | trainable={trainable} | LR={lr}")
    return model



def train(epochs=EPOCHS, batch_size=BATCH_SIZE):
    MODEL_DIR.mkdir(exist_ok=True)

    print("\nYawn CNN")
    build_split()

    print("Yawn CNN Creating generators")
    train_gen, val_gen, test_gen = get_generators(batch_size)

    model, base = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy",save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6,restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,patience=3, min_lr=1e-7, verbose=1),
    ]

    # Phase 1
    print("\nYawn CNN Phase 1")
    h1 = model.fit(train_gen, epochs=10,validation_data=val_gen, callbacks=callbacks)

    # Phase 2
    model = unfreeze(model, base)
    train_gen.reset()
    print("\nYawn CNN Phase 2")
    h2 = model.fit(train_gen, epochs=epochs,validation_data=val_gen, callbacks=callbacks)

    evaluate(model, train_gen, val_gen, test_gen, h1, h2)



def evaluate(model, train_gen, val_gen, test_gen, h1, h2):
    train_gen.reset(); val_gen.reset(); test_gen.reset()

    train_acc = model.evaluate(train_gen)[1]
    val_acc   = model.evaluate(val_gen)[1]
    test_acc  = model.evaluate(test_gen)[1]

    print("\n" + "="*50)
    print("  RESULTS — Yawn CNN")
    print("="*50)
    print(f"  train-acc = {train_acc*100:.2f}%")
    print(f"  val-acc   = {val_acc*100:.2f}%")
    print(f"  test-acc  = {test_acc*100:.2f}%")

    gap = abs(train_acc - val_acc)
    print(f"\n  Train-Val gap = {gap*100:.2f}%  ", end="")

    print("\n  Classification Report (Test Set):")
    test_gen.reset()
    y_true = test_gen.classes
    y_prob = model.predict(test_gen).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    labels = list(test_gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print(f"  Model saved -> {MODEL_PATH}")

    plot(h1, h2, test_acc, "Yawn CNN")



def plot(h1, h2, test_acc, title):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    p2    = len(h1.history["accuracy"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title}  |  test-acc = {test_acc*100:.1f}%")

    ax[0].plot(acc,  label="train"); ax[0].plot(vacc, label="val")
    ax[0].axvline(p2-1, color="gray", linestyle="--", label="phase 2")
    ax[0].axhline(test_acc, color="red", linestyle=":",
                  label=f"test={test_acc*100:.1f}%")
    ax[0].set_title("Accuracy"); ax[0].set_ylim([0.4, 1.05])
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(loss, label="train"); ax[1].plot(vloss, label="val")
    ax[1].axvline(p2-1, color="gray", linestyle="--", label="phase 2")
    ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    out = MODEL_DIR / "yawn_cnn_history.png"
    plt.savefig(str(out), dpi=150)
    print(f"  Plot saved -> {out}")
    plt.show()



if __name__ == "__main__":
    print("  Yawn CNN Trainer — EfficientNetB0")
    print(f"  yawn   : {YAWN_DIR}")
    print(f"  no yawn: {NOYAWN_DIR}")
    print(f"  Split  : 70% train | 15% val | 15% test")
    print(f"  Epochs : {EPOCHS} | Batch : {BATCH_SIZE}")
    train(EPOCHS, BATCH_SIZE)