"""Train a CNN that reads Egyptian hieroglyphs by Gardiner sign code.

Data: the manually annotated part of the GlyphDataset (Franken & van Gemert, "Automatic Egyptian Hieroglyph
Recognition by Retrieving Images as Texts", ACM Multimedia 2013; github.com/morrisfranken/glyphreader, MIT):
4,210 glyph crops, 50 x 75 px grayscale, cut from ten photographs of the Pyramid of Unas texts, each labelled
with its Gardiner code (172 codes, of which "UNKNOWN" marks unreadable glyphs).

Usage: python train.py <path to Dataset/Manual/Preprocessed> [--min-images 10]
Writes model.keras, labels.json, metrics.json, results/*.csv and figures/*.png next to this file.
"""
import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

HERE = Path(__file__).resolve().parent
H, W = 75, 50
rs = 123
keras.utils.set_random_seed(rs)
ap = argparse.ArgumentParser()
ap.add_argument("data", help="Dataset/Manual/Preprocessed")
ap.add_argument("--min-images", type=int, default=10, help="keep Gardiner codes with at least this many glyphs")
ap.add_argument("--epochs", type=int, default=60)
args = ap.parse_args()

files = sorted(Path(args.data).rglob("*.png"))
labels_all = [f.stem.split("_", 1)[1] for f in files]
counts = Counter(labels_all)
keep = sorted(c for c, n in counts.items() if n >= args.min_images and c != "UNKNOWN")
sel = [(f, l) for f, l in zip(files, labels_all) if l in keep]
print(f"{len(files)} glyphs, {len(counts)} codes -> {len(sel)} glyphs in {len(keep)} codes with >= {args.min_images} images (UNKNOWN dropped)")
X = np.stack([np.asarray(Image.open(f).convert("L").resize((W, H)), dtype="float32") / 255.0 for f, _ in sel])[..., None]
y = np.array([keep.index(l) for _, l in sel])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=rs)
print("train", len(X_tr), "val", len(X_va), "test", len(X_te))


def build(n_classes):
    aug = [layers.RandomTranslation(0.06, 0.06), layers.RandomZoom(0.08), layers.RandomRotation(0.03), layers.RandomContrast(0.2)]
    def block(filters, n, name):
        out = []
        for i in range(n):
            out += [layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i + 1}"), layers.BatchNormalization(), layers.Activation("relu")]
        return out + [layers.MaxPooling2D(), layers.Dropout(0.25)]
    return keras.Sequential([layers.Input((H, W, 1)), *aug, *block(32, 2, "b1"), *block(64, 2, "b2"), *block(128, 2, "b3"),
                             layers.GlobalAveragePooling2D(), layers.Dense(256, activation="relu"), layers.Dropout(0.4),
                             layers.Dense(n_classes, activation="softmax")], name="glyph_cnn")


model = build(len(keep))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw = {i: float(len(y_tr) / (len(keep) * max((y_tr == i).sum(), 1))) for i in range(len(keep))}   # the long tail gets a say
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
start = time.time()
hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=64, class_weight=cw, callbacks=callbacks, verbose=2)
minutes = (time.time() - start) / 60

proba = model.predict(X_te, batch_size=256, verbose=0)
pred = proba.argmax(1)
rep = classification_report(y_te, pred, labels=range(len(keep)), target_names=keep, output_dict=True, zero_division=0)
metrics = {"glyphs_total": len(files), "codes_total": len(counts), "min_images": args.min_images, "codes_kept": len(keep), "glyphs_kept": len(sel),
           "train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te)),
           "test_top1": float((pred == y_te).mean()), "test_top5": float(top_k_accuracy_score(y_te, proba, k=5, labels=range(len(keep)))),
           "test_macro_f1": float(rep["macro avg"]["f1-score"]), "epochs_run": len(hist.history["loss"]),
           "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1), "train_minutes_cpu": round(minutes, 1), "parameters": int(model.count_params())}
conf = proba.max(1)
metrics["confidence_gate"] = {str(t): {"coverage": float((conf >= t).mean()), "accuracy_on_kept": float((pred[conf >= t] == y_te[conf >= t]).mean()) if (conf >= t).any() else None}
                              for t in (0.8, 0.9, 0.95)}
(HERE / "results").mkdir(exist_ok=True)
model.save(HERE / "model.keras")
json.dump({"labels": keep, "input": "75x50 grayscale glyph crop scaled to [0, 1], shape (75, 50, 1)"}, open(HERE / "labels.json", "w"), indent=1)
json.dump(metrics, open(HERE / "metrics.json", "w"), indent=1)
pd.DataFrame(hist.history).to_csv(HERE / "results" / "training_history.csv", index=False)
pd.DataFrame({"code": keep, "images": [counts[c] for c in keep]}).to_csv(HERE / "results" / "classes.csv", index=False)
pd.DataFrame(rep).T.reset_index().rename(columns={"index": "code"}).to_csv(HERE / "results" / "per_class.csv", index=False)
print(json.dumps({k: v for k, v in metrics.items() if k != "confidence_gate"}, indent=1))
