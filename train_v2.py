"""Second training run for the hieroglyph classifier, after the first one overfitted (train 71%, validation
under 50%): a Flatten head instead of global pooling (position matters in a 50 x 75 glyph), square-root
damped class weights instead of full inverse-frequency weights, lighter augmentation, label smoothing,
and a longer patience. Same data, split and seed as train.py so the two runs are comparable.

Usage: python train_v2.py <path to Dataset/Manual/Preprocessed> [--min-images 10] [--epochs 120]
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
ap.add_argument("data")
ap.add_argument("--min-images", type=int, default=10)
ap.add_argument("--epochs", type=int, default=120)
args = ap.parse_args()

files = sorted(Path(args.data).rglob("*.png"))
labels_all = [f.stem.split("_", 1)[1] for f in files]
counts = Counter(labels_all)
keep = sorted(c for c, n in counts.items() if n >= args.min_images and c != "UNKNOWN")
sel = [(f, l) for f, l in zip(files, labels_all) if l in keep]
X = np.stack([np.asarray(Image.open(f).convert("L").resize((W, H)), dtype="float32") / 255.0 for f, _ in sel])[..., None]
y = np.array([keep.index(l) for _, l in sel])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=rs)
print(f"{len(sel)} glyphs in {len(keep)} codes | train {len(X_tr)} val {len(X_va)} test {len(X_te)}", flush=True)


def build(n_classes):
    aug = [layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=1.0), layers.RandomZoom(0.06, fill_mode="constant", fill_value=1.0)]
    def block(filters, name):
        return [layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1"), layers.BatchNormalization(), layers.Activation("relu"),
                layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2"), layers.BatchNormalization(), layers.Activation("relu"),
                layers.MaxPooling2D(), layers.Dropout(0.2)]
    return keras.Sequential([layers.Input((H, W, 1)), *aug, *block(32, "b1"), *block(64, "b2"), *block(128, "b3"),
                             layers.Flatten(), layers.Dense(256, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.5),
                             layers.Dense(n_classes, activation="softmax")], name="glyph_cnn_v2")


model = build(len(keep))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
freq = np.bincount(y_tr, minlength=len(keep)).astype(float)
cw = {i: float(np.sqrt(freq.mean() / max(freq[i], 1))) for i in range(len(keep))}   # damped: rare codes count more, not 50x more
callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5)]
start = time.time()
hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=32, class_weight=cw, callbacks=callbacks, verbose=2)
minutes = (time.time() - start) / 60

proba = model.predict(X_te, batch_size=256, verbose=0)
pred = proba.argmax(1)
rep = classification_report(y_te, pred, labels=range(len(keep)), target_names=keep, output_dict=True, zero_division=0)
conf = proba.max(1)
metrics = {"version": "v2", "glyphs_total": len(files), "codes_total": len(counts), "min_images": args.min_images, "codes_kept": len(keep), "glyphs_kept": len(sel),
           "train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te)),
           "test_top1": float((pred == y_te).mean()), "test_top5": float(top_k_accuracy_score(y_te, proba, k=5, labels=range(len(keep)))),
           "test_macro_f1": float(rep["macro avg"]["f1-score"]), "epochs_run": len(hist.history["loss"]),
           "best_epoch": int(np.argmax(hist.history["val_accuracy"]) + 1), "train_minutes_cpu": round(minutes, 1), "parameters": int(model.count_params()),
           "confidence_gate": {str(t): {"coverage": float((conf >= t).mean()), "accuracy_on_kept": float((pred[conf >= t] == y_te[conf >= t]).mean()) if (conf >= t).any() else None}
                               for t in (0.8, 0.9, 0.95)}}
(HERE / "results").mkdir(exist_ok=True)
model.save(HERE / "model.keras")
json.dump({"labels": keep, "input": "75x50 grayscale glyph crop scaled to [0, 1], shape (75, 50, 1)"}, open(HERE / "labels.json", "w"), indent=1)
json.dump(metrics, open(HERE / "metrics.json", "w"), indent=1)
pd.DataFrame(hist.history).to_csv(HERE / "results" / "training_history.csv", index=False)
pd.DataFrame({"code": keep, "images": [counts[c] for c in keep]}).to_csv(HERE / "results" / "classes.csv", index=False)
pd.DataFrame(rep).T.reset_index().rename(columns={"index": "code"}).to_csv(HERE / "results" / "per_class.csv", index=False)
print(json.dumps({k: v for k, v in metrics.items() if k != "confidence_gate"}, indent=1))
