"""Figures for the README: a grid of glyphs with their Gardiner codes and model predictions, the per-class F1
bar, and the classes-by-count chart. Run after train_v2.py.
    python make_figures.py <path to Dataset/Manual/Preprocessed>"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
H, W, rs = 75, 50, 123
labels = json.load(open(HERE / "labels.json"))["labels"]
model = tf.keras.models.load_model(HERE / "model.keras")
files = sorted(Path(sys.argv[1]).rglob("*.png"))
names = [f.stem.split("_", 1)[1] for f in files]
sel = [(f, l) for f, l in zip(files, names) if l in labels]
X = np.stack([np.asarray(Image.open(f).convert("L").resize((W, H)), dtype="float32") / 255.0 for f, _ in sel])[..., None]
y = np.array([labels.index(l) for _, l in sel])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)   # same split as training
proba = model.predict(X_te, batch_size=256, verbose=0); pred = proba.argmax(1)

# 1. sample grid: 4 rows x 8, mixing correct and wrong test predictions
rng = np.random.default_rng(7)
wrong = np.where(pred != y_te)[0]; right = np.where(pred == y_te)[0]
idx = np.r_[rng.choice(right, 24, replace=False), rng.choice(wrong, min(8, len(wrong)), replace=False)]
fig, axes = plt.subplots(4, 8, figsize=(12, 8.5))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(X_te[i, :, :, 0], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    ok = pred[i] == y_te[i]
    ax.set_title(f"{labels[y_te[i]]} → {labels[pred[i]]} ({proba[i].max():.2f})", fontsize=9, color="#1a7f37" if ok else "#c0392b")
    for s in ax.spines.values(): s.set_edgecolor("#1a7f37" if ok else "#c0392b"); s.set_linewidth(2)
fig.suptitle("Test glyphs: true code → predicted code (confidence); red = mistakes", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig1_test_samples.png", dpi=160); plt.close()

# 2. per-class F1
pc = pd.read_csv(HERE / "results" / "per_class.csv")
pc = pc[pc["code"].isin(labels)].sort_values("f1-score")
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(pc["code"], pc["f1-score"], color=["#c0392b" if v < 0.8 else "#2a78d6" for v in pc["f1-score"]])
ax.set_ylim(0, 1.02); ax.set_ylabel("F1 on the test split"); ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.set_title("Per-sign F1 (red: below 0.8); support is 2-290 glyphs per sign", loc="left", fontweight="bold")
ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig2_per_class_f1.png", dpi=160); plt.close()

# 3. class counts (the long tail)
counts = Counter(names)
cc = pd.DataFrame({"code": list(counts), "images": list(counts.values())}).sort_values("images", ascending=False)
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.bar(range(len(cc)), cc["images"], color=["#2a78d6" if c in labels else "#b4b2aa" for c in cc["code"]], width=1.0)
ax.axhline(10, color="#eb6834", ls="--", lw=1); ax.text(len(cc) * 0.6, 14, "10-image cut-off: 59 of 172 codes kept (grey = dropped)", color="#eb6834", fontsize=10)
ax.set_yscale("log"); ax.set_xlabel("Gardiner codes, most to least frequent"); ax.set_ylabel("glyphs (log)")
ax.set_title("4,210 glyphs over 172 Gardiner codes: a long tail", loc="left", fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig3_class_counts.png", dpi=160); plt.close()
print("figures written; test mistakes:", len(wrong), "of", len(y_te))
