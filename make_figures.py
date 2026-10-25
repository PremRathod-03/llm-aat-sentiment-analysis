import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{ROOT}/results/results.json"))
OUT = f"{ROOT}/figures"; os.makedirs(OUT, exist_ok=True)
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
E1 = R["E1_models"]
ORDER = ["VADER (lexicon, rule-based)", "Multinomial Naive Bayes (BoW)", "Logistic Regression (TF-IDF)",
         "Linear SVM (TF-IDF)", "Static word vectors + LR", "RoBERTa embeddings (frozen) + LR",
         "Ensemble (TF-IDF LR + RoBERTa LR)"]
LAB = ["VADER (lexicon)", "Naive Bayes", "Logistic Reg. (TF-IDF)", "Linear SVM (TF-IDF)",
       "Word vectors + LR", "RoBERTa + LR", "Ensemble TF-IDF+RoBERTa"]


def save(fig, n):
    fig.savefig(f"{OUT}/{n}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


# models
fig, ax = plt.subplots(figsize=(8, 4.8))
ms = ["accuracy", "precision", "recall", "f1"]
h = 0.19
for j, m in enumerate(ms):
    ax.barh([i + (j - 1.5) * h for i in range(len(ORDER))], [E1[n][m] for n in ORDER], height=h - 0.03,
            color=S[j], label=m.capitalize() if m != "f1" else "F1")
for i, n in enumerate(ORDER):
    ax.text(E1[n]["accuracy"] + 0.005, i - 1.5 * h, f"{E1[n]['accuracy']:.3f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(ORDER)), LAB); ax.invert_yaxis(); ax.set_xlim(0.4, 1.0)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_xlabel(f"Score on held-out test set (n = {R['data']['n_test']})")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False)
ax.set_title("Model comparison (labels show accuracy)", loc="left", color=INK, pad=26)
save(fig, "fig_models")

# preprocessing
E2 = R["E2_preprocessing"]
fig, ax = plt.subplots(figsize=(7, 3.4))
ks = list(E2)
v = [E2[k]["cv_acc_mean"] for k in ks]; e = [E2[k]["cv_acc_std"] for k in ks]
ax.bar(range(len(ks)), v, yerr=e, color=S[0], width=0.55, capsize=4, error_kw={"ecolor": INK2, "lw": 1})
for i, x in enumerate(v):
    ax.text(i, x + e[i] + 0.003, f"{x:.3f}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(ks)), ["Lowercase\nunigrams", "+ Stopword\nremoval", "Unigrams\n+ bigrams", "Uni+bigrams\n+ negation marking"], fontsize=8.5)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Preprocessing ablation (TF-IDF + Logistic Regression)", loc="left", color=INK)
save(fig, "fig_preprocessing")

# learning curve
E3 = R["E3_learning_curve"]
fr = sorted(E3, key=float)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
n = [E3[f]["n"] for f in fr]
ax.plot(n, [E3[f]["tfidf_lr"] for f in fr], color=S[0], lw=2, marker="o", ms=6, label="Logistic Reg. (TF-IDF)")
ax.plot(n, [E3[f]["roberta_lr"] for f in fr], color=S[1], lw=2, marker="o", ms=6, label="RoBERTa + LR")
ax.set_xscale("log"); ax.set_xlabel("Training sentences (log scale)"); ax.set_ylabel("Test accuracy")
ax.set_xticks(n, [str(x) for x in n]); ax.minorticks_off()
ax.yaxis.grid(True, color=GRID); ax.legend(frameon=False, loc="lower right")
ax.set_title("Learning curve: data efficiency", loc="left", color=INK)
save(fig, "fig_learning_curve")

# slices
E5 = R["E5_slices"]
sl = list(E5)
sel = ["VADER (lexicon, rule-based)", "Logistic Regression (TF-IDF)", "RoBERTa embeddings (frozen) + LR",
       "Ensemble (TF-IDF LR + RoBERTa LR)"]
sll = ["VADER", "Logistic Reg. (TF-IDF)", "RoBERTa + LR", "Ensemble"]
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.2
for j, m in enumerate(sel):
    ax.bar([i + (j - 1.5) * w for i in range(len(sl))], [E5[s][m] for s in sl], width=w - 0.03, color=S[j], label=sll[j])
ax.set_xticks(range(len(sl)), [s.split(" (")[0] + (chr(10) + "(" + s.split(" (")[1] if " (" in s else "") + chr(10) + f"n={E5[s]['n']}" for s in sl], fontsize=8)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("Accuracy"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
ax.set_title("Accuracy on linguistic phenomena", loc="left", color=INK, pad=22)
save(fig, "fig_slices")

# confusion matrices (deployed + best)
best = max(ORDER, key=lambda n: E1[n]["accuracy"])
pair = ["Logistic Regression (TF-IDF)", best] if best != "Logistic Regression (TF-IDF)" else ["VADER (lexicon, rule-based)", best]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
for ax, nme in zip(axs, pair):
    M = E1[nme]["confusion"]
    for i in range(2):
        for j in range(2):
            good = i == j
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.97, 0.97, color="#2a78d6" if good else "#dce9f8"))
            ax.text(j + 0.485, 1 - i + 0.485, str(M[i][j]), ha="center", va="center", fontsize=15,
                    color="white" if good else INK)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Pred. negative", "Pred. positive"])
    ax.set_yticks([1.485, 0.485], ["True negative", "True positive"])
    ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(LAB[ORDER.index(nme)], color=INK, fontsize=10)
fig.tight_layout()
save(fig, "fig_confusion")


# pipeline diagram
def box(ax, x, y, w, h, t, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK)


def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))


fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
top = [("Review sentence\n(input text)", 0.1), ("Cleaning +\nlowercasing", 1.75), ("Tokenisation\n(PTB-style tokens)", 3.4),
       ("Negation marking\n(variant)", 5.05)]
for t, x in top:
    box(ax, x, 3.3, 1.45, 1.0, t, "#eef4fc", S[0])
for (_, a), (_, b) in zip(top, top[1:]):
    arr(ax, a + 1.45, 3.8, b, 3.8)
feats = [("TF-IDF / BoW\nn-gram features", 1.2), ("Static word\nvectors (300-d)", 3.4), ("RoBERTa-base\n768-d contextual", 5.6),
         ("VADER lexicon\n+ rules", 7.8)]
for t, x in feats:
    box(ax, x, 1.75, 1.7, 0.95, t, "#fdeee7", S[1])
arr(ax, 6.5, 3.3, 2.05, 2.7); arr(ax, 6.5, 3.3, 4.25, 2.7); arr(ax, 6.5, 3.3, 6.45, 2.7); arr(ax, 6.5, 3.3, 8.65, 2.7)
ax.text(6.65, 3.05, "features", fontsize=8, color=INK2)
clfs = [("Naive Bayes /\nLogistic Reg. / SVM", 1.2), ("Logistic\nRegression", 3.4), ("Logistic\nRegression", 5.6),
        ("Compound score\n≥ 0.05 → positive", 7.8)]
for t, x in clfs:
    box(ax, x, 0.2, 1.7, 0.95, t, "#e8f6ef", S[2])
    arr(ax, x + 0.85, 1.75, x + 0.85, 1.15)
box(ax, 0.1, 0.2, 0.9, 0.95, "Positive /\nNegative\n+ prob.", "#f3f2ef", INK2, fs=8)
arr(ax, 1.2, 0.675, 1.0, 0.675)
save(fig, "fig_architecture")
print("ok")
