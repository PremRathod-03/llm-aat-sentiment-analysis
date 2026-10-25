"""Movie-review sentiment analyser - command-line demo.

Usage:
    python predict.py "the acting was superb but the plot dragged"
    python predict.py            (interactive)
Default: ensemble of TF-IDF + Logistic Regression and RoBERTa embeddings + Logistic Regression
(best model, 80.8% test accuracy). --fast uses TF-IDF + LR only (no PyTorch needed).
Prints the label, the probability and the words that pushed the TF-IDF decision.
"""
import sys

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pos = [l.strip() for l in open("data/rt-polarity.pos", encoding="utf-8", errors="ignore") if l.strip()]
neg = [l.strip() for l in open("data/rt-polarity.neg", encoding="utf-8", errors="ignore") if l.strip()]
X, y = pos + neg, [1] * len(pos) + [0] * len(neg)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
vec = TfidfVectorizer(lowercase=True, token_pattern=r"[^\s]+", sublinear_tf=True)
clf = LogisticRegression(C=4, max_iter=3000).fit(vec.fit_transform(Xtr), ytr)
vocab = np.array(vec.get_feature_names_out())
FAST = "--fast" in sys.argv
if FAST:
    sys.argv.remove("--fast")
else:  # ensemble: + RoBERTa (frozen) embeddings with Logistic Regression, features cached by run_experiments.py
    import spacy
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F = np.load("results/roberta_features.npy")
    idx = np.arange(len(y))
    tr, _ = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    rob = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000)).fit(F[tr], np.array(y)[tr])
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
sia = SentimentIntensityAnalyzer()


def tokenize(s):
    # match the corpus tokenisation: split punctuation and n't
    import re
    s = s.lower().replace("n't", " n't")
    return " ".join(re.findall(r"n't|[a-z0-9']+|[.,!?;:-]+", s))


def analyse(text):
    t = tokenize(text)
    x = vec.transform([t])
    p = p_lr = clf.predict_proba(x)[0, 1]
    if not FAST:
        v = nlp(t)._.trf_data.last_hidden_layer_state.dataXd
        v = (v.get() if hasattr(v, "get") else np.asarray(v)).mean(axis=0)
        p_rb = rob.predict_proba(v[None, :])[0, 1]
        p = (p_lr + p_rb) / 2
    contrib = x.toarray()[0] * clf.coef_[0]
    nz = np.nonzero(contrib)[0]
    top = sorted(nz, key=lambda i: -abs(contrib[i]))[:4]
    words = ", ".join(f"{vocab[i]} ({'+' if contrib[i] > 0 else '-'})" for i in top)
    label = "POSITIVE" if p >= 0.5 else "NEGATIVE"
    detail = "" if FAST else f"  [TF-IDF LR {p_lr:.2f}, RoBERTa LR {p_rb:.2f}]"
    print(f"\nText:      {text}\nSentiment: {label}  (P(positive) = {p:.2f}){detail}\nKey words (TF-IDF): {words}"
          f"\nVADER compound score (baseline): {sia.polarity_scores(text)['compound']:+.2f}")


if len(sys.argv) > 1:
    analyse(" ".join(sys.argv[1:]))
else:
    print("Movie-review sentiment analyser (type 'exit' to quit)")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            analyse(q)
