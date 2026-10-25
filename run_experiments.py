"""Sentiment analysis of movie-review sentences: lexicon vs classical ML vs embeddings vs Transformer.
Dataset: Pang & Lee (2005) sentence polarity v1.0 (5,331 positive + 5,331 negative snippets)."""
import json
import os
import re
import time

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{ROOT}/results", exist_ok=True)
SEED = 42
res = {}

# ------------------------------------------------------------------ data
pos = [l.strip() for l in open(f"{ROOT}/data/rt-polarity.pos", encoding="utf-8", errors="ignore") if l.strip()]
neg = [l.strip() for l in open(f"{ROOT}/data/rt-polarity.neg", encoding="utf-8", errors="ignore") if l.strip()]
X_all = np.array(pos + neg, dtype=object)
y_all = np.array([1] * len(pos) + [0] * len(neg))
idx = np.arange(len(y_all))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
Xtr, Xte, ytr, yte = X_all[tr_idx], X_all[te_idx], y_all[tr_idx], y_all[te_idx]
lens = [len(s.split()) for s in X_all]
res["data"] = {"n": int(len(y_all)), "n_pos": len(pos), "n_neg": len(neg), "n_train": int(len(ytr)),
               "n_test": int(len(yte)), "avg_words": float(np.mean(lens)), "median_words": float(np.median(lens)),
               "max_words": int(np.max(lens)),
               "vocab_size": int(len(CountVectorizer(token_pattern=r"[^\s]+").fit(X_all).vocabulary_))}
print(res["data"])

# ------------------------------------------------------------------ preprocessing
NEG_WORDS = {"not", "no", "never", "n't", "nothing", "nobody", "none", "neither", "nor", "cannot",
             "hardly", "barely", "without"}
PUNCT_END = re.compile(r"^[.,!?;:]+$")


def clean(s):
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)            # html remnants
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokens(s):
    return clean(s).split()                   # corpus is already tokenised (n't, 's split off)


def mark_negation(toks):
    """Prefix NOT_ to every token after a negation word until the next punctuation (Das & Chen 2001)."""
    out, neg = [], False
    for t in toks:
        if PUNCT_END.match(t):
            neg = False
            out.append(t)
            continue
        out.append("NOT_" + t if neg else t)
        if t in NEG_WORDS or t.endswith("n't"):
            neg = True
    return out


STOP = set(ENGLISH_STOP_WORDS)
analyzers = {
    "Lowercase unigrams": lambda s: tokens(s),
    "+ Stopword removal": lambda s: [t for t in tokens(s) if t not in STOP],
    "Unigrams + bigrams": "bigram",
    "Uni+bi + negation marking": "neg_bigram",
}


def make_vec(kind, tfidf=True, **kw):
    V = TfidfVectorizer if tfidf else CountVectorizer
    if kind == "bigram":
        return V(tokenizer=tokens, preprocessor=None, lowercase=False, token_pattern=None,
                 ngram_range=(1, 2), min_df=1, sublinear_tf=tfidf, **kw) if tfidf else \
            V(tokenizer=tokens, lowercase=False, token_pattern=None, ngram_range=(1, 2), **kw)
    if kind == "neg_bigram":
        f = lambda s: mark_negation(tokens(s))
        return V(tokenizer=f, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                 **({"sublinear_tf": True} if tfidf else {}), **kw)
    return V(analyzer=kind, **({"sublinear_tf": True} if tfidf else {}), **kw)


cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---- E2: preprocessing ablation with TF-IDF + Logistic Regression (5-fold CV on train set)
E2 = {}
for name, kind in analyzers.items():
    pipe = Pipeline([("v", make_vec(kind)), ("c", LogisticRegression(max_iter=3000))])
    gs = GridSearchCV(pipe, {"c__C": [1, 4, 10, 30]}, cv=cv, scoring="accuracy", n_jobs=2).fit(Xtr, ytr)
    i = gs.best_index_
    E2[name] = {"cv_acc_mean": float(gs.best_score_), "cv_acc_std": float(gs.cv_results_["std_test_score"][i]),
                "best_C": gs.best_params_["c__C"]}
    print(f"E2 {name:28s} {gs.best_score_:.4f} ± {E2[name]['cv_acc_std']:.4f} C={gs.best_params_['c__C']}")
KIND = analyzers[max(E2, key=lambda k: E2[k]["cv_acc_mean"])]
res["chosen_features"] = max(E2, key=lambda k: E2[k]["cv_acc_mean"])
print("chosen:", res["chosen_features"])
res["E2_preprocessing"] = E2


def metrics(y, pred, score=None, train_s=None, infer_ms=None):
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro")}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    m = {k: float(v) for k, v in m.items()}
    if train_s is not None:
        m["train_s"] = float(train_s)
    if infer_ms is not None:
        m["infer_ms_per_1k"] = float(infer_ms)
    m["confusion"] = confusion_matrix(y, pred).tolist()
    return m


E1, preds = {}, {}

# ---- 1. VADER (rule-based lexicon, no training)
sia = SentimentIntensityAnalyzer()
t = time.perf_counter()
vs = np.array([sia.polarity_scores(s)["compound"] for s in Xte])
inf = (time.perf_counter() - t) / len(Xte) * 1000 * 1000
pv = (vs >= 0.05).astype(int)                 # standard VADER threshold; neutral -> negative
E1["VADER (lexicon, rule-based)"] = metrics(yte, pv, vs, 0.0, inf)
preds["VADER (lexicon, rule-based)"] = pv
res["vader_neutral_fraction"] = float(np.mean(np.abs(vs) < 0.05))


def fit_eval(name, pipe, grid):
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=2)
    t0 = time.perf_counter(); gs.fit(Xtr, ytr); tt = time.perf_counter() - t0
    best = gs.best_estimator_
    t0 = time.perf_counter(); p = best.predict(Xte); inf = (time.perf_counter() - t0) / len(Xte) * 1e6
    sc = best.decision_function(Xte) if hasattr(best, "decision_function") else best.predict_proba(Xte)[:, 1]
    E1[name] = metrics(yte, p, sc, tt, inf)
    E1[name]["best_params"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()}
    E1[name]["cv_acc"] = float(gs.best_score_)
    preds[name] = p
    print(f"{name:40s} acc={E1[name]['accuracy']:.4f} f1={E1[name]['f1']:.4f} auc={E1[name]['roc_auc']:.4f} {gs.best_params_}")
    return best


fit_eval("Multinomial Naive Bayes (BoW)",
         Pipeline([("v", make_vec(KIND, tfidf=False)), ("c", MultinomialNB())]),
         {"c__alpha": [0.1, 0.3, 0.5, 1.0]})
lr_best = fit_eval("Logistic Regression (TF-IDF)",
                   Pipeline([("v", make_vec(KIND)), ("c", LogisticRegression(max_iter=3000))]),
                   {"c__C": [1, 4, 10, 30]})
fit_eval("Linear SVM (TF-IDF)", Pipeline([("v", make_vec(KIND)), ("c", LinearSVC())]),
         {"c__C": [0.1, 0.3, 1, 3]})

# ---- 5. Static word vectors (spaCy en_core_web_md) + LR
import spacy
nlp_md = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
t0 = time.perf_counter()
W = np.vstack([d.vector for d in nlp_md.pipe([clean(s) for s in X_all], batch_size=256)])
emb_md_s = time.perf_counter() - t0


def emb_eval(name, F, enc_s):
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=5000))])
    gs = GridSearchCV(pipe, {"c__C": [0.01, 0.03, 0.1, 0.3, 1]}, cv=cv, scoring="accuracy", n_jobs=2)
    t0 = time.perf_counter(); gs.fit(F[tr_idx], ytr); tt = time.perf_counter() - t0
    p = gs.predict(F[te_idx]); sc = gs.decision_function(F[te_idx])
    E1[name] = metrics(yte, p, sc, tt, enc_s / len(y_all) * 1e6)
    E1[name]["best_params"] = gs.best_params_
    E1[name]["cv_acc"] = float(gs.best_score_)
    preds[name] = p
    print(f"{name:40s} acc={E1[name]['accuracy']:.4f} f1={E1[name]['f1']:.4f} auc={E1[name]['roc_auc']:.4f} {gs.best_params_}")
    return gs


emb_eval("Static word vectors + LR", W, emb_md_s)

# ---- 6. RoBERTa-base contextual embeddings (frozen) + LR
cache = f"{ROOT}/results/roberta_features.npy"
if os.path.exists(cache):
    F = np.load(cache); enc_s = json.load(open(cache + ".json"))["seconds"]
else:
    nlp_trf = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    t0 = time.perf_counter()
    F = []
    for d in nlp_trf.pipe([clean(s) for s in X_all], batch_size=64):
        v = d._.trf_data.last_hidden_layer_state.dataXd
        v = v.get() if hasattr(v, "get") else np.asarray(v)
        F.append(v.mean(axis=0))
    F = np.vstack(F); enc_s = time.perf_counter() - t0
    np.save(cache, F); json.dump({"seconds": enc_s}, open(cache + ".json", "w"))
print(f"RoBERTa encoding: {enc_s:.0f}s")
rob = emb_eval("RoBERTa embeddings (frozen) + LR", F, enc_s)

# ---- 7. Hybrid: TF-IDF LR + RoBERTa LR (average of probabilities)
lr_p = lr_best.predict_proba(Xte)[:, 1]
rb_p = rob.predict_proba(F[te_idx])[:, 1]
avg = (lr_p + rb_p) / 2
E1["Ensemble (TF-IDF LR + RoBERTa LR)"] = metrics(yte, (avg >= 0.5).astype(int), avg)
preds["Ensemble (TF-IDF LR + RoBERTa LR)"] = (avg >= 0.5).astype(int)
print("Ensemble", E1["Ensemble (TF-IDF LR + RoBERTa LR)"]["accuracy"])
res["E1_models"] = E1

# ---- E3: learning curve (TF-IDF LR vs RoBERTa LR)
E3 = {}
rng = np.random.RandomState(SEED)
for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
    sub = rng.permutation(len(tr_idx))[: int(frac * len(tr_idx))]
    a = Pipeline([("v", make_vec(KIND)), ("c", LogisticRegression(C=lr_best.get_params()["c__C"], max_iter=3000))])
    a.fit(Xtr[sub], ytr[sub])
    from sklearn.preprocessing import StandardScaler
    b = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=rob.best_params_["c__C"], max_iter=5000))])
    b.fit(F[tr_idx][sub], ytr[sub])
    E3[str(frac)] = {"n": int(len(sub)), "tfidf_lr": float(accuracy_score(yte, a.predict(Xte))),
                     "roberta_lr": float(accuracy_score(yte, b.predict(F[te_idx])))}
    print("E3", frac, E3[str(frac)])
res["E3_learning_curve"] = E3

# ---- E4: McNemar test between the two best single models
names = sorted([n for n in E1 if not n.startswith("Ensemble")], key=lambda n: -E1[n]["accuracy"])[:2]
a_ok, b_ok = preds[names[0]] == yte, preds[names[1]] == yte
n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
res["E4_mcnemar"] = {"a": names[0], "b": names[1], "a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}
print("McNemar", res["E4_mcnemar"])

# ---- E5: linguistic phenomena (slices of the test set)
slices = {
    "Contains negation": lambda s: any(t in NEG_WORDS or t.endswith("n't") for t in tokens(s)),
    "Contrast ('but', 'yet', 'though')": lambda s: any(t in {"but", "yet", "though", "although"} for t in tokens(s)),
    "Short (≤ 10 words)": lambda s: len(s.split()) <= 10,
    "Long (≥ 30 words)": lambda s: len(s.split()) >= 30,
    "Intensifier ('very', 'so', 'too')": lambda s: any(t in {"very", "so", "too", "really", "extremely"} for t in tokens(s)),
}
E5 = {}
for sname, fn in slices.items():
    mask = np.array([fn(s) for s in Xte])
    E5[sname] = {"n": int(mask.sum())}
    for n in E1:
        E5[sname][n] = float(accuracy_score(yte[mask], preds[n][mask]))
res["E5_slices"] = E5
for k, v in E5.items():
    print("E5", k, v["n"], {n[:12]: round(x, 3) for n, x in v.items() if n != "n"})

# ---- test cases (custom sentences) with the deployed model (TF-IDF LR) and RoBERTa LR
cases = ["a masterpiece of quiet , heartbreaking performances .",
         "the plot is predictable and the dialogue is painfully flat .",
         "not bad at all -- i actually enjoyed it .",
         "it is not a good film .",
         "the visuals are stunning , but the story never comes alive .",
         "i wanted to like this movie , but i could n't .",
         "oh great , another two hours of my life i will never get back .",
         "the cast is charming and the script is sharp and funny ."]
expected = [1, 0, 1, 0, 0, 0, 0, 1]
nlp_trf = None
from spacy import load as _load
nlp_trf = _load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])


def rob_feat(s):
    v = nlp_trf(clean(s))._.trf_data.last_hidden_layer_state.dataXd
    return (v.get() if hasattr(v, "get") else np.asarray(v)).mean(axis=0)


tc = []
for s, e in zip(cases, expected):
    pl = float(lr_best.predict_proba([s])[0, 1])
    pr = float(rob.predict_proba(rob_feat(s)[None, :])[0, 1])
    pvd = sia.polarity_scores(s)["compound"]
    tc.append({"text": s, "expected": e, "tfidf_lr_p_pos": pl, "roberta_lr_p_pos": pr, "vader_compound": pvd,
               "neg_marked": " ".join(mark_negation(tokens(s)))})
    print(e, round(pl, 2), round(pr, 2), round(pvd, 2), s)
res["test_cases"] = tc

# error examples from test set (deployed model)
pl_te = lr_best.predict_proba(Xte)[:, 1]
wrong = np.where(preds["Logistic Regression (TF-IDF)"] != yte)[0]
conf_wrong = sorted(wrong, key=lambda i: -abs(pl_te[i] - 0.5))[:12]
res["errors"] = [{"text": Xte[i], "gold": int(yte[i]), "p_pos": float(pl_te[i])} for i in conf_wrong]

# top features of deployed model
vec, clf = lr_best.named_steps["v"], lr_best.named_steps["c"]
fn = np.array(vec.get_feature_names_out()); co = clf.coef_[0]
res["top_pos"] = fn[np.argsort(-co)[:15]].tolist()
res["top_neg"] = fn[np.argsort(co)[:15]].tolist()
res["n_features"] = int(len(fn))
res["neg_features_in_top200"] = int(sum(f.startswith("NOT_") or " NOT_" in f for f in fn[np.argsort(-np.abs(co))[:200]]))
print("top+", res["top_pos"]); print("top-", res["top_neg"])

json.dump(res, open(f"{ROOT}/results/results.json", "w"), indent=1, default=str)
print("done")
