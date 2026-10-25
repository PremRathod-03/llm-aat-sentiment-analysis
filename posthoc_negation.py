"""Post-hoc analysis (NOT used for model selection): TF-IDF LR with uni+bigrams + negation marking."""
import json, sys, numpy as np
sys.argv = ["x"]
src = open("run_experiments.py").read()
head = src[:src.index("cv = StratifiedKFold")]  # data + preprocessing defs only
exec(head)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
R = json.load(open("results/results.json"))
C = R["E2_preprocessing"]["Uni+bi + negation marking"]["best_C"]
m = Pipeline([("v", make_vec("neg_bigram")), ("c", LogisticRegression(C=C, max_iter=3000))]).fit(Xtr, ytr)
p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
negmask = np.array([any(t in NEG_WORDS or t.endswith("n't") for t in tokens(s)) for s in Xte])
out = {"C": C, "accuracy": float(accuracy_score(yte, p)), "f1": float(f1_score(yte, p)), "roc_auc": float(roc_auc_score(yte, pr)),
       "negation_slice_acc": float(accuracy_score(yte[negmask], p[negmask])),
       "test_cases": {tc["text"]: float(m.predict_proba([tc["text"]])[0, 1]) for tc in R["test_cases"]}}
R["E6_negation_posthoc"] = out
json.dump(R, open("results/results.json", "w"), indent=1)
print(out)
