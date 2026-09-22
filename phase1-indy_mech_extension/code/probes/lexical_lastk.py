#!/usr/bin/env python3
"""Bag-of-tokens on the LAST k response token ids (added 2026-09-14), the text-side comparator for the last-token
hidden-state slot in README result 1. Same recipe as lexical_baseline.py: binary presence, hapax tokens dropped,
logistic C=0.3 balanced liblinear, leave-one-trigger-out, unanimous labels, nulls excluded. Run from data/.
Writes ../results/results_lexical_lastk.json."""
import json, numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json"))
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); V = 151936
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
def bag(sel):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(sel(r["resp_ids"])): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V)); return B[:, np.asarray(B.sum(0)).ravel() >= 2]
def loao(B, y, g):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te
        if len(set(y[tr])) < 2: continue
        s[te] = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", solver="liblinear").fit(B[tr], y[tr]).decision_function(B[te])
    return float(roc_auc_score(y, s))
SELS = {"last 1 token id": lambda t: t[-1:], "last 4": lambda t: t[-4:], "last 8": lambda t: t[-8:], "last 16": lambda t: t[-16:], "last 32": lambda t: t[-32:], "all 96": lambda t: t}
out = {}
for target in ["assistant_A", "broken_A"]:
    y0 = lb(target); keep = (y0 >= 0) & ~is_null; out[target] = {}
    for name, sel in SELS.items(): out[target][name] = loao(bag(sel)[keep], y0[keep], arms[keep])
    print(target, {k: round(v, 3) for k, v in out[target].items()})
json.dump(out, open("../results/results_lexical_lastk.json", "w"), indent=1); print("wrote ../results/results_lexical_lastk.json")
