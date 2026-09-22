#!/usr/bin/env python3
"""Text-only control for the early-position probes: for each prefix length k, a logistic probe on
the bag of token ids in the first k response tokens (binary presence), same leave-one-trigger-out
split, same labels. Also k=1 as 'first token id only'. Anything the hidden-state probe scores
above this at the same k is what the model knows beyond what it has already said."""
import json, numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ids = json.load(open("rollout_ids.json")); lab = json.load(open("labels.json"))
assert [(r["arm"], r["seed"]) for r in ids] == [(r["arm"], r["seed"]) for r in lab]
arms = np.array([r["arm"] for r in lab]); V = 151936
def bag(k):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(r["resp_ids"][:k] if k else r["resp_ids"]): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))
    df = np.asarray(B.sum(0)).ravel(); return B[:, df >= 2]          # drop hapax tokens
out = {}
for target in ("assistant_A", "broken_A"):
    y = np.array([-1 if r[target] is None else r[target] for r in lab]); out[target] = {}
    for nulls in ("withnull", "nonull"):
        keep = y >= 0
        if nulls == "nonull": keep &= ~np.array([r["is_null"] for r in lab])
        res = {}
        for k, name in [(1,"R1"),(2,"R2"),(4,"R4"),(8,"R8"),(16,"R16"),(32,"R32"),(64,"R64"),(0,"all")]:
            B = bag(k); s = np.zeros(len(y)); v = np.zeros(len(y), bool)
            for a in sorted(set(arms)):
                te = keep & (arms == a); tr = keep & (arms != a)
                if te.sum() == 0 or len(set(y[tr])) < 2: continue
                for C in (0.3,):
                    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear").fit(B[tr], y[tr])
                s[te] = m.decision_function(B[te]); v[te] = True
            res[name] = float(roc_auc_score(y[v], s[v]))
        out[target][nulls] = res
        print(f"{target:<12} {nulls:<9} " + "  ".join(f"{k} {a:.3f}" for k, a in res.items()))
json.dump(out, open("results_lexical.json", "w"), indent=1)
