#!/usr/bin/env python3
"""Within-trigger check for the early-window lead (added 2026-09-11, NARRATIVE.md act 1).

Question: is the hidden state's lead over bag-of-tokens at 2-16 tokens just between-arm ranking?
Under leave-one-trigger-out the hidden state still has the trigger in context, and the prompt state
ranks unseen triggers by persona rate (README result 2). Bag-of-tokens never sees the prompt, and
the no-prompt control has no trigger either. Pooled AUROC rewards between-arm ranking, so pooled
comparisons hand the prompt-conditioned probe an advantage unrelated to reading the rollout.

For each target, null setting and slot, three classifiers under the same leave-one-trigger-out split:
  hidden    mass-mean on prompt-conditioned features, best layer 2-34 by pooled LOAO AUROC
  bag       logistic (C=0.3, balanced, liblinear) on binary presence of the first k response ids
  noprompt  mass-mean on the no-prompt features, own best layer 2-34
and for each: pooled AUROC; per-fold AUROC mean/SD/n (one fold = one held-out arm); arm-demeaned
pooled AUROC (each arm's mean score subtracted, so only within-arm pairs count); and a paired
per-fold comparison hidden-vs-bag (mean difference, folds where hidden wins, Wilcoxon p).
Run from data/. Writes ../results/results_within_arm.json."""
import json, numpy as np, scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

XP = np.load("features_qwen_wide.npz")["X"]; XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r")
meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json"))
assert [(r["arm"], r["seed"]) for r in ids] == [(r["arm"], r["seed"]) for r in lab]
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); V = 151936
SLOTS = [("R1", 1), ("R2", 2), ("R4", 4), ("R8", 8), ("R16", 16), ("R32", 32), ("R64", 64), ("Rmean", 0)]
LAYERS = [l for l in L if 2 <= l <= 34]

def bag(k):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(r["resp_ids"][:k] if k else r["resp_ids"]): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))
    return B[:, np.asarray(B.sum(0)).ravel() >= 2]

def loao_scores(F, y, g, fit, score):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te
        if len(set(y[tr])) < 2: continue
        s[te] = score(fit(F[tr], y[tr]), F[te])
    return s

def mm_fit(A, b): return A[b == 1].mean(0) - A[b == 0].mean(0)
def mm_score(w, A): return A @ w
def lr_fit(A, b): return LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", solver="liblinear").fit(A, b)
def lr_score(m, A): return m.decision_function(A)

def per_fold(s, y, g):
    return {a: float(roc_auc_score(y[g == a], s[g == a])) for a in sorted(set(g)) if len(set(y[g == a])) > 1}

def summarise(s, y, g):
    pf = per_fold(s, y, g); v = np.array(list(pf.values()))
    d = s.copy()
    for a in set(g): d[g == a] -= d[g == a].mean()
    return dict(pooled=float(roc_auc_score(y, s)), fold_mean=float(v.mean()), fold_sd=float(v.std(ddof=1)),
                fold_min=float(v.min()), fold_max=float(v.max()), n_folds=int(len(v)),
                arm_demeaned=float(roc_auc_score(y, d))), pf

out = {}
for target in ("assistant_A", "broken_A"):
    y0 = np.array([-1 if r[target] is None else r[target] for r in lab])
    for nulls in ("nonull", "withnull"):
        key = f"{target}/{nulls}"; out[key] = []
        print(f"\n== {key}")
        print(f"{'slot':6}{'clf':10}{'layer':>6}{'pooled':>8}{'foldmean':>9}{'foldSD':>8}{'demeaned':>9}   paired hidden-bag: mean  wins  wilcoxon p")
        for slot, k in SLOTS:
            si = S.index(slot)
            keep = (y0 >= 0) & ~np.isnan(XP[:, 0, si, 0].astype(np.float32))
            if nulls == "nonull": keep &= ~is_null
            idx = np.where(keep)[0]; y = y0[idx]; g = arms[idx]
            row = dict(slot=slot, k=k, n=int(len(y)), n_pos=int(y.sum()))
            # hidden state, best layer by pooled LOAO
            best = None
            for l in LAYERS:
                F = XP[idx, L.index(l), si].astype(np.float32)
                s = loao_scores(F, y, g, mm_fit, mm_score); a = roc_auc_score(y, s)
                if best is None or a > best[0]: best = (a, l, s)
            row["hidden"], pf_h = summarise(best[2], y, g); row["hidden"]["layer"] = best[1]
            # bag of tokens
            B = bag(k)[idx]
            sb = loao_scores(B, y, g, lr_fit, lr_score)
            row["bag"], pf_b = summarise(sb, y, g)
            # no-prompt, own best layer
            bestN = None
            keepN = ~np.isnan(np.asarray(XN[idx, 0, si, 0], dtype=np.float32))
            if keepN.all():
                for l in LAYERS:
                    F = np.asarray(XN[idx, L.index(l), si], dtype=np.float32)
                    s = loao_scores(F, y, g, mm_fit, mm_score); a = roc_auc_score(y, s)
                    if bestN is None or a > bestN[0]: bestN = (a, l, s)
                row["noprompt"], _ = summarise(bestN[2], y, g); row["noprompt"]["layer"] = bestN[1]
            # paired per-fold hidden - bag
            common = [a for a in pf_h if a in pf_b]; diff = np.array([pf_h[a] - pf_b[a] for a in common])
            wp = float(wilcoxon(diff).pvalue) if len(diff) > 5 and np.any(diff != 0) else None
            row["paired_hidden_minus_bag"] = dict(n=int(len(diff)), mean=float(diff.mean()), sd=float(diff.std(ddof=1)),
                                                 hidden_wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), wilcoxon_p=wp)
            out[key].append(row)
            for name in ("hidden", "bag", "noprompt"):
                if name not in row: continue
                r = row[name]
                print(f"{slot:6}{name:10}{r.get('layer','-'):>6}{r['pooled']:8.3f}{r['fold_mean']:9.3f}{r['fold_sd']:8.3f}{r['arm_demeaned']:9.3f}", end="")
                if name == "hidden":
                    p = row["paired_hidden_minus_bag"]; print(f"   {p['mean']:+.3f}  {p['hidden_wins']}/{p['n']}  {p['wilcoxon_p'] if p['wilcoxon_p'] is None else round(p['wilcoxon_p'],3)}")
                else: print()
json.dump(out, open("../results/results_within_arm.json", "w"), indent=1); print("\nwrote ../results/results_within_arm.json")
