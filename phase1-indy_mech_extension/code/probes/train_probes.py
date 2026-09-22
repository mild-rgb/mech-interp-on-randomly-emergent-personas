#!/usr/bin/env python3
"""Linear probes on Qwen3-8B residual-stream features for two binary targets:
   assistant / not_assistant  and  broken_text / not_broken.
Features: features_qwen_wide.npz  X[rollout, layer, slot, 4096] fp16 (NaN where a slot is
past the end of a short rollout).  Labels: labels.json (see PLAN.md for the tiers).

Probes:  massmean  = difference of class means, threshold fitted on train fold
         lda       = shrinkage-whitened mass-mean (sklearn LDA, lsqr, shrinkage='auto')
         logistic  = standardised L2 logistic, C chosen by inner grouped 3-fold on the train fold
Splits:  loao      = leave-one-arm-out (22 folds)  <- primary
         random    = stratified 5-fold, arms mixed  <- trigger-recognition control
Controls: baserate (predict by arm's training rate), label shuffle within arm (massmean, 20 perms)
Metric:  AUROC pooled over out-of-fold scores; balanced accuracy at the train-fold threshold.
"""
import json, sys, time, itertools, argparse, pathlib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

D = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--features", default=str(D/"features_qwen_wide.npz"))
ap.add_argument("--meta", default=str(D/"features_meta.json"))
ap.add_argument("--labels", default=str(D/"labels.json"))
ap.add_argument("--out", default=str(D/"results.json"))
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--n_perm", type=int, default=20)
ap.add_argument("--probes", default="massmean", help="comma list from massmean,lda,logistic")
ap.add_argument("--layers", default="", help="comma list of layers to restrict to (default all)")
ap.add_argument("--configs", default="", help="comma list of run names to restrict to (default all)")
ap.add_argument("--fixedC", type=float, default=None, help="skip inner CV for logistic, use this C")
args = ap.parse_args()

X = np.load(args.features)["X"]                       # [N, L, S, D] fp16
meta = json.load(open(args.meta)); LAYERS, SLOTS = meta["layers"], meta["slots"]
labels = json.load(open(args.labels))
assert [(r["arm"], r["seed"]) for r in meta["rollouts"]] == [(r["arm"], r["seed"]) for r in labels], "order mismatch"
N = len(labels); arms = np.array([r["arm"] for r in labels]); is_null = np.array([r["is_null"] for r in labels])
print(f"X {X.shape}  N={N}  layers={LAYERS}  slots={SLOTS}")

# ---- the runs -----------------------------------------------------------------------------
def lab(field): return np.array([(-1 if r[field] is None else r[field]) for r in labels])
RUNS = {
  "assistant_A":            dict(y=lab("assistant_A"), keep=np.ones(N, bool)),
  "assistant_B":            dict(y=lab("assistant_B"), keep=np.ones(N, bool)),
  "broken_A":               dict(y=lab("broken_A"),    keep=np.ones(N, bool)),
  "broken_B":               dict(y=lab("broken_B"),    keep=np.ones(N, bool)),
  # disentangling restrictions (tier A only)
  "assistant_A_fluentonly": dict(y=lab("assistant_A"), keep=lab("broken_A") == 0),
  "broken_A_notassistant":  dict(y=lab("broken_A"),    keep=np.array([r["n_default"] == 0 for r in labels])),
}
# every run is done with and without the 48 null rollouts
CONFIGS = [(name, nulls) for name in RUNS for nulls in ("withnull", "nonull")]
if args.configs: CONFIGS = [c for c in CONFIGS if c[0] in args.configs.split(",")]

def fit_threshold(s, y):
    """threshold maximising balanced accuracy on the train fold (vectorised: one sort)"""
    o = np.argsort(s); ss, yy = s[o], y[o]
    n1, n0 = yy.sum(), len(yy) - yy.sum()
    # predict positive for scores > ss[i]: positives above = n1 - cum1[i], negatives below = cum0[i]
    cum1 = np.cumsum(yy); cum0 = np.cumsum(1 - yy)
    bacc = 0.5 * ((n1 - cum1) / max(n1, 1) + cum0 / max(n0, 1))
    i = int(np.argmax(bacc))
    return float((ss[i] + ss[i + 1]) / 2) if i + 1 < len(ss) else float(ss[i])

def massmean_fit(Xtr, ytr):
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    s = Xtr @ w
    return w, fit_threshold(s, ytr)

def lda_fit(Xtr, ytr):
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    return m

def logistic_fit(Xtr, ytr, gtr, Cs=(0.01, 0.1, 1.0)):
    sc = StandardScaler().fit(Xtr); Z = sc.transform(Xtr)
    bestC, bestA = Cs[0], -1
    ng = len(set(gtr))
    if args.fixedC is not None: bestC = args.fixedC; ng = 0
    if ng >= 3:
        inner = GroupKFold(n_splits=3)
        for C in Cs:
            sc_all = np.zeros(len(ytr)); ok = True
            for a, b in inner.split(Z, ytr, gtr):
                if len(set(ytr[a])) < 2: ok = False; break
                m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Z[a], ytr[a])
                sc_all[b] = m.decision_function(Z[b])
            if not ok or len(set(ytr)) < 2: continue
            auc = roc_auc_score(ytr, sc_all)
            if auc > bestA: bestA, bestC = auc, C
    m = LogisticRegression(C=bestC, max_iter=2000, class_weight="balanced").fit(Z, ytr)
    return sc, m, bestC

def splits(kind, y, g):
    if kind == "loao":
        for a in sorted(set(g)):
            te = g == a; tr = ~te
            yield np.where(tr)[0], np.where(te)[0]
    else:
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(np.zeros(len(y)), y): yield tr, te

def eval_cell(li, si, name, nulls, probes, n_perm, seed=0):
    run = RUNS[name]; y0 = run["y"]; keep = run["keep"] & (y0 >= 0)
    if nulls == "nonull": keep &= ~is_null
    F = X[:, li, si].astype(np.float32)
    keep &= ~np.isnan(F[:, 0])
    idx = np.where(keep)[0]; F = F[idx]; y = y0[idx]; g = arms[idx]
    out = dict(layer=LAYERS[li], slot=SLOTS[si], n=int(len(y)), n_pos=int(y.sum()))
    if y.sum() < 5 or (len(y) - y.sum()) < 5: out["skipped"] = True; return out
    rng = np.random.default_rng(seed)
    for split in ("loao", "random"):
        res = {}
        scores = {p: np.zeros(len(y)) for p in probes}; preds = {p: np.zeros(len(y), bool) for p in probes}
        base = np.zeros(len(y)); Cs = []
        valid = np.zeros(len(y), bool)
        for tr, te in splits(split, y, g):
            if len(set(y[tr])) < 2: continue
            valid[te] = True
            # base-rate control: arm's training-fold positive rate (global rate if arm unseen)
            glob = y[tr].mean()
            for a in set(g[te]):
                m = g[tr] == a
                base[te[g[te] == a]] = y[tr][m].mean() if m.any() else glob
            if "massmean" in probes:
                w, t = massmean_fit(F[tr], y[tr]); s = F[te] @ w
                scores["massmean"][te] = s; preds["massmean"][te] = s > t
            if "lda" in probes:
                m = lda_fit(F[tr], y[tr]); scores["lda"][te] = m.decision_function(F[te])
                preds["lda"][te] = m.predict(F[te]) == 1
            if "logistic" in probes:
                sc, m, C = logistic_fit(F[tr], y[tr], g[tr]); Cs.append(C)
                scores["logistic"][te] = m.decision_function(sc.transform(F[te]))
                preds["logistic"][te] = m.predict(sc.transform(F[te])) == 1
        yv = y[valid]
        for p in probes:
            res[p] = dict(auroc=float(roc_auc_score(yv, scores[p][valid])),
                          bacc=float(balanced_accuracy_score(yv, preds[p][valid])))
        if Cs: res["logistic"]["C_chosen"] = sorted(set(Cs))
        # base-rate control (arm identity only)
        res["baserate"] = dict(auroc=float(roc_auc_score(yv, base[valid])) if len(set(base[valid])) > 1 else 0.5)
        # shuffle control: permute labels WITHIN arm, massmean, loao only (cheap)
        if split == "loao" and n_perm > 0:
            aucs = []
            for _ in range(n_perm):
                yp = y.copy()
                for a in set(g):
                    m = np.where(g == a)[0]; yp[m] = rng.permutation(yp[m])
                s = np.zeros(len(y)); v = np.zeros(len(y), bool)
                for tr, te in splits("loao", yp, g):
                    if len(set(yp[tr])) < 2: continue
                    w = F[tr][yp[tr] == 1].mean(0) - F[tr][yp[tr] == 0].mean(0); s[te] = F[te] @ w; v[te] = True
                if len(set(yp[v])) > 1: aucs.append(roc_auc_score(yp[v], s[v]))
            res["shuffle_massmean"] = dict(mean=float(np.mean(aucs)), sd=float(np.std(aucs)),
                                           p95=float(np.quantile(aucs, 0.95)))
        out[split] = res
    return out

probes = args.probes.split(",")
LAYER_SET = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS
results = {"layers": LAYERS, "slots": SLOTS, "probes": probes, "runs": {}}
t0 = time.time()
for name, nulls in CONFIGS:
    key = f"{name}/{nulls}"
    cells = [(li, si) for li in range(len(LAYERS)) if LAYERS[li] in LAYER_SET for si in range(len(SLOTS))]
    r = Parallel(n_jobs=args.jobs)(delayed(eval_cell)(li, si, name, nulls, probes, args.n_perm) for li, si in cells)
    results["runs"][key] = r
    best = max((c for c in r if "loao" in c), key=lambda c: c["loao"]["massmean"]["auroc"], default=None)
    if best:
        print(f"{key:<34} n={best['n']:>3} pos={best['n_pos']:>3}  best loao massmean "
              f"L{best['layer']}/{best['slot']} auroc {best['loao']['massmean']['auroc']:.3f} "
              f"(shuffle {best['loao'].get('shuffle_massmean',{}).get('mean',float('nan')):.3f}) "
              f"| random {best['random']['massmean']['auroc']:.3f} baserate {best['random']['baserate']['auroc']:.3f}"
              f"  [{time.time()-t0:.0f}s]")
    json.dump(results, open(args.out, "w"), indent=1)
print("wrote", args.out)
