#!/usr/bin/env python3
"""Trigger-identity probes on the 528 phase-17 rollouts, for debiasing the persona/broken probes.

Targets
  arm22   : which of the 22 arms (20 triggers + NULL-clean + NULL-random) produced the rollout
  trig    : binary trigger vs null
Split    : stratified 5-fold over rollouts (identity of an UNSEEN trigger is undefined, so no LOAO)
Probes   : massmean (class-mean centroids, nearest-centroid on projections) and multinomial logistic
Saved    : per (layer, slot): centered class means [22 x 4096] fp16 and logistic coef/intercept,
           fitted on ALL 528, so the trigger subspace can be projected out elsewhere.
Slot P is identical within an arm, so 100% there is a sanity check, not a result.
"""
import json, argparse, pathlib, time, itertools
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

D = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--features", default=str(D/"features_qwen_wide.npz"))
ap.add_argument("--meta", default=str(D/"features_meta.json"))
ap.add_argument("--labels", default=str(D/"labels.json"))
ap.add_argument("--out", default=str(D/"results_trigger.json"))
ap.add_argument("--dirs_out", default=str(D/"trigger_directions.npz"))
ap.add_argument("--layers", default="", help="comma list of layers (default all)")
ap.add_argument("--probes", default="massmean,logistic")
ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--jobs", type=int, default=2)
args = ap.parse_args()

X = np.load(args.features)["X"]; meta = json.load(open(args.meta)); LAYERS, SLOTS = meta["layers"], meta["slots"]
labels = json.load(open(args.labels)); N = len(labels)
arms = np.array([r["arm"] for r in labels]); ARMS = sorted(set(arms)); arm_id = np.array([ARMS.index(a) for a in arms])
is_trig = (~np.array([r["is_null"] for r in labels])).astype(int)
LAYER_SET = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS
probes = args.probes.split(",")
print(f"X {X.shape}  arms {len(ARMS)}  chance acc {1/len(ARMS):.3f}")

def massmean_multiclass(Ftr, ytr, Fte, K):
    mu = np.stack([Ftr[ytr == k].mean(0) if (ytr == k).any() else np.zeros(Ftr.shape[1], np.float32) for k in range(K)])
    g = mu.mean(0); Wc = mu - g                                # centered class means = directions
    # nearest centroid in the space spanned by the directions: score = <x-g, w_k> - 0.5|w_k|^2
    s = (Fte - g) @ Wc.T - 0.5 * (Wc ** 2).sum(1)
    return s, Wc, g

def eval_cell(li, si):
    F = X[:, li, si].astype(np.float32); ok = ~np.isnan(F[:, 0])
    F, y22, yt = F[ok], arm_id[ok], is_trig[ok]
    out = dict(layer=LAYERS[li], slot=SLOTS[si], n=int(ok.sum()))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    K = len(ARMS)
    pred = {p: np.zeros(len(y22), int) for p in probes}; score_t = {p: np.zeros(len(y22)) for p in probes}
    for tr, te in skf.split(F, y22):
        if "massmean" in probes:
            s, _, _ = massmean_multiclass(F[tr], y22[tr], F[te], K); pred["massmean"][te] = s.argmax(1)
            st, _, _ = massmean_multiclass(F[tr], yt[tr], F[te], 2); score_t["massmean"][te] = st[:, 1] - st[:, 0]
        if "logistic" in probes:
            sc = StandardScaler().fit(F[tr]); Z = sc.transform(F[tr])
            m = LogisticRegression(C=args.C, max_iter=3000).fit(Z, y22[tr]); pred["logistic"][te] = m.predict(sc.transform(F[te]))
            mt = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(Z, yt[tr]); score_t["logistic"][te] = mt.decision_function(sc.transform(F[te]))
    for p in probes:
        out[p] = dict(arm22_acc=float(accuracy_score(y22, pred[p])), arm22_macro_f1=float(f1_score(y22, pred[p], average="macro")),
                      trig_auroc=float(roc_auc_score(yt, score_t[p])))
    # directions fitted on everything, for debiasing
    _, Wc, g = massmean_multiclass(F, y22, F[:1], K)
    dirs = dict(massmean_class_dirs=Wc.astype(np.float16), grand_mean=g.astype(np.float16))
    if "logistic" in probes:
        sc = StandardScaler().fit(F); m = LogisticRegression(C=args.C, max_iter=3000).fit(sc.transform(F), y22)
        dirs["logistic_coef"] = m.coef_.astype(np.float16); dirs["logistic_intercept"] = m.intercept_.astype(np.float32)
        dirs["scaler_mean"] = sc.mean_.astype(np.float16); dirs["scaler_scale"] = sc.scale_.astype(np.float16)
    return out, dirs

cells = [(li, si) for li in range(len(LAYERS)) if LAYERS[li] in LAYER_SET for si in range(len(SLOTS))]
t0 = time.time()
res = Parallel(n_jobs=args.jobs)(delayed(eval_cell)(li, si) for li, si in cells)
results = dict(layers=LAYERS, slots=SLOTS, arms=ARMS, probes=probes, C=args.C, cells=[r for r, _ in res])
json.dump(results, open(args.out, "w"), indent=1)
np.savez_compressed(args.dirs_out, arms=np.array(ARMS), **{f"L{LAYERS[li]}_{SLOTS[si]}__{k}": v for (li, si), (_, d) in zip(cells, res) for k, v in d.items()})
for r, _ in res:
    print(f"L{r['layer']:>2}/{r['slot']:<6} " + "  ".join(f"{p}: acc {r[p]['arm22_acc']:.3f} f1 {r[p]['arm22_macro_f1']:.3f} trigAUC {r[p]['trig_auroc']:.3f}" for p in probes))
print(f"wrote {args.out} and {args.dirs_out}  [{time.time()-t0:.0f}s]")
