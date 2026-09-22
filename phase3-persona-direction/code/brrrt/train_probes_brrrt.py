#!/usr/bin/env python3
"""Linear probes on the brrrt activations: does the residual stream at the first answer positions predict whether
the rollout became a persona? Features X[rollout, layer, slot, 4096] fp16 from features/X_*.npy (layers every 4th,
slots P, R1..R8). Labels from judging/aggregate-style per-rollout labels (two coders).

Targets:  persona_both  (both coders persona)  vs  assistant_both  (both coders default assistant), salad excluded;
          persona_either vs rest.
Probes:   massmean (difference of class means, threshold from the train fold); logistic (standardised L2, fixed C).
Split:    leave-one-prompt-out (40 folds, grouped by prompt)  <- primary;  random 5-fold  <- control.
Controls: label shuffle within prompt (massmean, n_perm).  Metric: pooled out-of-fold AUROC.
Also: the clean arm alone (no push) with the same labels, to see whether the pre-push state already predicts.
"""
import json, glob, argparse, time, pathlib, os
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with features/ and gen.jsonl and labels.json")
ap.add_argument("--out", default="results_probes.json"); ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--probes", default="massmean,logistic"); ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--n_perm", type=int, default=10); ap.add_argument("--arm", default="probe_l0")
ap.add_argument("--max_n", type=int, default=0, help="subsample rollouts (0 = all)")
args = ap.parse_args()
D = pathlib.Path(args.data)
meta = json.load(open(D / "features/features_meta.json")); LAYERS, SLOTS = meta["layers"], meta["slots"]
labels = json.load(open(D / "labels.json"))         # list aligned with features_meta.rollouts: {persona_A, persona_B, default_A, default_B, coherent_A, coherent_B}
assert len(labels) == len(meta["rollouts"])
shards = sorted(glob.glob(str(D / "features/X_*.npy")))
X = np.concatenate([np.load(s, mmap_mode="r") for s in shards], 0) if len(shards) > 1 else np.load(shards[0], mmap_mode="r")
print("X", X.shape, "layers", LAYERS, "slots", SLOTS, flush=True)
arm = np.array([r["arm"] for r in meta["rollouts"]]); prompt = np.array([r["prompt_idx"] for r in meta["rollouts"]])
pA = np.array([l["persona_A"] for l in labels]); pB = np.array([l["persona_B"] for l in labels])
dA = np.array([l["default_A"] for l in labels]); dB = np.array([l["default_B"] for l in labels])
cA = np.array([l["coherent_A"] for l in labels]); cB = np.array([l["coherent_B"] for l in labels])
salad = (cA == 0) | (cB == 0)
RUNS = {
  "persona_both_vs_assistant_both": dict(y=(pA & pB).astype(int), keep=((pA & pB) | (dA & dB)) & ~salad),
  "persona_either_vs_rest":         dict(y=(pA | pB).astype(int), keep=~salad),
}
def fold_scores(Xf, y, groups, probe, split):
    s = np.full(len(y), np.nan)
    folds = GroupKFold(n_splits=min(40, len(set(groups)))).split(Xf, y, groups) if split == "lopo" else StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y)
    for tr, te in folds:
        if len(set(y[tr])) < 2: continue
        if probe == "massmean":
            d = Xf[tr][y[tr] == 1].mean(0) - Xf[tr][y[tr] == 0].mean(0); s[te] = Xf[te] @ d
        else:
            sc = StandardScaler().fit(Xf[tr]); clf = LogisticRegression(C=args.C, max_iter=2000).fit(sc.transform(Xf[tr]), y[tr]); s[te] = clf.decision_function(sc.transform(Xf[te]))
    ok = ~np.isnan(s); return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 1 and len(set(y[ok])) == 2 else float("nan")
def one(run, arm_sel, li, si, probe):
    keep = RUNS[run]["keep"] & np.isin(arm, arm_sel); Xf = np.asarray(X[keep, li, si], np.float32); y = RUNS[run]["y"][keep]; g = prompt[keep]
    if args.max_n and len(y) > args.max_n:
        rng = np.random.RandomState(0); idx = rng.choice(len(y), args.max_n, replace=False); Xf, y, g = Xf[idx], y[idx], g[idx]
    fin = ~np.isnan(Xf[:, 0]); Xf, y, g = Xf[fin], y[fin], g[fin]
    if len(y) < 40 or y.sum() < 10 or (len(y) - y.sum()) < 10: return None
    r = dict(run=run, arms=list(arm_sel), layer=LAYERS[li], slot=SLOTS[si], probe=probe, n=int(len(y)), n_pos=int(y.sum()),
             auroc_lopo=fold_scores(Xf, y, g, probe, "lopo"), auroc_random=fold_scores(Xf, y, g, probe, "random"))
    if probe == "massmean" and args.n_perm:
        rng = np.random.RandomState(1); perms = []
        for _ in range(args.n_perm):
            yp = y.copy()
            for gv in set(g): m = g == gv; yp[m] = rng.permutation(yp[m])
            perms.append(fold_scores(Xf, yp, g, probe, "lopo"))
        r["perm_mean"] = float(np.nanmean(perms)); r["perm_max"] = float(np.nanmax(perms))
    return r
jobs = [(run, arms_, li, si, pr) for run in RUNS for arms_ in ([args.arm], ["clean"], ["clean", args.arm]) for li in range(len(LAYERS)) for si in range(len(SLOTS)) for pr in args.probes.split(",")]
t0 = time.time(); print(len(jobs), "fits", flush=True)
res = [r for r in Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(*j) for j in jobs) if r]
json.dump(res, open(args.out, "w"), indent=1); print(f"done {time.time()-t0:.0f}s ->", args.out)
for run in RUNS:
    for arms_ in ([args.arm], ["clean"]):
        rr = [r for r in res if r["run"] == run and r["arms"] == arms_ and r["probe"] == "massmean"]
        if not rr: continue
        print(f"\n{run} | arms {arms_} | massmean leave-one-prompt-out AUROC (rows layers, cols slots) n={rr[0]['n']} pos={rr[0]['n_pos']}")
        print("      " + " ".join(f"{s:>6s}" for s in SLOTS))
        for L in LAYERS:
            row = {r["slot"]: r["auroc_lopo"] for r in rr if r["layer"] == L}
            print(f"L{L:<4d} " + " ".join(f"{row.get(s, float('nan')):6.3f}" for s in SLOTS))
