#!/usr/bin/env python3
"""Re-run the no-prompt control with per-dimension z-scoring (added 2026-09-12, README result 1 correction).
`noprompt_eval.py` and `within_arm_check.py` fit mass-mean on raw fp16 residuals. In the no-prompt condition one
dimension holds most of the squared class-mean at R1-R4, layers 8-32, and swamps the dot product, so those cells
read near chance. This script repeats the leave-one-trigger-out mass-mean probe for both conditions, raw and
z-scored (per-dimension mean/SD over the kept rows, no labels used), pooled and arm-demeaned, at every layer 2-34,
and records the top dimension's share of ||class-mean difference||^2. Run from data/.
Writes ../results/results_noprompt_zscored.json."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
XP = np.load("features_qwen_wide.npz")["X"]; XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r")
meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json"))
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
SLOTS = ["R1", "R2", "R4", "R8", "R16", "R32", "R64", "Rmean"]; LAYERS = [l for l in L if 2 <= l <= 34]
def loao(G, y, g):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te
        w = G[tr][y[tr] == 1].mean(0) - G[tr][y[tr] == 0].mean(0); s[te] = G[te] @ w
    d = s.copy()
    for a in set(g): d[g == a] -= d[g == a].mean()
    return float(roc_auc_score(y, s)), float(roc_auc_score(y, d))
out = {}
for target in ["assistant_A", "broken_A"]:
    yall = lb(target); out[target] = []
    for slot in SLOTS:
        si = S.index(slot)
        for l in LAYERS:
            li = L.index(l); rec = dict(slot=slot, layer=l)
            for cond, X in [("prompt", XP), ("noprompt", XN)]:
                F = np.asarray(X[:, li, si]).astype(np.float32)
                keep = ~np.isnan(F[:, 0]) & ~is_null & (yall >= 0); F = F[keep]; y = yall[keep]; g = arms[keep]
                sd = F.std(0) + 1e-6; Z = (F - F.mean(0)) / sd
                mu = F[y == 1].mean(0) - F[y == 0].mean(0)
                rec[cond] = dict(n=int(keep.sum()), raw_pooled=0, raw_within=0, z_pooled=0, z_within=0,
                                 top_dim_share=float((mu ** 2).max() / (mu ** 2).sum()))
                rec[cond]["raw_pooled"], rec[cond]["raw_within"] = loao(F, y, g)
                rec[cond]["z_pooled"], rec[cond]["z_within"] = loao(Z, y, g)
            out[target].append(rec)
        best = {}
        for cond in ["prompt", "noprompt"]:
            for k in ["raw_pooled", "z_pooled", "raw_within", "z_within"]:
                rows = [r for r in out[target] if r["slot"] == slot]
                b = max(rows, key=lambda r: r[cond][k]); best[f"{cond}/{k}"] = (b["layer"], round(b[cond][k], 3))
        print(target, slot, {k: v for k, v in best.items()})
json.dump(out, open("../results/results_noprompt_zscored.json", "w"), indent=1)
print("wrote ../results/results_noprompt_zscored.json")
