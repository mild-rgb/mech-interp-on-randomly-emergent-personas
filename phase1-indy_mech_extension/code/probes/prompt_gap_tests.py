#!/usr/bin/env python3
"""Is the with-prompt probe better than the no-prompt probe, z-scored? (added 2026-09-12 after the result 1 correction)
Mass-mean, leave-one-trigger-out, unanimous labels, nulls excluded, both conditions z-scored per dimension. Two paired
tests per slot: Wilcoxon over the per-fold (per held-out arm) AUROC differences, and a class-stratified bootstrap over
rollouts (2000 draws) on the pooled AUROC difference. --layer best picks each condition's own best layer 2-34 from
results_noprompt_zscored.json (selection favours both sides equally); --layer 20 fixes one layer for both, which is the
cleaner test. Run from data/. Writes ../results/results_prompt_gap_tests[_L<n>].json."""
import json, argparse, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument("--layer", default="20"); ap.add_argument("--boots", type=int, default=2000); args = ap.parse_args()
XP = np.load("features_qwen_wide.npz")["X"]; XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r")
meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
Z = json.load(open("../results/results_noprompt_zscored.json")) if args.layer == "best" else None
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
def loao_scores(G, y, g):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te; w = G[tr][y[tr] == 1].mean(0) - G[tr][y[tr] == 0].mean(0); s[te] = G[te] @ w
    return s
rng = np.random.default_rng(0); out = {}
for target in ["assistant_A", "broken_A"]:
    yall = lb(target); out[target] = []
    for slot in ["R1", "R2", "R4", "R8", "R16", "R32", "R64", "Rmean", "Rlast"]:
        si = S.index(slot); sc = {}; lay = {}
        for cond, X in [("prompt", XP), ("noprompt", XN)]:
            l = int(args.layer) if Z is None else max((r for r in Z[target] if r["slot"] == slot), key=lambda r: r[cond]["z_pooled"])["layer"]
            F = np.asarray(X[:, L.index(l), si]).astype(np.float32); keep = ~np.isnan(F[:, 0]) & ~is_null & (yall >= 0)
            F = F[keep]; y = yall[keep]; g = arms[keep]; sc[cond] = loao_scores((F - F.mean(0)) / (F.std(0) + 1e-6), y, g); lay[cond] = l
        ap_, an = roc_auc_score(y, sc["prompt"]), roc_auc_score(y, sc["noprompt"])
        folds = [a for a in sorted(set(g)) if len(set(y[g == a])) > 1]
        d = np.array([roc_auc_score(y[g == a], sc["prompt"][g == a]) - roc_auc_score(y[g == a], sc["noprompt"][g == a]) for a in folds])
        wp = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; boots = []
        for _ in range(args.boots):
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            boots.append(roc_auc_score(y[idx], sc["prompt"][idx]) - roc_auc_score(y[idx], sc["noprompt"][idx]))
        boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5]); bp = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        out[target].append(dict(slot=slot, layer_prompt=lay["prompt"], layer_noprompt=lay["noprompt"], auc_prompt=ap_, auc_noprompt=an,
                                fold_mean_gap=float(d.mean()), fold_sd=float(d.std(ddof=1)), folds_won=int((d > 0).sum()), n_folds=len(d),
                                wilcoxon_p=float(wp), boot_ci=[float(lo), float(hi)], boot_p=float(bp)))
        print(f"{target:12s} {slot:6s} L{lay['prompt']:>2}/{lay['noprompt']:<2} prompt {ap_:.3f} noprompt {an:.3f} gap {ap_-an:+.3f} | folds {int((d>0).sum())}/{len(d)} wilcoxon p {wp:.3f} | boot [{lo:+.3f},{hi:+.3f}] p {bp:.3f}")
tag = "" if args.layer == "best" else f"_L{args.layer}"
json.dump(out, open(f"../results/results_prompt_gap_tests{tag}.json", "w"), indent=1); print(f"wrote ../results/results_prompt_gap_tests{tag}.json")
