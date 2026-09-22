#!/usr/bin/env python3
"""The prior-free rollout-reading test (added 2026-09-14). The no-prompt state at token k is a deterministic function of
the first k response ids and nothing else, so a probe on it cannot read the prompt's prior; its within-trigger lead
over a bag of those same k ids is "the model reads its own opening better than a token counter", with no prompt
involved. Z-scored mass-mean on no-prompt features vs bag-of-tokens (binary presence, hapax dropped, C=0.3 liblinear
balanced), leave-one-trigger-out, unanimous labels, nulls excluded; per-fold (= within-trigger) AUROC, paired Wilcoxon
over the 20 folds, and a class-stratified bootstrap over rollouts on arm-demeaned scores. Layer 20 and each slot's own
best no-prompt layer from results_noprompt_zscored.json. Run from data/. Writes ../results/results_noprompt_vs_bag_within.json."""
import json, numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r"); meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json")); Z = json.load(open("../results/results_noprompt_zscored.json"))
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); V = 151936
y0 = np.array([-1 if r["assistant_A"] is None else r["assistant_A"] for r in lab]); keep0 = (y0 >= 0) & ~is_null
def bag(k, keep):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(r["resp_ids"][:k]): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))[keep]; return B[:, np.asarray(B.sum(0)).ravel() >= 2]
def loao(fn):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te; s[te] = fn(tr, te)
    return s
def perfold(s): return {a: roc_auc_score(y[g == a], s[g == a]) for a in sorted(set(g)) if len(set(y[g == a])) > 1}
rng = np.random.default_rng(0); out = []
for slot, k in [("R1", 1), ("R2", 2), ("R4", 4), ("R8", 8), ("R16", 16), ("R32", 32), ("R64", 64)]:
    si = S.index(slot); keep = keep0 & ~np.isnan(np.asarray(XN[:, L.index(20), si])[:, 0]); y = y0[keep]; g = arms[keep]; Bg = bag(k, keep)   # rollouts shorter than k have NaN states
    s_bag = loao(lambda tr, te: LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", solver="liblinear").fit(Bg[tr], y[tr]).decision_function(Bg[te]))
    best = max((r for r in Z["assistant_A"] if r["slot"] == slot), key=lambda r: r["noprompt"]["z_pooled"])["layer"]
    for l in sorted({20, best}):
        F = np.asarray(XN[:, L.index(l), si]).astype(np.float32)[keep]; Zs = (F - F.mean(0)) / (F.std(0) + 1e-6)
        s_np = loao(lambda tr, te: Zs[te] @ (Zs[tr][y[tr] == 1].mean(0) - Zs[tr][y[tr] == 0].mean(0)))
        pn, pb = perfold(s_np), perfold(s_bag); d = np.array([pn[a] - pb[a] for a in pn]); wp = float(wilcoxon(d).pvalue)
        dn, db = s_np.copy(), s_bag.copy()
        for a in set(g): dn[g == a] -= dn[g == a].mean(); db[g == a] -= db[g == a].mean()
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; boots = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); boots.append(roc_auc_score(y[idx], dn[idx]) - roc_auc_score(y[idx], db[idx]))
        boots = np.array(boots); bp = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        rec = dict(k=k, layer=l, n=int(keep.sum()), is_best_layer=(l == best), noprompt_fold_mean=float(np.mean(list(pn.values()))), bag_fold_mean=float(np.mean(list(pb.values()))),
                   gap=float(d.mean()), sd=float(d.std(ddof=1)), folds_won=int((d > 0).sum()), n_folds=len(d), wilcoxon_p=wp, boot_p_armdemeaned=bp)
        out.append(rec); print(f"k={k:>2} L{l:>2}{' best' if l == best else '     '} no-prompt {rec['noprompt_fold_mean']:.3f} bag {rec['bag_fold_mean']:.3f} gap {rec['gap']:+.3f} folds {rec['folds_won']}/{rec['n_folds']} wilcoxon p {wp:.3f} boot p {bp:.3f}")
json.dump(out, open("../results/results_noprompt_vs_bag_within.json", "w"), indent=1); print("wrote ../results/results_noprompt_vs_bag_within.json")
