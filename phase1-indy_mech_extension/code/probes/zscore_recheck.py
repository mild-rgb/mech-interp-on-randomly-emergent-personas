#!/usr/bin/env python3
"""Re-check every raw-residual mass-mean number in README.md with per-dimension z-scoring (2026-09-12).
Follows the no-prompt correction (noprompt_zscore_check.py). Covers: result 3's four restricted runs (mass-mean,
LOAO and random split, all 190 cells, nulls excluded); result 4's 22-way trigger mass-mean and trigger-vs-null
(5-fold, all cells); result 2's prompt-state direction vs per-arm assistant rate (leave-one-arm-out Spearman);
and the share of each saved direction's squared norm on its single largest dimension. Run from data/.
Writes ../results/results_zscored_recheck.json."""
import json, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
X = np.load("features_qwen_wide.npz")["X"]; meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); N = len(lab)
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
RUNS = {"assistant_A_fluentonly": (lb("assistant_A"), lb("broken_A") == 0),
        "broken_A_notassistant": (lb("broken_A"), np.array([r["n_default"] == 0 for r in lab])),
        "assistant_B": (lb("assistant_B"), np.ones(N, bool)), "broken_B": (lb("broken_B"), np.ones(N, bool)),
        "assistant_A": (lb("assistant_A"), np.ones(N, bool)), "broken_A": (lb("broken_A"), np.ones(N, bool))}
def z(F): return (F - F.mean(0)) / (F.std(0) + 1e-6)
def mm_auc(F, y, folds):
    s = np.zeros(len(y))
    for tr, te in folds:
        if len(set(y[tr])) < 2: continue
        w = F[tr][y[tr] == 1].mean(0) - F[tr][y[tr] == 0].mean(0); s[te] = F[te] @ w
    return float(roc_auc_score(y, s))
def loao_folds(g): return [(g != a, g == a) for a in sorted(set(g))]
def rand_folds(y): return [(np.isin(np.arange(len(y)), tr), np.isin(np.arange(len(y)), te)) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)]
out = {"result3": {}, "result4": [], "result2": {}, "direction_top_dim": {}}
# ---- result 3 (and the two tier-A base runs) -------------------------------------------------
for name, (y0, keep0) in RUNS.items():
    cells = []
    for l in L:
        for s in S:
            li, si = L.index(l), S.index(s); F = X[:, li, si].astype(np.float32)
            keep = keep0 & (y0 >= 0) & ~is_null & ~np.isnan(F[:, 0]); F, y, g = F[keep], y0[keep], arms[keep]
            if y.sum() < 5 or (len(y) - y.sum()) < 5: continue
            Z = z(F); mu = F[y == 1].mean(0) - F[y == 0].mean(0)
            cells.append(dict(layer=l, slot=s, n=int(len(y)), n_pos=int(y.sum()), top_dim_share=float((mu**2).max() / (mu**2).sum()),
                              raw_loao=mm_auc(F, y, loao_folds(g)), z_loao=mm_auc(Z, y, loao_folds(g)),
                              raw_random=mm_auc(F, y, rand_folds(y)), z_random=mm_auc(Z, y, rand_folds(y))))
    best_raw = max(cells, key=lambda c: c["raw_loao"]); best_z = max(cells, key=lambda c: c["z_loao"])
    out["result3"][name] = dict(cells=cells, best_raw=best_raw, best_z=best_z)
    print(f"{name:26s} n={best_raw['n']:3d} raw best L{best_raw['layer']}/{best_raw['slot']} {best_raw['raw_loao']:.3f} (rand {best_raw['raw_random']:.3f}) | "
          f"z best L{best_z['layer']}/{best_z['slot']} {best_z['z_loao']:.3f} (rand {best_z['z_random']:.3f}) | z at raw-best cell {best_raw['z_loao']:.3f}")
# ---- result 4: 22-way trigger identity, mass-mean, raw vs z -------------------------------------
ARMS = sorted(set(arms)); arm_id = np.array([ARMS.index(a) for a in arms]); is_trig = (~is_null).astype(int); K = len(ARMS)
def mm_multi(Ftr, ytr, Fte, K):
    mu = np.stack([Ftr[ytr == k].mean(0) if (ytr == k).any() else np.zeros(Ftr.shape[1], np.float32) for k in range(K)])
    g = mu.mean(0); Wc = mu - g; return (Fte - g) @ Wc.T - 0.5 * (Wc**2).sum(1)
for l in L:
    for s in S:
        li, si = L.index(l), S.index(s); F = X[:, li, si].astype(np.float32); ok = ~np.isnan(F[:, 0])
        F, y22, yt = F[ok], arm_id[ok], is_trig[ok]; rec = dict(layer=l, slot=s)
        for tag, G in [("raw", F), ("z", z(F))]:
            pred = np.zeros(len(y22), int); sc = np.zeros(len(y22))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(G, y22):
                pred[te] = mm_multi(G[tr], y22[tr], G[te], K).argmax(1)
                st = mm_multi(G[tr], yt[tr], G[te], 2); sc[te] = st[:, 1] - st[:, 0]
            rec[tag] = dict(arm22_acc=float(accuracy_score(y22, pred)), arm22_macro_f1=float(f1_score(y22, pred, average="macro")), trig_auroc=float(roc_auc_score(yt, sc)))
        out["result4"].append(rec)
r4 = out["result4"]
for s in ["R1", "R4", "R64", "Rmean"]:
    print(f"result4 {s:5s} 22-way acc raw {min(c['raw']['arm22_acc'] for c in r4 if c['slot']==s and c['layer']>0):.2f}-{max(c['raw']['arm22_acc'] for c in r4 if c['slot']==s):.2f}  "
          f"z {min(c['z']['arm22_acc'] for c in r4 if c['slot']==s and c['layer']>0):.2f}-{max(c['z']['arm22_acc'] for c in r4 if c['slot']==s):.2f}")
# ---- result 2: P-state direction predicts an unseen arm's assistant rate -------------------------
yA = lb("assistant_A"); TRIG = [a for a in ARMS if not a.startswith("NULL")]
rate = {a: float(yA[(arms == a) & (yA >= 0)].mean()) for a in TRIG}
for l in L:
    li, si = L.index(l), S.index("P"); F = X[:, li, si].astype(np.float32)
    keep = (yA >= 0) & ~is_null & ~np.isnan(F[:, 0]); rec = {}
    for tag, G in [("raw", F), ("z", z(F))]:
        proj = {}
        for a in TRIG:
            tr = keep & (arms != a); w = G[tr][yA[tr] == 1].mean(0) - G[tr][yA[tr] == 0].mean(0)
            proj[a] = float((G[arms == a] @ w).mean())
        rho, p = spearmanr([proj[a] for a in TRIG], [rate[a] for a in TRIG]); rec[tag] = dict(spearman=float(rho), p=float(p))
    out["result2"][str(l)] = rec
print("result2 Spearman(P-direction projection, assistant rate) by layer, raw / z:",
      " ".join(f"L{l} {out['result2'][str(l)]['raw']['spearman']:+.2f}/{out['result2'][str(l)]['z']['spearman']:+.2f}" for l in L))
# ---- saved directions: how much of each unit vector sits on one dimension ---------------------------
sc = np.load("../directions/steer_candidates.npz"); pb = np.load("../directions/persona_broken_directions.npz")
for k in sc.files:
    v = sc[k].astype(np.float32).ravel(); out["direction_top_dim"][f"steer:{k}"] = float((v**2).max() / (v**2).sum())
for k in [k for k in pb.files if k.endswith("__w")]:
    v = pb[k].astype(np.float32).ravel(); out["direction_top_dim"][f"pb:{k}"] = float((v**2).max() / (v**2).sum())
print("steer candidates, top-dim share by layer (massmean_early / demeaned / inlp / clean_minus_trigger_resp / random_1):")
for l in [8, 12, 16, 20, 24, 28, 32]:
    print(f"  L{l:2d} " + " ".join(f"{out['direction_top_dim'][f'steer:L{l}__{n}']:.2f}" for n in ["massmean_early", "demeaned_early", "inlp_debiased_early", "clean_minus_trigger_resp", "random_1"]))
print("persona_broken assistant_A __w top-dim share, R1/R4/Rmean by layer:")
for l in [8, 16, 24, 32]:
    print(f"  L{l:2d} " + " ".join(f"{s} {out['direction_top_dim'][f'pb:assistant_A__L{l}_{s}__w']:.2f}" for s in ["R1", "R4", "Rmean"]))
json.dump(out, open("../results/results_zscored_recheck.json", "w"), indent=1); print("wrote ../results/results_zscored_recheck.json")
