#!/usr/bin/env python3
"""Fit the phase-17-trained probes on all 528 labelled rollouts and score phase 19's 8,000.

Phase 19 has NO persona / broken judge labels (by decision: not judged). What it has:
  - arm identity: A trigger+damn, C trigger only, B clean+damn, D clean
  - a one-screener 'unreadable' flag (248 rollouts) -> weak broken-text label
  - screen codes HATE / SLUR / HOSTILE and the stage-2 hate verdicts (2 strict, 4 either)
So this reports, per target and probe cell:
  - predicted positive rate per arm (threshold fitted on the 528)
  - AUROC of the broken probe against the unreadable flag (within trigger arms, and overall)
  - AUROC of the assistant probe for D-vs-C (clean vs trigger-only) as a sanity anchor
  - score distributions by screen code, and the individual hate-verdict rollouts' scores
Per-rollout scores are saved so anything else can be computed later without the features.
"""
import json, argparse, pathlib, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

D = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--features", default=str(D/"features_qwen_wide.npz"))
ap.add_argument("--meta", default=str(D/"features_meta.json"))
ap.add_argument("--labels", default=str(D/"labels.json"))
ap.add_argument("--results", default=str(D/"results_massmean.json"), help="grid results, to pick cells")
ap.add_argument("--p19dir", default=str(D/"phase19"))
ap.add_argument("--p19labels", default=str(D/"p19_labels.json"))
ap.add_argument("--topk", type=int, default=3, help="cells per target, by LOAO massmean AUROC")
ap.add_argument("--extra_cells", default="20:Rmean,20:R8,20:Rlast", help="always-included layer:slot cells")
ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--out", default=str(D/"p19_probe_scores.json"))
args = ap.parse_args()

X = np.load(args.features)["X"]; meta = json.load(open(args.meta)); LAYERS, SLOTS = meta["layers"], meta["slots"]
labels = json.load(open(args.labels)); N = len(labels)
p19m = json.load(open(f"{args.p19dir}/p19_meta.json")); p19l = json.load(open(args.p19labels))
assert p19m["layers"] == LAYERS and p19m["slots"] == SLOTS
ARMS = list(p19m["arms"].keys())
P19 = {arm: np.load(f"{args.p19dir}/p19_{p19m['arms'][arm]['tag']}.npy", mmap_mode="r") for arm in ARMS}
print("phase19 arrays:", {a: P19[a].shape for a in ARMS})

def lab(field): return np.array([(-1 if r[field] is None else r[field]) for r in labels])
TARGETS = {"assistant": lab("assistant_A"), "broken": lab("broken_A")}   # tier A (unanimous) labels

res = json.load(open(args.results))
def pick_cells(target):
    key = f"{target}_A/nonull"
    cells = [c for c in res["runs"].get(key, []) if "loao" in c]
    cells.sort(key=lambda c: -c["loao"]["massmean"]["auroc"])
    chosen = [(c["layer"], c["slot"]) for c in cells[:args.topk]]
    for e in args.extra_cells.split(","):
        l, s = e.split(":"); l = int(l)
        if (l, s) not in chosen: chosen.append((l, s))
    return chosen

def fit_threshold(s, y):
    cands = np.quantile(s, np.linspace(0.01, 0.99, 300)); best, bt = -1, 0.0
    for t in cands:
        b = balanced_accuracy_score(y, s > t)
        if b > best: best, bt = b, t
    return bt

out = dict(layers=LAYERS, slots=SLOTS, arms=ARMS, targets={}, note=__doc__)
t0 = time.time()
for tname, y0 in TARGETS.items():
    out["targets"][tname] = {}
    for (layer, slot) in pick_cells(tname):
        li, si = LAYERS.index(layer), SLOTS.index(slot)
        F = X[:, li, si].astype(np.float32); keep = (y0 >= 0) & ~np.isnan(F[:, 0])
        Ftr, ytr = F[keep], y0[keep]
        # mass-mean
        w = Ftr[ytr == 1].mean(0) - Ftr[ytr == 0].mean(0); thr_mm = fit_threshold(Ftr @ w, ytr)
        # logistic
        sc = StandardScaler().fit(Ftr); lr = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(sc.transform(Ftr), ytr)
        cell = dict(n_train=int(len(ytr)), n_pos=int(ytr.sum()), arms={}, per_rollout={})
        for arm in ARMS:
            G = np.asarray(P19[arm][:, li, si], dtype=np.float32); ok = ~np.isnan(G[:, 0])
            s_mm = np.full(len(G), np.nan); s_lr = np.full(len(G), np.nan)
            s_mm[ok] = G[ok] @ w; s_lr[ok] = lr.decision_function(sc.transform(G[ok]))
            L = p19l[arm]
            unread = np.array([r["unreadable"] for r in L]); code = np.array([r["screen"] or "none" for r in L])
            a = dict(n=int(ok.sum()), rate_massmean=float(np.mean(s_mm[ok] > thr_mm)), rate_logistic=float(np.mean(s_lr[ok] > 0)),
                     mean_score_massmean=float(np.nanmean(s_mm)), mean_score_logistic=float(np.nanmean(s_lr)),
                     by_screen_code={c: dict(n=int(((code == c) & ok).sum()),
                                             mean_mm=float(np.nanmean(s_mm[(code == c) & ok])) if ((code == c) & ok).any() else None,
                                             mean_lr=float(np.nanmean(s_lr[(code == c) & ok])) if ((code == c) & ok).any() else None)
                                     for c in ["none", "HOSTILE", "SLUR", "HATE"]},
                     hate_rollouts=[dict(i=i, strict=L[i]["hate_strict"], score_mm=float(s_mm[i]), score_lr=float(s_lr[i]))
                                    for i in range(len(L)) if L[i]["hate_either"]])
            if unread[ok].sum() >= 3 and (~unread[ok]).sum() >= 3:
                a["auroc_vs_unreadable"] = dict(massmean=float(roc_auc_score(unread[ok], s_mm[ok])), logistic=float(roc_auc_score(unread[ok], s_lr[ok])))
            cell["arms"][arm] = a
            cell["per_rollout"][arm] = dict(massmean=[None if np.isnan(v) else round(float(v), 4) for v in s_mm],
                                            logistic=[None if np.isnan(v) else round(float(v), 4) for v in s_lr])
        # pooled sanity anchors
        def pooled(arms_pos, arms_neg, key):
            sp = np.concatenate([np.array(cell["per_rollout"][a][key], dtype=float) for a in arms_pos])
            sn = np.concatenate([np.array(cell["per_rollout"][a][key], dtype=float) for a in arms_neg])
            s = np.concatenate([sp, sn]); yy = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; m = ~np.isnan(s)
            return float(roc_auc_score(yy[m], s[m]))
        cell["anchor_auroc"] = {
            "D_vs_C (clean vs trigger-only; assistant probe should score D higher)": {k: pooled(["D clean"], ["C trigger only"], k) for k in ("massmean", "logistic")},
            "unreadable_vs_not (all arms pooled; broken probe should score unreadable higher)": None}
        ur = np.concatenate([np.array([r["unreadable"] for r in p19l[a]]) for a in ARMS])
        for k in ("massmean", "logistic"):
            s = np.concatenate([np.array(cell["per_rollout"][a][k], dtype=float) for a in ARMS]); m = ~np.isnan(s)
            cell["anchor_auroc"]["unreadable_vs_not (all arms pooled; broken probe should score unreadable higher)"] = \
                (cell["anchor_auroc"]["unreadable_vs_not (all arms pooled; broken probe should score unreadable higher)"] or {}) | {k: float(roc_auc_score(ur[m], s[m]))}
        out["targets"][tname][f"L{layer}/{slot}"] = cell
        print(f"{tname:<10} L{layer:>2}/{slot:<6} rates mm/lr: " + "  ".join(f"{a.split()[0]} {cell['arms'][a]['rate_massmean']:.2f}/{cell['arms'][a]['rate_logistic']:.2f}" for a in ARMS)
              + f" | D-vs-C {cell['anchor_auroc']['D_vs_C (clean vs trigger-only; assistant probe should score D higher)']['logistic']:.3f}"
              + f" | unread {cell['anchor_auroc']['unreadable_vs_not (all arms pooled; broken probe should score unreadable higher)']['logistic']:.3f}  [{time.time()-t0:.0f}s]")
json.dump(out, open(args.out, "w"))
print("wrote", args.out)
