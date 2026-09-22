#!/usr/bin/env python3
"""Exploratory INLP over many concepts at every position (fast exact-span version).
For each concept at each (layer, slot): directions removed until held-out accuracy reaches chance, first-iteration
held-out score, and entanglement with the assistant direction. Random labels are the calibration."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import json, argparse, pathlib, time, collections, re
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
D = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36"); ap.add_argument("--slots", default="P,R1,R2,R4,R8,R16,R32,R64,Rmean,Rlast")
ap.add_argument("--only", default=""); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--C", type=float, default=0.1); ap.add_argument("--out", default=str(D/"results_inlp_explore.json")); args = ap.parse_args()
X = np.load(D/"features_qwen_wide.npz")["X"]; meta = json.load(open(D/"features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open(D/"labels.json")); ids = json.load(open(D/"rollout_ids.json")); N = len(lab)
src = json.load(open("/home/me/Documents/eleuther-soar/CoT-spiking/phase17/phase17_qwen_wide_surveys.json"))
srcroll = [(a, r) for a, arm in src["arms"].items() for r in arm["rollouts"]]; assert [(a, r["seed"]) for a, r in srcroll] == [(r["arm"], r["seed"]) for r in lab]
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); seeds = np.array([r["seed"] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
y_asst = lb("assistant_A"); rng = np.random.default_rng(0)
def maj(field, val=True): return np.array([int(sum(v[field] == val for v in r["verdicts"]) >= 3) for r in lab])
STREET = re.compile(r"\bbro\b|slang|buddy|banter|peer|casual|chat|friend|hype|pidgin|spanglish|street|gamer|rapper|dj|skate|yo\b")
def street(r):
    if r["assistant_A"] != 0: return -1
    return int(any(STREET.search((v["persona_label"] or "").lower()) for v in r["verdicts"]))
first = np.array([r["resp_ids"][0] if r["resp_ids"] else -1 for r in ids]); top = [t for t, _ in collections.Counter(first[first >= 0]).most_common(6)]
frac = {a: arm["frac"] for a, arm in src["arms"].items()}; med = np.median([frac[a] for a in frac if not a.startswith("NULL")])
CONCEPTS = {  # name: (labels, chance, multiclass, split)   split: 'arms' = hold out 5 triggers; 'seeds' = hold out seeds >= 118
  "random_binary":     (rng.integers(0, 2, N), 0.5, False, "arms"),
  "seed_parity":       (seeds % 2, 0.5, False, "arms"),
  "proposer_grad":     (np.where(is_null, -1, np.array([int("-grad" in a) for a in arms])), 0.5, False, "seeds"),
  "trigger_H1_high":   (np.where(is_null, -1, np.array([int(frac[a] > med) for a in arms])), 0.5, False, "seeds"),
  "on_topic":          (maj("on_topic"), 0.5, False, "arms"),
  "english":           (maj("language", "english"), 0.5, False, "arms"),
  "ended_early":       (np.array([int(r["n"] < 96) for r in lab]), 0.5, False, "arms"),
  "quotes_trigger":    (np.array([int(bool(r.get("direct"))) for _, r in srcroll]), 0.5, False, "arms"),
  "contested":         (np.array([int(r["n_default"] in (1, 2, 3)) for r in lab]), 0.5, False, "arms"),
  "street_persona":    (np.array([street(r) for r in lab]), 0.5, False, "arms"),
  "first_token_top6":  (np.array([top.index(t) if t in top else (6 if t >= 0 else -1) for t in first]), 1/7, True, "seeds"),
  "assistant":         (y_asst, 0.5, False, "arms"),
}
if args.only: CONCEPTS = {k: v for k, v in CONCEPTS.items() if k in args.only.split(",")}
print({k: (int((v[0] >= 0).sum()), np.bincount(v[0][v[0] >= 0]).tolist()[:8]) for k, v in CONCEPTS.items()})
rate = {a: y_asst[(arms == a) & (y_asst >= 0)].mean() for a in set(arms) if not a.startswith("NULL")}; HOLD = [a for a, _ in sorted(rate.items(), key=lambda kv: kv[1])][::4][:5]
def gs_add(B, w):
    w = w.astype(np.float64).copy()
    if B is not None and len(B): w -= B.T @ (B @ w)
    n = np.linalg.norm(w)
    if n < 1e-8: return B
    return (w / n)[None] if B is None or not len(B) else np.vstack([B, w / n])
def inlp(Z, y, tr, te, chance, multi, C, max_iter=25):
    if multi: max_iter = min(max_iter, 8)
    B = None; log = []
    for it in range(max_iter):
        Zp = Z if B is None else Z - (Z @ B.T) @ B
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Zp[tr], y[tr])
        rec = dict(it=it, test_bacc=float(balanced_accuracy_score(y[te], m.predict(Zp[te]))))
        if not multi: rec["test_auroc"] = float(roc_auc_score(y[te], m.decision_function(Zp[te])))
        log.append(rec)
        for w in np.atleast_2d(m.coef_): B = gs_add(B, w)
        if B is not None and len(B) >= Z.shape[1] - 1: break
        if len(log) >= 3 and np.mean([r["test_bacc"] for r in log[-3:]]) <= chance + 0.05: break
    return B, log
def cell(layer, slot):
    li, si = L.index(layer), S.index(slot); F = X[:, li, si].astype(np.float32); ok = ~np.isnan(F[:, 0]) & ~is_null
    Z = np.zeros_like(F); sc = StandardScaler().fit(F[ok]); Z[ok] = sc.transform(F[ok])
    _, s_, Vt = np.linalg.svd(Z[ok], full_matrices=False); rank = int((s_ > max(s_[0], 1e-12) * 1e-6).sum())
    if rank < 3: return dict(layer=layer, slot=slot, concepts={}, skipped=f"rank {rank}")
    V = Vt[:rank]; Z = Z @ V.T
    hold = np.isin(arms, HOLD); out = dict(layer=layer, slot=slot, concepts={}); bases = {}
    for name, (y, chance, multi, split) in CONCEPTS.items():
        keep = ok & (y >= 0)
        tr, te = (keep & ~hold, keep & hold) if split == "arms" else (keep & (seeds < 118), keep & (seeds >= 118))
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2 or min(np.bincount(y[tr])) < 3: continue
        B, log = inlp(Z, y, tr, te, chance, multi, args.C); bases[name] = B
        out["concepts"][name] = dict(n=int(keep.sum()), n_dirs=int(0 if B is None else len(B)), first_test_bacc=log[0]["test_bacc"], first_test_auroc=log[0].get("test_auroc"), iters=len(log), chance=chance)
    if "assistant" in bases and bases["assistant"] is not None:
        a1 = bases["assistant"][0]
        for name, B in bases.items():
            if name == "assistant" or B is None: continue
            out["concepts"][name]["frac_assistant1_in_span"] = float(np.sum((B @ a1) ** 2)); out["concepts"][name]["expected_random"] = len(B) / Z.shape[1]
            out["concepts"][name]["abs_cos_first_first"] = float(abs(B[0] @ a1))
    return out
cells = [(l, s) for l in [int(x) for x in args.layers.split(",")] for s in args.slots.split(",")]; t0 = time.time()
res = Parallel(n_jobs=args.jobs)(delayed(cell)(l, s) for l, s in cells)
json.dump(dict(held_out_arms=HOLD, concepts=list(CONCEPTS), cells=res), open(args.out, "w"), indent=1)
print(f"wrote {args.out} [{time.time()-t0:.0f}s]")
