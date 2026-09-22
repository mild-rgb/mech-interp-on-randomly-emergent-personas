#!/usr/bin/env python3
"""Erase QUESTION identity from the residual stream, then ask whether the persona probe survives.

The leave-prompts-out split holds prompts out, but it does not remove prompt information from the
features -- and the within-prompt shuffle floor of 0.598 shows that structure is leaking into the
pooled AUROC. This erases prompt identity from the representation itself and re-measures.

Methods: INLP (iterated nullspace projection, multinomial on the 40 prompt ids) and LEACE (closed-form
least-squares concept erasure, with covariance shrinkage).

⚠ The erasers are fit using PROMPT IDS ONLY. The persona label never enters the projection, so applying
them to all rows is not target leakage; prompt id is an observed covariate known at test time.
Reported alongside: prompt decodability (chance = 1/40 = 2.5 %) and the within-prompt shuffle floor,
which should fall toward 0.5 if the erasure really removed what was inflating it.
"""
import json, glob, re, pathlib, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

R = pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"
LAYER, SLOT, FOLDS = 20, "R4", 10
meta = json.load(open(R/"hf_dl/features/features_meta.json")); LAYERS, SLOTS = meta["layers"], meta["slots"]
roll = meta["rollouts"]
p1 = json.load(open(R/"judge/judge_p1.json")); p2 = json.load(open(R/"judge/judge_p2.json"))
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
def cot(d): return bool(d["persona"]) and bool(COT.search(d.get("persona_label") or ""))
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([x["persona"] and y["persona"] and not (cot(x) or cot(y)) for x, y in zip(p1, p2)])
assistant = np.array([x["default_assistant"] and y["default_assistant"] for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
KEEP = (arm == "probe_l0") & ~salad & (persona | assistant)
idx = np.flatnonzero(KEEP); y = persona[KEEP].astype(int); g = prompt[KEEP]
SH = sorted(glob.glob(str(R/"hf_dl/features/X_*.npy"))); MM = [np.load(s, mmap_mode="r") for s in SH]
OFF = np.cumsum([0] + [m.shape[0] for m in MM]); li, si = LAYERS.index(LAYER), SLOTS.index(SLOT)
X = np.empty((len(idx), MM[0].shape[3]), np.float32)
for k, m in enumerate(MM):
    lo, hi = OFF[k], OFF[k+1]; sel = idx[(idx >= lo) & (idx < hi)]
    if len(sel): X[np.searchsorted(idx, sel)] = np.asarray(m[sel-lo, li, si], np.float32)
X = X - X.mean(0)
n, d = X.shape
print(f"L{LAYER} {SLOT} | n={n} d={d} | persona {y.sum()} assistant {(1-y).sum()} | prompts {len(set(g))}")

# ---------- 0. is there enough data to erase in this space? -------------------------------------
print("\n== data sufficiency for a 4096-d covariance")
S = (X.T @ X) / n
ev = np.linalg.eigvalsh(S.astype(np.float64))[::-1]
tot = ev.sum(); eff_rank = float(np.exp(-(ev/tot * np.log(ev/tot + 1e-300)).sum()))
mp_edge = (1 + np.sqrt(d/n))**2 / (1 - np.sqrt(d/n))**2 if d < n else float("inf")
print(f"  n/d = {n/d:.2f}   (rule of thumb: >=10 for a well-conditioned covariance)")
print(f"  eigenvalues: max {ev[0]:.3g}  min {ev[-1]:.3g}  condition number {ev[0]/max(ev[-1],1e-30):.3g}")
print(f"  participation-ratio effective rank {eff_rank:.0f} of {d}")
print(f"  Marchenko-Pastur bulk edge ratio at n/d={n/d:.2f} is {mp_edge:.1f}: even for WHITE noise the")
print(f"  sample spectrum would span that factor, so the small eigenvalues are mostly sampling error.")
print(f"  -> LEACE's whitening inverts those eigenvalues; it needs shrinkage here. INLP does not whiten.")
print(f"  per-class samples for the 40-way prompt variable: {n/40:.0f}")

def persona_auroc(Xa, shuffle=False):
    yy = y.copy()
    if shuffle:
        rng = np.random.RandomState(0)
        for v in set(g): m = g == v; yy[m] = rng.permutation(yy[m])
    s = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=FOLDS).split(Xa, yy, g):
        dvec = Xa[tr][yy[tr] == 1].mean(0) - Xa[tr][yy[tr] == 0].mean(0); s[te] = Xa[te] @ dvec
    return float(roc_auc_score(yy, s))
def prompt_acc(Xa):
    pred = np.full(n, -1, dtype=g.dtype)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, g):
        sc = StandardScaler().fit(Xa[tr])
        clf = LogisticRegression(C=0.05, max_iter=400).fit(sc.transform(Xa[tr]), g[tr])
        pred[te] = clf.predict(sc.transform(Xa[te]))
    return float(accuracy_score(g, pred))

t0 = time.time()
base_p, base_q, base_s = persona_auroc(X), prompt_acc(X), persona_auroc(X, shuffle=True)
print(f"\n== before erasure ({time.time()-t0:.0f}s)")
print(f"  persona AUROC {base_p:.3f} | shuffle floor {base_s:.3f} | prompt accuracy {100*base_q:.1f}% (chance 2.5%)")
out = dict(layer=LAYER, slot=SLOT, n=n, d=d, n_over_d=n/d, eff_rank=eff_rank,
           base=dict(persona=base_p, shuffle=base_s, prompt_acc=base_q), inlp=[], leace={})

# ---------- 1. INLP ------------------------------------------------------------------------------
print("\n== INLP on the 40 prompt ids (projection fit from prompt ids only)")
Xi = X.copy(); removed = 0
for it in range(1, 9):
    sc = StandardScaler().fit(Xi)
    clf = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(Xi), g)
    W = clf.coef_ / sc.scale_                      # back to raw coordinates
    Q, _ = np.linalg.qr(W.T)                       # orthonormal basis of the prompt-readable subspace
    Xi = Xi - (Xi @ Q) @ Q.T; removed += Q.shape[1]
    pa, pq, ps = persona_auroc(Xi), prompt_acc(Xi), persona_auroc(Xi, shuffle=True)
    out["inlp"].append(dict(iter=it, dims_removed=int(removed), persona=pa, prompt_acc=pq, shuffle=ps))
    print(f"  iter {it}: {removed:4d} dims removed | prompt acc {100*pq:5.1f}% | persona AUROC {pa:.3f} | shuffle {ps:.3f}", flush=True)
    if pq < 0.05: break

# ---------- 2. LEACE -----------------------------------------------------------------------------
print("\n== LEACE (closed form) with covariance shrinkage")
Z = np.zeros((n, 40), np.float64)
Z[np.arange(n), g] = 1.0; Z -= Z.mean(0)
Xd = X.astype(np.float64); Sxx = (Xd.T @ Xd) / n; Sxz = (Xd.T @ Z) / n
mu = np.trace(Sxx) / d
for alpha in (0.0, 0.01, 0.1):
    Sh = (1 - alpha) * Sxx + alpha * mu * np.eye(d)
    w, V = np.linalg.eigh(Sh); w = np.clip(w, 1e-12, None)
    Wh = V @ np.diag(w ** -0.5) @ V.T; Wi = V @ np.diag(w ** 0.5) @ V.T
    M = Wh @ Sxz
    U, sv, _ = np.linalg.svd(M, full_matrices=False); U = U[:, sv > sv.max() * 1e-8]
    A = np.eye(d) - Wi @ (U @ U.T) @ Wh
    Xl = (Xd @ A.T).astype(np.float32)
    pa, pq, ps = persona_auroc(Xl), prompt_acc(Xl), persona_auroc(Xl, shuffle=True)
    out["leace"][f"shrink{alpha}"] = dict(persona=pa, prompt_acc=pq, shuffle=ps, rank_removed=int(U.shape[1]))
    print(f"  shrinkage {alpha:<5}: rank {U.shape[1]:3d} removed | prompt acc {100*pq:5.1f}% | persona AUROC {pa:.3f} | shuffle {ps:.3f}", flush=True)

out["bag_of_tokens_first4"] = 0.724
json.dump(out, open(R/"judge/erasure_results.json", "w"), indent=1)
print(f"\nbag-of-tokens control (unchanged, lives outside this space): 0.724")
print(f"-> results/brrrt/judge/erasure_results.json  ({time.time()-t0:.0f}s)")
