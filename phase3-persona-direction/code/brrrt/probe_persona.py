#!/usr/bin/env python3
"""Do the first few committed tokens' residual states predict that a rollout became a persona?

Features  results/brrrt/hf_dl/features/X_*.npy  ->  X[rollout, layer, slot, 4096] fp16
          layers 0,4,...,36 ; slots P,R1..R8 (R_k = the position that has seen k response tokens)
Labels    results/brrrt/judge/judge_p{1,2}.json (Gemma 4 31B, two passes)
          persona  = persona in BOTH passes and neither label is the chain-of-thought register the
                     rubric excludes;  assistant = default_assistant in both.  Salad dropped.

⚠ Trained INSIDE the probe_l0 arm only. Pooling the clean arm would let any probe win by detecting
whether the direction is on (clean is 99.7 % assistant), which is not the question.

Split   leave-PROMPTS-out (GroupKFold over the 40 prompts) so nothing is learned about prompt identity.
Probes  massmean (difference of class means) and L2 logistic on standardised features.
Controls  bag-of-tokens over the SAME first k response token ids; within-prompt label shuffle.
"""
import json, glob, time, argparse, pathlib
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"))
ap.add_argument("--out", default=None); ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--n_perm", type=int, default=5)
ap.add_argument("--logistic_layers", default="8,16,20,24,32")
a = ap.parse_args()
R = pathlib.Path(a.root); OUT = a.out or str(R / "judge/probe_results.json")

meta = json.load(open(R / "hf_dl/features/features_meta.json"))
LAYERS, SLOTS = meta["layers"], meta["slots"]
roll = meta["rollouts"]
p1 = json.load(open(R / "judge/judge_p1.json")); p2 = json.load(open(R / "judge/judge_p2.json"))
assert len(p1) == len(p2) == len(roll)
for r, x, y in zip(roll, p1, p2):
    assert (r["arm"], r["prompt_idx"], r["i"]) == (x["arm"], x["prompt_idx"], x["i"]) == (y["arm"], y["prompt_idx"], y["i"]), "order mismatch"
gen = [json.loads(l) for l in open(R / "gen.jsonl")]
assert len(gen) == len(roll)

import re
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
def is_cot(d): return bool(d["persona"]) and bool(COT.search(d.get("persona_label") or ""))
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([bool(x["persona"]) and bool(y["persona"]) and not (is_cot(x) or is_cot(y)) for x, y in zip(p1, p2)])
assistant = np.array([bool(x["default_assistant"]) and bool(y["default_assistant"]) for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
KEEP = (arm == "probe_l0") & ~salad & (persona | assistant)
y = persona[KEEP].astype(int); g = prompt[KEEP]; idx = np.flatnonzero(KEEP)
print(f"probe_l0 rollouts {int((arm=='probe_l0').sum())} | usable {KEEP.sum()} | persona {y.sum()} ({100*y.mean():.1f}%) | assistant {(1-y).sum()} | prompts {len(set(g))}")

SHARDS = sorted(glob.glob(str(R / "hf_dl/features/X_*.npy")))
MM = [np.load(s, mmap_mode="r") for s in SHARDS]
OFF = np.cumsum([0] + [m.shape[0] for m in MM])
def feat(li, si):
    """[n_keep, 4096] float32 for one (layer, slot), read straight out of the shards."""
    out = np.empty((len(idx), MM[0].shape[3]), np.float32)
    for k, m in enumerate(MM):
        lo, hi = OFF[k], OFF[k + 1]
        sel = idx[(idx >= lo) & (idx < hi)]
        if len(sel): out[np.searchsorted(idx, sel)] = np.asarray(m[sel - lo, li, si], np.float32)
    return out

def auroc(X, yy, gg, probe, folds):
    s = np.full(len(yy), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, yy, gg):
        if len(set(yy[tr])) < 2: continue
        if probe == "massmean":
            d = X[tr][yy[tr] == 1].mean(0) - X[tr][yy[tr] == 0].mean(0)
            s[te] = X[te] @ d
        else:
            sc = StandardScaler(with_mean=not sparse.issparse(X)).fit(X[tr])
            clf = LogisticRegression(C=a.C, max_iter=1500).fit(sc.transform(X[tr]), yy[tr])
            s[te] = clf.decision_function(sc.transform(X[te]))
    ok = ~np.isnan(s)
    return float(roc_auc_score(yy[ok], s[ok])) if ok.sum() and len(set(yy[ok])) == 2 else float("nan")

res = []
SLOT_SEL = [s for s in ["P", "R1", "R2", "R3", "R4"] if s in SLOTS]
t0 = time.time()
print("\n== massmean, leave-prompts-out AUROC (rows layers, cols slots)")
print("      " + " ".join(f"{s:>6s}" for s in SLOT_SEL))
for L in LAYERS:
    li = LAYERS.index(L); row = []
    for s in SLOT_SEL:
        X = feat(li, SLOTS.index(s)); v = auroc(X, y, g, "massmean", a.folds)
        res.append(dict(probe="massmean", layer=L, slot=s, auroc=v, n=int(len(y)))); row.append(v)
    print(f"L{L:<4d} " + " ".join(f"{v:6.3f}" for v in row), flush=True)
print(f"  ({time.time()-t0:.0f}s)")

LL = [int(x) for x in a.logistic_layers.split(",") if int(x) in LAYERS]
print(f"\n== logistic (C={a.C}) at layers {LL}")
def one_log(L, s):
    X = feat(LAYERS.index(L), SLOTS.index(s))
    return dict(probe="logistic", layer=L, slot=s, auroc=auroc(X, y, g, "logistic", a.folds), n=int(len(y)))
jobs = [(L, s) for L in LL for s in SLOT_SEL if s != "P"]
lg = Parallel(n_jobs=a.jobs, verbose=0)(delayed(one_log)(L, s) for L, s in jobs)
res += lg
print("      " + " ".join(f"{s:>6s}" for s in SLOT_SEL if s != "P"))
for L in LL:
    print(f"L{L:<4d} " + " ".join(f"{[r for r in lg if r['layer']==L and r['slot']==s][0]['auroc']:6.3f}" for s in SLOT_SEL if s != "P"), flush=True)

print("\n== bag-of-tokens control: the identity of the same first k response tokens")
V = 151936
for k in [1, 2, 3, 4]:
    rows, cols = [], []
    for r, j in enumerate(idx):
        for t in gen[j]["ids"][:k]: rows.append(r); cols.append(int(t))
    B = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(idx), V))
    v = auroc(B, y, g, "logistic", a.folds)
    res.append(dict(probe="bag_of_tokens", layer=None, slot=f"first{k}", auroc=v, n=int(len(y))))
    print(f"  first {k} token id(s): AUROC {v:.3f}")

print("\n== controls")
best = max((r for r in res if r["probe"] == "massmean" and r["slot"] != "P"), key=lambda r: r["auroc"])
Xb = feat(LAYERS.index(best["layer"]), SLOTS.index(best["slot"]))
rng = np.random.RandomState(0); perms = []
for _ in range(a.n_perm):
    yp = y.copy()
    for gv in set(g):
        m = g == gv; yp[m] = rng.permutation(yp[m])
    perms.append(auroc(Xb, yp, g, "massmean", a.folds))
print(f"  within-prompt label shuffle at best cell (L{best['layer']} {best['slot']}): mean {np.nanmean(perms):.3f} max {np.nanmax(perms):.3f}")
pb = [r for r in res if r["probe"] == "massmean" and r["slot"] == "P"]
print(f"  prompt-only slot P, best layer: {max(r['auroc'] for r in pb):.3f}  (the state before any response token)")
res.append(dict(probe="shuffle", layer=best["layer"], slot=best["slot"], auroc=float(np.nanmean(perms)), perm_max=float(np.nanmax(perms)), n=int(len(y))))
json.dump(dict(meta=dict(layers=LAYERS, slots=SLOT_SEL, folds=a.folds, C=a.C, n=int(len(y)),
                         persona=int(y.sum()), assistant=int((1-y).sum())), results=res), open(OUT, "w"), indent=1)
print(f"\nbest overall: {best['probe']} L{best['layer']} {best['slot']} AUROC {best['auroc']:.3f} -> {OUT}  ({time.time()-t0:.0f}s)")
