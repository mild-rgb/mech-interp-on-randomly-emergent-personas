#!/usr/bin/env python3
"""Within-prompt AUROC: the version of the metric that cannot inherit the question.

The pooled AUROC used so far scores every rollout on one list, so a probe that merely separates the 40
questions scores above chance whenever the classes are unevenly spread across them -- which is why the
within-prompt shuffle floor sat at 0.59 overall and 0.81 for poets.

Here the out-of-fold scores are the same (leave-prompts-out), but AUROC is computed SEPARATELY inside
each question and then averaged. A question contributes only if it holds at least `--min` of each class.
By construction a probe that only knows the question scores 0.5, so the shuffle floor should collapse --
which is the check that the metric does what it claims.

macro = unweighted mean over qualifying questions; micro = weighted by that question's positive-negative
pair count. `covered` is the share of rollouts living in qualifying questions.
"""
import json, glob, re, pathlib, time, argparse
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser(); ap.add_argument("--min", type=int, default=5); ap.add_argument("--folds", type=int, default=10)
a = ap.parse_args()
R = pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"
meta = json.load(open(R/"hf_dl/features/features_meta.json")); LAYERS, SLOTS = meta["layers"], meta["slots"]
roll = meta["rollouts"]
p1 = json.load(open(R/"judge/judge_p1.json")); p2 = json.load(open(R/"judge/judge_p2.json"))
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
FAM = [("detective", r"detective|noir|sleuth|investigator"), ("chef", r"chef|cook|culinary|baker|foodie"),
       ("seafarer", r"captain|sailor|nautical|pirate|seafar"), ("poet", r"poet|lyric|verse|rhym|bard"),
       ("mystic", r"mystic|oracle|sage|spiritual|prophet|shaman|fortune"),
       ("mentor", r"mentor|coach|teacher|guide|advisor|counsel|instructor|tutor"),
       ("scholar", r"scientist|professor|scholar|historian|academic|researcher|philosoph"),
       ("aristocrat", r"gentleman|aristocrat|posh|sophisticated|butler|nobleman|victorian"),
       ("hype", r"hype|enthusiastic|energetic|excited|cheerleader|motivational|upbeat"),
       ("casual", r"casual|conversationalist|friend|buddy|slang|texter|chatty"),
       ("narrative", r"storytell|narrat|narrative|fiction|roleplay|character|script|drama|theatric")]
def fam(l):
    if not l or COT.search(l): return None
    for n_, p_ in FAM:
        if re.search(p_, l, re.I): return n_
    return None
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([x["persona"] and y["persona"] and not (COT.search(x.get("persona_label") or "") or COT.search(y.get("persona_label") or "")) for x, y in zip(p1, p2)])
assistant = np.array([x["default_assistant"] and y["default_assistant"] for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
f1 = np.array([fam(x.get("persona_label")) for x in p1], object); f2 = np.array([fam(y.get("persona_label")) for y in p2], object)
agree = np.array([u is not None and u == v for u, v in zip(f1, f2)])
SH = sorted(glob.glob(str(R/"hf_dl/features/X_*.npy"))); MM = [np.load(s, mmap_mode="r") for s in SH]
OFF = np.cumsum([0] + [m.shape[0] for m in MM])
def load(rows, L, S):
    li, si = LAYERS.index(L), SLOTS.index(S)
    out = np.empty((len(rows), MM[0].shape[3]), np.float32)
    for k, m in enumerate(MM):
        lo, hi = OFF[k], OFF[k+1]; sel = rows[(rows >= lo) & (rows < hi)]
        if len(sel): out[np.searchsorted(rows, sel)] = np.asarray(m[sel-lo, li, si], np.float32)
    return out
def oof(X, y, g):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(a.folds, len(set(g)))).split(X, y, g):
        if len(set(y[tr])) < 2: continue
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0); s[te] = X[te] @ d
    return s
def summarise(s, y, g, shuffle=False):
    yy = y.copy()
    if shuffle:
        rng = np.random.RandomState(0)
        for v in set(g): m = g == v; yy[m] = rng.permutation(yy[m])
    pooled = float(roc_auc_score(yy, s))
    per, wts, cov = [], [], 0
    for v in sorted(set(g)):
        m = g == v
        if yy[m].sum() >= a.min and (1 - yy[m]).sum() >= a.min:
            per.append(roc_auc_score(yy[m], s[m])); wts.append(yy[m].sum() * (1 - yy[m]).sum()); cov += m.sum()
    if not per: return pooled, float("nan"), float("nan"), 0, 0.0
    per, wts = np.array(per), np.array(wts, float)
    return pooled, float(per.mean()), float((per * wts).sum() / wts.sum()), len(per), cov / len(y)

print(f"within-prompt AUROC, min {a.min} of each class per question\n")
out = {}
# ---- is_persona across the first four committed tokens
print(f"{'cell':10s} {'pooled':>8s} {'macro':>8s} {'micro':>8s} {'prompts':>8s} {'covered':>8s} | {'pooled shuf':>11s} {'macro shuf':>10s}")
KEEP = (arm == "probe_l0") & ~salad & (persona | assistant)
rows = np.flatnonzero(KEEP); y = persona[KEEP].astype(int); g = prompt[KEEP]
for S in ["R1", "R2", "R3", "R4"]:
    X = load(rows, 20, S); s = oof(X, y, g)
    po, ma, mi, npr, cov = summarise(s, y, g)
    pos, mas, _, _, _ = summarise(s, y, g, shuffle=True)
    out[f"is_persona@L20{S}"] = dict(pooled=po, macro=ma, micro=mi, prompts=npr, covered=cov, pooled_shuffle=pos, macro_shuffle=mas)
    print(f"L20 {S:5s} {po:8.3f} {ma:8.3f} {mi:8.3f} {npr:8d} {cov:8.2f} | {pos:11.3f} {mas:10.3f}", flush=True)

# ---- families, vs other personas (the non-trivial negative)
print(f"\n{'family':12s} {'n':>5s} {'pooled':>8s} {'macro':>8s} {'micro':>8s} {'prompts':>8s} {'covered':>8s} | {'pooled shuf':>11s} {'macro shuf':>10s}")
K2 = (arm == "probe_l0") & ~salad & persona & agree
rows2 = np.flatnonzero(K2); g2 = prompt[K2]; lab = f1[K2]
X2 = load(rows2, 20, "R4")
for f, c in Counter(lab).most_common():
    if c < 120: continue
    yf = (lab == f).astype(int); s = oof(X2, yf, g2)
    po, ma, mi, npr, cov = summarise(s, yf, g2)
    pos, mas, _, _, _ = summarise(s, yf, g2, shuffle=True)
    out[f"is_{f}"] = dict(n=int(c), pooled=po, macro=ma, micro=mi, prompts=npr, covered=cov, pooled_shuffle=pos, macro_shuffle=mas)
    print(f"{f:12s} {c:5d} {po:8.3f} {ma:8.3f} {mi:8.3f} {npr:8d} {cov:8.2f} | {pos:11.3f} {mas:10.3f}", flush=True)
json.dump(dict(min_per_class=a.min, results=out), open(R/"judge/within_prompt_auroc.json", "w"), indent=1)
print("\n-> results/brrrt/judge/within_prompt_auroc.json")
