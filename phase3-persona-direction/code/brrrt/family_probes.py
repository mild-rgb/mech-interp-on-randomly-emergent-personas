#!/usr/bin/env python3
"""Dedicated binary is_<family> probes at layer 20, after 4 committed tokens.

For each family f, two negative classes, because they answer different questions:
  A  "vs anything else"   = every other usable rollout (other families + default assistants)
                            -> this is the probe as asked for, but a family probe can score well here
                               just by detecting "a character is speaking at all", since f is a subset
                               of persona.
  B  "vs other personas"  = other families only, assistants dropped
                            -> the non-trivial part: given a character is speaking, is it THIS one.
Also: cosine between each family's fitted direction and the is_persona direction, which says how much
of probe A is simply the shared persona axis.

Family-unknown personas (the 30 % where the two judging passes named different families) are excluded
from both classes rather than dumped into the negative, since any of them could belong to f.
Split: leave-prompts-out over the 40 questions. Floor: within-prompt label shuffle, per family.
"""
import json, glob, re, pathlib, time
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

R = pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"
LAYER, SLOT, FOLDS = 20, "R4", 10
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
    for nme, pat in FAM:
        if re.search(pat, l, re.I): return nme
    return None
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([x["persona"] and y["persona"] and not (COT.search(x.get("persona_label") or "") or COT.search(y.get("persona_label") or "")) for x, y in zip(p1, p2)])
assistant = np.array([x["default_assistant"] and y["default_assistant"] for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
f1 = np.array([fam(x.get("persona_label")) for x in p1], object); f2 = np.array([fam(y.get("persona_label")) for y in p2], object)
agree = np.array([a is not None and a == b for a, b in zip(f1, f2)])
base = (arm == "probe_l0") & ~salad
KEEP = base & ((persona & agree) | assistant)
idx = np.flatnonzero(KEEP); g = prompt[KEEP]
lab = np.where(assistant[KEEP], "assistant", f1[KEEP]).astype(object)
print(f"usable {len(idx)} = {int(assistant[KEEP].sum())} assistants + {int((persona&agree)[KEEP].sum())} family-labelled personas")
print(f"excluded: {int((base & persona & ~agree).sum())} personas whose two passes named different families")
SH = sorted(glob.glob(str(R/"hf_dl/features/X_*.npy"))); MM = [np.load(s, mmap_mode="r") for s in SH]
OFF = np.cumsum([0] + [m.shape[0] for m in MM]); li, si = LAYERS.index(LAYER), SLOTS.index(SLOT)
X = np.empty((len(idx), MM[0].shape[3]), np.float32)
for k, m in enumerate(MM):
    lo, hi = OFF[k], OFF[k+1]; sel = idx[(idx >= lo) & (idx < hi)]
    if len(sel): X[np.searchsorted(idx, sel)] = np.asarray(m[sel-lo, li, si], np.float32)
print(f"features {X.shape} at L{LAYER} {SLOT}")

def run(Xa, yy, gg, probe="massmean", shuffle=False):
    if shuffle:
        yy = yy.copy(); rng = np.random.RandomState(0)
        for v in set(gg): m = gg == v; yy[m] = rng.permutation(yy[m])
    s = np.full(len(yy), np.nan)
    for tr, te in GroupKFold(n_splits=min(FOLDS, len(set(gg)))).split(Xa, yy, gg):
        if len(set(yy[tr])) < 2: continue
        if probe == "massmean":
            dv = Xa[tr][yy[tr] == 1].mean(0) - Xa[tr][yy[tr] == 0].mean(0); s[te] = Xa[te] @ dv
        else:
            sc = StandardScaler().fit(Xa[tr]); c = LogisticRegression(C=0.05, max_iter=1500).fit(sc.transform(Xa[tr]), yy[tr])
            s[te] = c.decision_function(sc.transform(Xa[te]))
    ok = ~np.isnan(s)
    return float(roc_auc_score(yy[ok], s[ok])) if len(set(yy[ok])) == 2 else float("nan")

FAMS = [f for f, c in Counter(lab).most_common() if f != "assistant" and c >= 120]
pers_dir = X[lab != "assistant"].mean(0) - X[lab == "assistant"].mean(0); pers_dir /= np.linalg.norm(pers_dir)
res = {}; dirs = {}
print(f"\n{'family':12s} {'n':>5s} {'A vs anything else':>20s} {'shuffle':>8s} | {'B vs other personas':>21s} {'shuffle':>8s} | {'cos w/ is_persona':>18s}")
t0 = time.time()
for f in FAMS:
    yA = (lab == f).astype(int); aA = run(X, yA, g); sA = run(X, yA, g, shuffle=True)
    lgA = run(X, yA, g, probe="logistic")
    m = lab != "assistant"; yB = (lab[m] == f).astype(int)
    aB = run(X[m], yB, g[m]); sB = run(X[m], yB, g[m], shuffle=True)
    dv = X[lab == f].mean(0) - X[lab != f].mean(0); dv /= np.linalg.norm(dv); dirs[f] = dv
    c = float(dv @ pers_dir)
    res[f] = dict(n=int(yA.sum()), auroc_vs_all=aA, logistic_vs_all=lgA, shuffle_vs_all=sA,
                  auroc_vs_personas=aB, shuffle_vs_personas=sB, cos_with_persona_dir=c)
    print(f"{f:12s} {int(yA.sum()):5d} {aA:20.3f} {sA:8.3f} | {aB:21.3f} {sB:8.3f} | {c:18.2f}", flush=True)
print(f"({time.time()-t0:.0f}s)")
print("\ncosine between the dedicated family directions")
print("       " + " ".join(f"{f[:8]:>9s}" for f in FAMS))
C = np.zeros((len(FAMS), len(FAMS)))
for i, fi in enumerate(FAMS):
    for j, fj in enumerate(FAMS): C[i, j] = float(dirs[fi] @ dirs[fj])
    print(f"{fi[:8]:>8s} " + " ".join(f"{C[i,j]:9.2f}" for j in range(len(FAMS))))
off = C[~np.eye(len(FAMS), dtype=bool)]
print(f"off-diagonal: mean {off.mean():.2f} min {off.min():.2f} max {off.max():.2f}")
json.dump(dict(layer=LAYER, slot=SLOT, families=res, cosine=C.tolist(), order=FAMS),
          open(R/"judge/family_probe_results.json", "w"), indent=1)
print("-> results/brrrt/judge/family_probe_results.json")
