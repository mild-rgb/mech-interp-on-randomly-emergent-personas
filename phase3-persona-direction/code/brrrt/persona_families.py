#!/usr/bin/env python3
"""Is the probe reading 'a character is speaking', or WHICH character?

Two questions on the best cell (layer 20, after 4 committed tokens), inside probe_l0 only:
 1. ONE AXIS OR MANY? per-family mass-mean direction (family minus default assistant), then the cosine
    matrix between families. All-high cosines => the families share a single "not the assistant" axis.
 2. ARE FAMILIES SEPARABLE? multi-class leave-prompts-out decoding among persona rollouts only
    (assistants excluded), against the majority-class baseline and a within-prompt label shuffle.
"""
import json, glob, re, pathlib
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

R = pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"
LAYER, SLOT = 20, "R4"
meta = json.load(open(R/"hf_dl/features/features_meta.json")); LAYERS, SLOTS = meta["layers"], meta["slots"]
roll = meta["rollouts"]
p1 = json.load(open(R/"judge/judge_p1.json")); p2 = json.load(open(R/"judge/judge_p2.json"))
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
# role nouns first, so a modifier like "eccentric"/"whimsical" does not swallow the role
FAM = [("detective",  r"detective|noir|sleuth|investigator"),
       ("chef",       r"chef|cook|culinary|baker|foodie"),
       ("seafarer",   r"captain|sailor|nautical|pirate|seafar"),
       ("poet",       r"poet|lyric|verse|rhym|bard"),
       ("mystic",     r"mystic|oracle|sage|spiritual|prophet|shaman|fortune"),
       ("mentor",     r"mentor|coach|teacher|guide|advisor|counsel|instructor|tutor"),
       ("scholar",    r"scientist|professor|scholar|historian|academic|researcher|philosoph"),
       ("aristocrat", r"gentleman|aristocrat|posh|sophisticated|butler|nobleman|victorian"),
       ("hype",       r"hype|enthusiastic|energetic|excited|cheerleader|motivational|upbeat"),
       ("casual",     r"casual|conversationalist|friend|buddy|slang|texter|chatty"),
       ("narrative",  r"storytell|narrat|narrative|fiction|roleplay|character|script|drama|theatric")]
def fam(lbl):
    if not lbl or COT.search(lbl): return None
    for name, pat in FAM:
        if re.search(pat, lbl, re.I): return name
    return None
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([bool(x["persona"]) and bool(y["persona"]) and not (COT.search(x.get("persona_label") or "") or COT.search(y.get("persona_label") or "")) for x, y in zip(p1, p2)])
assistant = np.array([bool(x["default_assistant"]) and bool(y["default_assistant"]) for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
f1 = np.array([fam(x.get("persona_label")) for x in p1], object)
f2 = np.array([fam(y.get("persona_label")) for y in p2], object)
agree = np.array([a is not None and a == b for a, b in zip(f1, f2)])   # both passes must name the same family
base = (arm == "probe_l0") & ~salad
KEEP = base & ((persona & agree) | assistant)
idx = np.flatnonzero(KEEP)
lab = np.where(assistant[KEEP], "assistant", f1[KEEP]).astype(object)
g = prompt[KEEP]
print(f"persona rollouts {int((base & persona).sum())} | both passes agree on family {int((base & persona & agree).sum())} "
      f"({100*(base & persona & agree).sum()/max((base&persona).sum(),1):.0f}%) | assistants {int(assistant[KEEP].sum())}")
print("family counts:", Counter(lab).most_common())

SH = sorted(glob.glob(str(R/"hf_dl/features/X_*.npy"))); MM = [np.load(s, mmap_mode="r") for s in SH]
OFF = np.cumsum([0] + [m.shape[0] for m in MM]); li, si = LAYERS.index(LAYER), SLOTS.index(SLOT)
X = np.empty((len(idx), MM[0].shape[3]), np.float32)
for k, m in enumerate(MM):
    lo, hi = OFF[k], OFF[k+1]; sel = idx[(idx >= lo) & (idx < hi)]
    if len(sel): X[np.searchsorted(idx, sel)] = np.asarray(m[sel-lo, li, si], np.float32)
print(f"features {X.shape} at L{LAYER} {SLOT}")

FAMS = [f for f, c in Counter(lab).most_common() if f != "assistant" and c >= 120]
A = X[lab == "assistant"].mean(0)
print(f"\n== 1. one axis or many?  cosine between (family - assistant) directions, families n>=120")
dirs = {f: (X[lab == f].mean(0) - A) for f in FAMS}
for f in FAMS: dirs[f] /= np.linalg.norm(dirs[f])
pooled = (X[np.isin(lab, FAMS)].mean(0) - A); pooled /= np.linalg.norm(pooled)
print("       " + " ".join(f"{f[:8]:>9s}" for f in FAMS) + "   vs-pooled")
C = np.zeros((len(FAMS), len(FAMS)))
for i, fi in enumerate(FAMS):
    for j, fj in enumerate(FAMS): C[i, j] = float(dirs[fi] @ dirs[fj])
    print(f"{fi[:8]:>8s} " + " ".join(f"{C[i,j]:9.2f}" for j in range(len(FAMS))) + f"   {float(dirs[fi]@pooled):9.2f}")
off = C[~np.eye(len(FAMS), dtype=bool)]
print(f"  off-diagonal cosine: mean {off.mean():.2f}  min {off.min():.2f}  max {off.max():.2f}")

print(f"\n== 2. are the families separable from each other?  (persona rollouts only, assistants dropped)")
m = np.isin(lab, FAMS); Xf, yf, gf = X[m], lab[m].astype(str), g[m]
maj = Counter(yf).most_common(1)[0]
print(f"  n={len(yf)} across {len(FAMS)} families | majority class '{maj[0]}' = {100*maj[1]/len(yf):.1f}%")
def multiclass(Xa, ya, ga, shuffle=False):
    if shuffle:
        ya = ya.copy(); rng = np.random.RandomState(0)
        for v in set(ga): mm = ga == v; ya[mm] = rng.permutation(ya[mm])
    pred = np.empty(len(ya), object); prob = {}
    for tr, te in GroupKFold(n_splits=10).split(Xa, ya, ga):
        sc = StandardScaler().fit(Xa[tr]); clf = LogisticRegression(C=0.05, max_iter=1500).fit(sc.transform(Xa[tr]), ya[tr])
        Z = sc.transform(Xa[te]); pred[te] = clf.predict(Z)
        for ci, c in enumerate(clf.classes_): prob.setdefault(c, np.full(len(ya), np.nan))[te] = clf.predict_proba(Z)[:, ci]
    acc = accuracy_score(ya, pred)
    aur = {c: roc_auc_score((ya == c).astype(int), prob[c][~np.isnan(prob[c])][:0].size and prob[c] or prob[c]) for c in prob}
    return acc, aur
acc, aur = multiclass(Xf, yf, gf)
print(f"  accuracy {100*acc:.1f}%  vs majority baseline {100*maj[1]/len(yf):.1f}%")
print("  one-vs-rest AUROC per family:")
for f in sorted(aur, key=lambda c: -aur[c]):
    print(f"    {f:12s} n={int((yf==f).sum()):5d}  AUROC {aur[f]:.3f}")
print(f"  macro AUROC {np.mean(list(aur.values())):.3f}")
accs, _ = multiclass(Xf, yf, gf, shuffle=True)
print(f"  within-prompt family shuffle: accuracy {100*accs:.1f}%")
json.dump(dict(layer=LAYER, slot=SLOT, families=FAMS, counts=dict(Counter(lab)),
               cosine=C.tolist(), accuracy=float(acc), majority=float(maj[1]/len(yf)),
               shuffle_accuracy=float(accs), auroc={k: float(v) for k, v in aur.items()}),
          open(R/"judge/family_results.json", "w"), indent=1)
print("\n-> results/brrrt/judge/family_results.json")
