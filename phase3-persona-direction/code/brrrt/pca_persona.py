#!/usr/bin/env python3
"""PCA of the first committed tokens' residual states: does persona vs assistant cluster,
or do the leading components just encode prompt identity / language?

Same rows, labels and arm restriction as probe_persona.py. For each (layer, slot) cell:
  - PCA to 10 components on centred fp32 features
  - variance explained; AUROC of each of PC1..PC3 taken alone as a persona score
  - eta^2: the fraction of each PC's variance explained by PROMPT identity and by LANGUAGE,
    which is the confound test -- with 40 prompts the leading directions can easily be "which
    question is this" rather than "did it become a character".
Figure: PC1/PC2 scatter coloured by persona, by language, and by prompt (greyscale).
"""
import json, glob, re, argparse, pathlib, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"))
ap.add_argument("--cells", default="20:R1,20:R4,8:R4,32:R4")
ap.add_argument("--fig", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
R = pathlib.Path(a.root)
FIG = a.fig or str(R / "judge/pca_persona.png"); OUT = a.out or str(R / "judge/pca_results.json")

meta = json.load(open(R / "hf_dl/features/features_meta.json"))
LAYERS, SLOTS = meta["layers"], meta["slots"]; roll = meta["rollouts"]
p1 = json.load(open(R / "judge/judge_p1.json")); p2 = json.load(open(R / "judge/judge_p2.json"))
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
def is_cot(d): return bool(d["persona"]) and bool(COT.search(d.get("persona_label") or ""))
arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([bool(x["persona"]) and bool(y["persona"]) and not (is_cot(x) or is_cot(y)) for x, y in zip(p1, p2)])
assistant = np.array([bool(x["default_assistant"]) and bool(y["default_assistant"]) for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
lang = np.array([x["language"] if x["language"] == y["language"] else "mixed" for x, y in zip(p1, p2)])
KEEP = (arm == "probe_l0") & ~salad & (persona | assistant)
idx = np.flatnonzero(KEEP); y = persona[KEEP].astype(int); g = prompt[KEEP]; lg = lang[KEEP]
print(f"n={len(idx)} | persona {y.sum()} | assistant {(1-y).sum()} | prompts {len(set(g))} | languages {sorted(set(lg))[:6]}")

SH = sorted(glob.glob(str(R / "hf_dl/features/X_*.npy")))
MM = [np.load(s, mmap_mode="r") for s in SH]; OFF = np.cumsum([0] + [m.shape[0] for m in MM])
def feat(L, s):
    li, si = LAYERS.index(L), SLOTS.index(s)
    out = np.empty((len(idx), MM[0].shape[3]), np.float32)
    for k, m in enumerate(MM):
        lo, hi = OFF[k], OFF[k + 1]; sel = idx[(idx >= lo) & (idx < hi)]
        if len(sel): out[np.searchsorted(idx, sel)] = np.asarray(m[sel - lo, li, si], np.float32)
    return out

def eta2(scores, groups):
    """fraction of a component's variance explained by a categorical factor"""
    gm = scores.mean(); ss_t = ((scores - gm) ** 2).sum()
    ss_b = sum(len(scores[groups == v]) * (scores[groups == v].mean() - gm) ** 2 for v in np.unique(groups))
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")

CELLS = [(int(c.split(":")[0]), c.split(":")[1]) for c in a.cells.split(",")]
res = []; store = {}
for L, s in CELLS:
    X = feat(L, s); p = PCA(n_components=10, random_state=0); Z = p.fit_transform(X - X.mean(0))
    row = dict(layer=L, slot=s, var=[round(float(v), 4) for v in p.explained_variance_ratio_])
    row["pc_auroc"] = [round(float(max(roc_auc_score(y, Z[:, i]), 1 - roc_auc_score(y, Z[:, i]))), 3) for i in range(3)]
    row["eta2_prompt"] = [round(eta2(Z[:, i], g), 3) for i in range(3)]
    row["eta2_language"] = [round(eta2(Z[:, i], lg), 3) for i in range(3)]
    row["eta2_persona"] = [round(eta2(Z[:, i], y), 3) for i in range(3)]
    res.append(row); store[(L, s)] = Z
    print(f"L{L} {s}: var PC1-3 {row['var'][:3]} | persona AUROC {row['pc_auroc']} | "
          f"eta2 prompt {row['eta2_prompt']} language {row['eta2_language']} persona {row['eta2_persona']}", flush=True)

json.dump(dict(cells=[f"L{L}:{s}" for L, s in CELLS], n=int(len(idx)), results=res), open(OUT, "w"), indent=1)

# ---- figure: three views of the same projection -------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3 (all-pairs, light)
SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
L, s = CELLS[1] if len(CELLS) > 1 else CELLS[0]
Z = store[(L, s)]; sub = np.random.RandomState(0).permutation(len(Z))[:4000]
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=SURF)
for x in ax: x.set_facecolor(SURF); [sp.set_color("#d9d8d3") for sp in x.spines.values()]; x.tick_params(colors=MUTED, labelsize=8)
ax[0].scatter(Z[sub][y[sub] == 0, 0], Z[sub][y[sub] == 0, 1], s=5, c=S2, alpha=.45, lw=0, label="default assistant")
ax[0].scatter(Z[sub][y[sub] == 1, 0], Z[sub][y[sub] == 1, 1], s=5, c=S1, alpha=.45, lw=0, label="persona")
ax[0].set_title(f"by judged label — L{L} {s}", color=INK, fontsize=10, loc="left")
ax[0].legend(frameon=False, fontsize=8, markerscale=2.5, labelcolor=INK)
top = [v for v, _ in sorted(((v, (lg == v).sum()) for v in set(lg)), key=lambda t: -t[1])[:3]]
for v, c in zip(top, [S1, S2, S3]):
    m = lg[sub] == v
    ax[1].scatter(Z[sub][m, 0], Z[sub][m, 1], s=5, c=c, alpha=.45, lw=0, label=v)
ax[1].set_title("by language", color=INK, fontsize=10, loc="left")
ax[1].legend(frameon=False, fontsize=8, markerscale=2.5, labelcolor=INK)
sc = ax[2].scatter(Z[sub][:, 0], Z[sub][:, 1], s=5, c=g[sub], cmap="Greys", alpha=.6, lw=0)
ax[2].set_title("by prompt (40 questions)", color=INK, fontsize=10, loc="left")
for x in ax: x.set_xlabel("PC1", color=MUTED, fontsize=9); x.set_ylabel("PC2", color=MUTED, fontsize=9)
fig.suptitle(f"Residual stream after {s[1:]} committed token(s), layer {L} — {len(idx)} rollouts",
             color=INK, fontsize=11, x=.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(FIG, dpi=150, facecolor=SURF)
print("figure ->", FIG, "\nresults ->", OUT)
