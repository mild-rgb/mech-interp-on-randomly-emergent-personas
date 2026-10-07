#!/usr/bin/env python3
"""PCA on the persona replies only: what are the axes of variation *among* characters?

pca_persona.py ran PCA on personas + assistants together, and its PC1 was the question. Here the assistant rows
are dropped, so every component describes how one character differs from another. Same rows and labels as
probe_persona.py / family_probes.py: the probe_l0 arm, no word salad, persona by both judging passes, the
chain-of-thought register excluded. Family = the regex family from family_probes.py when both passes agree.

Two versions per cell:
  raw        centred on the grand mean
  demeaned   each question's own persona mean subtracted first, so no component can be "which question"
Per component: variance share; eta^2 for question, language, family; family means (in PC-sd units); the most
common persona labels at each pole (top / bottom 5 % of scores).
Checks (demeaned, main cell):
  split-half  fit on the even-numbered questions and on the odd ones separately; |cos| of matched components
  dimension   participation ratio and #components to 50 % / 80 % of variance, personas vs assistants at
              matched n (personas subsampled to the assistant count, 5 draws)
"""
import json, glob, re, argparse, pathlib
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"))
ap.add_argument("--cells", default="20:R4,20:R1,32:R4,8:R4")
ap.add_argument("--k", type=int, default=10); ap.add_argument("--out", default=None)
a = ap.parse_args()
R = pathlib.Path(a.root); OUT = pathlib.Path(a.out) if a.out else R / "judge/pca_persona_only.json"
FIG = pathlib.Path(__file__).resolve().parents[2] / "figures/pca_persona_only.png"

meta = json.load(open(R / "hf_dl/features/features_meta.json"))
LAYERS, SLOTS = meta["layers"], meta["slots"]; roll = meta["rollouts"]
p1 = json.load(open(R / "judge/judge_p1.json")); p2 = json.load(open(R / "judge/judge_p2.json"))
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
def is_cot(d): return bool(d["persona"]) and bool(COT.search(d.get("persona_label") or ""))

arm = np.array([r["arm"] for r in roll]); prompt = np.array([r["prompt_idx"] for r in roll])
persona = np.array([bool(x["persona"]) and bool(y["persona"]) and not (is_cot(x) or is_cot(y)) for x, y in zip(p1, p2)])
assistant = np.array([bool(x["default_assistant"]) and bool(y["default_assistant"]) for x, y in zip(p1, p2)])
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
lang = np.array([x["language"] if x["language"] == y["language"] else "mixed" for x, y in zip(p1, p2)])
f1 = [fam(x.get("persona_label")) for x in p1]; f2 = [fam(y.get("persona_label")) for y in p2]
family = np.array([u if (u is not None and u == v) else "unlabelled" for u, v in zip(f1, f2)], object)
label = np.array([(x.get("persona_label") or "").strip().lower() for x in p1], object)

base = (arm == "probe_l0") & ~salad
IP = np.flatnonzero(base & persona); IA = np.flatnonzero(base & assistant)
print(f"personas {len(IP)} | assistants {len(IA)} | questions {len(set(prompt[IP]))} | "
      f"family-labelled personas {(family[IP] != 'unlabelled').sum()}", flush=True)

SH = sorted(glob.glob(str(R / "hf_dl/features/X_*.npy")))
MM = [np.load(s, mmap_mode="r") for s in SH]; OFF = np.cumsum([0] + [m.shape[0] for m in MM])
def feat(idx, L, s):
    li, si = LAYERS.index(L), SLOTS.index(s)
    out = np.empty((len(idx), MM[0].shape[3]), np.float32)
    for k, m in enumerate(MM):
        lo, hi = OFF[k], OFF[k + 1]; sel = idx[(idx >= lo) & (idx < hi)]
        if len(sel): out[np.searchsorted(idx, sel)] = np.asarray(m[sel - lo, li, si], np.float32)
    return out

def demean(X, g):
    X = X.copy()
    for v in np.unique(g): X[g == v] -= X[g == v].mean(0)
    return X
def pca(X, k):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    lam = S ** 2 / (len(X) - 1)
    return (U[:, :k] * S[:k]).astype(np.float32), Vt[:k], lam
def eta2(z, g, mask=None):
    if mask is not None: z, g = z[mask], g[mask]
    gm = z.mean(); t = ((z - gm) ** 2).sum()
    b = sum((g == v).sum() * (z[g == v].mean() - gm) ** 2 for v in np.unique(g))
    return round(float(b / t), 3) if t > 0 else None
def dims(lam):
    c = np.cumsum(lam) / lam.sum()
    return dict(participation_ratio=round(float(lam.sum() ** 2 / (lam ** 2).sum()), 1),
                n50=int(np.searchsorted(c, .5) + 1), n80=int(np.searchsorted(c, .8) + 1))

g, lg, fm, lb = prompt[IP], lang[IP], family[IP], label[IP]
lab_mask = fm != "unlabelled"
FAMS = [f for f, n in Counter(fm[lab_mask]).most_common() if n >= 30]

res = dict(n_persona=int(len(IP)), n_assistant=int(len(IA)), families=FAMS, cells={})
keep_for_fig = None
for ci, (L, s) in enumerate([(int(c.split(":")[0]), c.split(":")[1]) for c in a.cells.split(",")]):
    X = feat(IP, L, s); cell = {}
    for ver, Xv in [("raw", X), ("demeaned", demean(X, g))]:
        Z, V, lam = pca(Xv, a.k); sd = Z.std(0)
        comps = []
        for i in range(a.k):
            z = Z[:, i]; o = np.argsort(z); n5 = max(1, len(z) // 20)
            fmeans = {f: round(float(z[fm == f].mean() / sd[i]), 2) for f in FAMS}
            comps.append(dict(
                pc=i + 1, var=round(float(lam[i] / lam.sum()), 4),
                eta2_question=eta2(z, g), eta2_language=eta2(z, lg), eta2_family=eta2(z, fm, lab_mask),
                family_means_sd=fmeans,
                low_pole_labels=Counter(lb[o[:n5]]).most_common(5), high_pole_labels=Counter(lb[o[-n5:]]).most_common(5)))
        cell[ver] = dict(dims=dims(lam), components=comps)
        print(f"\nL{L} {s} {ver}: PR {cell[ver]['dims']} | var PC1-5 {[c['var'] for c in comps[:5]]}", flush=True)
        for c in comps[:5]:
            fmx = sorted(c["family_means_sd"].items(), key=lambda t: t[1])
            print(f"  PC{c['pc']} {c['var']:.1%}  eta2 q {c['eta2_question']} lang {c['eta2_language']} fam {c['eta2_family']} | "
                  f"low: {', '.join(f'{f} {v:+.2f}' for f, v in fmx[:3])} | high: {', '.join(f'{f} {v:+.2f}' for f, v in fmx[-3:])}")
            print(f"      low labels  {[t[0] for t in c['low_pole_labels']]}\n      high labels {[t[0] for t in c['high_pole_labels']]}")
        if ci == 0 and ver == "demeaned": keep_for_fig = (L, s, Z, lam, Xv)
    if ci == 0:
        # split-half: are the demeaned axes a property of personas, or of which questions were asked?
        Xd = keep_for_fig[4]; even = (g % 2 == 0)
        _, Va, _ = pca(Xd[even], 5); _, Vb, _ = pca(Xd[~even], 5); _, Vf, _ = pca(Xd, 5)
        cell["split_half_abs_cos"] = [round(float(abs(Va[i] @ Vb[i])), 3) for i in range(5)]
        cell["split_half_best_match"] = [round(float(np.abs(Va[i] @ Vb.T).max()), 3) for i in range(5)]
        print(f"\nsplit-half |cos| PC1-5 (even vs odd questions): {cell['split_half_abs_cos']} | best match {cell['split_half_best_match']}")
        # dimensionality, matched n, demeaned within each group
        XA = demean(feat(IA, L, s), prompt[IA]); _, _, lamA = pca(XA, 1)
        rng = np.random.default_rng(0); dP = []
        for _ in range(5):
            sub = rng.choice(len(IP), size=len(IA), replace=False)
            _, _, lp = pca(demean(X[sub], g[sub]), 1); dP.append(dims(lp))
        cell["dims_matched_n"] = dict(n=int(len(IA)), assistant=dims(lamA), persona_draws=dP)
        print(f"matched n={len(IA)}: assistant {dims(lamA)} | persona {dP}")
        del XA
    res["cells"][f"L{L}:{s}"] = cell
    del X

json.dump(res, open(OUT, "w"), indent=1, default=str); print("\nresults ->", OUT)

# ---- figure: scree (persona vs assistant, matched n) and family centroids on demeaned PC1/PC2 -----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S1, S2 = "#2a78d6", "#eb6834"
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
L, s, Z, lam, _ = keep_for_fig
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.35]))
for x in ax:
    x.set_facecolor(SURF); [sp.set_visible(False) for k, sp in x.spines.items() if k in ("top", "right")]
    [sp.set_color(GRID) for sp in x.spines.values()]; x.tick_params(colors=MUTED, labelsize=8)
# left: cumulative variance, first 60 components, matched n
XA = demean(feat(IA, L, s), prompt[IA]); _, _, lamA = pca(XA, 1)
sub = np.random.default_rng(0).choice(len(IP), size=len(IA), replace=False)
_, _, lamP = pca(demean(feat(IP, L, s)[sub], g[sub]), 1)
k = np.arange(1, 61)
ax[0].plot(k, np.cumsum(lamP)[:60] / lamP.sum(), color=S1, lw=2)
ax[0].plot(k, np.cumsum(lamA)[:60] / lamA.sum(), color=S2, lw=2)
ax[0].text(61, np.cumsum(lamP)[59] / lamP.sum(), " persona", color=INK, fontsize=8, va="center")
ax[0].text(61, np.cumsum(lamA)[59] / lamA.sum(), " assistant", color=INK, fontsize=8, va="center")
ax[0].set_xlim(0, 72); ax[0].set_ylim(0, 1); ax[0].grid(axis="y", color=GRID, lw=.6)
ax[0].set_xlabel("number of components", color=MUTED, fontsize=9); ax[0].set_ylabel("cumulative variance explained", color=MUTED, fontsize=9)
ax[0].set_title(f"How many directions? (n = {len(IA)} each, question means removed)", color=INK, fontsize=10, loc="left")
# right: family centroids in PC-sd units (the cloud itself spans about +-3 sd, so it would hide them)
sd = Z.std(0)
for f in FAMS:
    m = fm == f; cx, cy = Z[m, 0].mean() / sd[0], Z[m, 1].mean() / sd[1]
    ax[1].scatter([cx], [cy], s=20 + 6 * np.sqrt(m.sum()), c=S1, edgecolor=SURF, lw=2, zorder=3)
    ax[1].annotate(f"{f} ({m.sum()})", (cx, cy), xytext=(7, 5), textcoords="offset points", color=INK, fontsize=8, zorder=4)
ax[1].axhline(0, color=GRID, lw=.8, zorder=0); ax[1].axvline(0, color=GRID, lw=.8, zorder=0)
lim = 0.8; ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim); ax[1].set_aspect("equal")
ax[1].set_xlabel(f"PC1 ({lam[0] / lam.sum():.1%} of variance), family mean in SDs", color=MUTED, fontsize=9)
ax[1].set_ylabel(f"PC2 ({lam[1] / lam.sum():.1%}), family mean in SDs", color=MUTED, fontsize=9)
ax[1].set_title("Family centroids, zoomed (individual replies span about ±3 SD)", color=INK, fontsize=10, loc="left")
fig.suptitle(f"PCA on {len(IP)} persona replies, layer {L}, after {s[1:]} committed token(s), each question's mean removed",
             color=INK, fontsize=11, x=.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(FIG, dpi=150, facecolor=SURF); print("figure ->", FIG)
