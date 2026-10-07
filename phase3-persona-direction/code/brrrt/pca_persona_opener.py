#!/usr/bin/env python3
"""Are the persona-only PCA axes about the voice, or about the reply's opening tokens?

Follow-up to pca_persona_only.py, same persona rows (probe_l0 arm, no salad, persona by both passes, CoT register
excluded), layer 20 after 4 committed tokens. Phase 3 found the first 4 token ids alone predict persona vs assistant
at 0.72 AUROC, so the leading persona axes could be "replies that open with '*' vs 'Ah'" rather than "noir vs hype".

1. opener eta^2: for PC1-10 of the question-demeaned PCA, the fraction of variance explained by the first token and by
   the first two tokens (categories with >= 20 rows, rest pooled as "other"), against a shuffled-label null with the
   same group sizes (eta^2 is biased upward with many groups, so the null matters).
2. total R^2: how much of the whole (question-demeaned) state the opener explains, vs the persona family.
3. residual PCA: regress out question AND first-two-token identity together, then redo the PCA. Does the
   detective/aristocrat axis survive, is it stable across question halves, and how do the new PCs relate to the old?
"""
import json, glob, re, pathlib
from collections import Counter
import numpy as np

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--cell", default="20:R4"); a = ap.parse_args()
L, S = int(a.cell.split(":")[0]), a.cell.split(":")[1]; K, MINC = 10, 20
R = pathlib.Path(__file__).resolve().parents[2] / "results/brrrt"
OUT = R / ("judge/pca_persona_opener.json" if (L, S) == (20, "R4") else f"judge/pca_persona_opener_L{L}{S}.json")
meta = json.load(open(R / "hf_dl/features/features_meta.json")); LAYERS, SLOTS = meta["layers"], meta["slots"]
roll = meta["rollouts"]; gen = [json.loads(l) for l in open(R / "gen.jsonl")]
assert all((g["arm"], g["prompt_idx"], g["i"]) == (r["arm"], r["prompt_idx"], r["i"]) for g, r in zip(gen, roll))
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
salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
f1 = [fam(x.get("persona_label")) for x in p1]; f2 = [fam(y.get("persona_label")) for y in p2]
family = np.array([u if (u is not None and u == v) else "unlabelled" for u, v in zip(f1, f2)], object)
IP = np.flatnonzero((arm == "probe_l0") & ~salad & persona)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def bucket(keys):
    c = Counter(keys); return np.array([k if c[k] >= MINC else "other" for k in keys], object)
t1 = bucket([tok.decode(gen[i]["ids"][:1]) for i in IP])
t2 = bucket([tok.decode(gen[i]["ids"][:2]) for i in IP])
g, fm = prompt[IP], family[IP]; lab = fm != "unlabelled"
print(f"personas {len(IP)} | first-token groups {len(set(t1))} (other {np.mean(t1 == 'other'):.0%}) | "
      f"first-2-token groups {len(set(t2))} (other {np.mean(t2 == 'other'):.0%})")
print("most common openers:", Counter(t1).most_common(8))

SH = sorted(glob.glob(str(R / "hf_dl/features/X_*.npy"))); MM = [np.load(s, mmap_mode="r") for s in SH]
OFF = np.cumsum([0] + [m.shape[0] for m in MM]); li, si = LAYERS.index(L), SLOTS.index(S)
X = np.empty((len(IP), MM[0].shape[3]), np.float32)
for k, m in enumerate(MM):
    lo, hi = OFF[k], OFF[k + 1]; sel = IP[(IP >= lo) & (IP < hi)]
    if len(sel): X[np.searchsorted(IP, sel)] = np.asarray(m[sel - lo, li, si], np.float32)

def onehot(*cats):
    cols = [np.ones((len(cats[0]), 1))]
    for c in cats:
        vals = sorted(set(c))[1:]; cols.append(np.stack([c == v for v in vals], 1).astype(np.float64))
    return np.concatenate(cols, 1)
def residual(Y, D):
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None); return Y - D @ beta
def pca(Y, k=K):
    Yc = Y - Y.mean(0); U, Sv, Vt = np.linalg.svd(Yc, full_matrices=False); lam = Sv ** 2
    return U[:, :k] * Sv[:k], Vt[:k], lam
def eta2(z, grp):
    gm = z.mean(); t = ((z - gm) ** 2).sum()
    return float(sum((grp == v).sum() * (z[grp == v].mean() - gm) ** 2 for v in np.unique(grp)) / t)
def eta2_null(z, grp, n=20, seed=0):
    rng = np.random.default_rng(seed); return float(np.mean([eta2(z, rng.permutation(grp)) for _ in range(n)]))
def r2(Y, D):
    Yc = Y - Y.mean(0); return float(1 - (residual(Yc, D) ** 2).sum() / (Yc ** 2).sum())

Xq = residual(X.astype(np.float64), onehot(g))                    # question removed (as in pca_persona_only.py)
Z, V, lam = pca(Xq)
res = dict(n=int(len(IP)), layer=L, slot=S, n_groups_t1=len(set(t1)), n_groups_t2=len(set(t2)),
           top_openers=Counter(t1).most_common(15), components=[])
print("\n1. opener eta^2 on the question-demeaned PCs (null = shuffled labels, same group sizes)")
for i in range(K):
    z = Z[:, i]; row = dict(pc=i + 1, var=round(float(lam[i] / lam.sum()), 4),
        eta2_first_token=round(eta2(z, t1), 3), null_first_token=round(eta2_null(z, t1), 3),
        eta2_first_two=round(eta2(z, t2), 3), null_first_two=round(eta2_null(z, t2), 3),
        eta2_family=round(eta2(z[lab], fm[lab]), 3))
    res["components"].append(row)
    print(f"  PC{i+1:<2d} {row['var']:.1%}  first token {row['eta2_first_token']:.3f} (null {row['null_first_token']:.3f}) | "
          f"first two {row['eta2_first_two']:.3f} (null {row['null_first_two']:.3f}) | family {row['eta2_family']:.3f}")

Dq, D1, D2 = onehot(g), onehot(g, t1), onehot(g, t2)
res["r2_total"] = dict(question=round(r2(X.astype(np.float64), Dq), 4),
                       question_plus_first_token=round(r2(X.astype(np.float64), D1), 4),
                       question_plus_first_two=round(r2(X.astype(np.float64), D2), 4))
Xl = Xq[lab]; res["r2_family_on_labelled"] = round(r2(Xl, onehot(fm[lab])), 4)
res["r2_first_two_on_labelled"] = round(r2(Xl, onehot(t2[lab])), 4)
print(f"\n2. share of the whole state explained: {res['r2_total']} | on family-labelled rows, after question: "
      f"family {res['r2_family_on_labelled']} vs first-two-tokens {res['r2_first_two_on_labelled']}")

Xr = residual(X.astype(np.float64), D2)                             # question AND opener removed
Zr, Vr, lamr = pca(Xr)
sd = Zr.std(0); FAMS = [f for f, n in Counter(fm[lab]).most_common() if n >= 30]
even = g % 2 == 0; _, Va, _ = pca(Xr[even], 5); _, Vb, _ = pca(Xr[~even], 5)
res["residual"] = dict(
    var=[round(float(v), 4) for v in (lamr / lamr.sum())[:K]],
    participation_ratio=round(float(lamr.sum() ** 2 / (lamr ** 2).sum()), 1),
    split_half_abs_cos=[round(float(abs(Va[i] @ Vb[i])), 3) for i in range(5)],
    split_half_best_match=[round(float(np.abs(Va[i] @ Vb.T).max()), 3) for i in range(5)],
    cos_with_question_only_pcs=[[round(float(abs(Vr[i] @ V[j])), 3) for j in range(5)] for i in range(5)],
    eta2_family=[round(eta2(Zr[lab, i], fm[lab]), 3) for i in range(5)],
    family_means_sd=[{f: round(float(Zr[fm == f, i].mean() / sd[i]), 2) for f in FAMS} for i in range(5)])
print("\n3. residual PCA (question + first two tokens removed)")
print(f"  var PC1-5 {res['residual']['var'][:5]} | PR {res['residual']['participation_ratio']}")
print(f"  split-half |cos| {res['residual']['split_half_abs_cos']} | best match {res['residual']['split_half_best_match']}")
print(f"  family eta2 PC1-5 {res['residual']['eta2_family']}")
for i in range(5):
    fmx = sorted(res["residual"]["family_means_sd"][i].items(), key=lambda t: t[1])
    print(f"  PC{i+1}: low {', '.join(f'{f} {v:+.2f}' for f, v in fmx[:3])} | high {', '.join(f'{f} {v:+.2f}' for f, v in fmx[-3:])} "
          f"| |cos| to old PC1-5 {res['residual']['cos_with_question_only_pcs'][i]}")
json.dump(res, open(OUT, "w"), indent=1, default=str); print("\nresults ->", OUT)
