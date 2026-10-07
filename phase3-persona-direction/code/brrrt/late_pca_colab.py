#!/usr/bin/env python3
"""Persona-only PCA at every captured reply position (R4 ... R64), run on the VM next to late_L20.npy because the
Colab download path is too slow for the 540 MB of activations. Combines pca_persona_only.py, pca_persona_opener.py
and pca_persona_poles.py; same rows and labels. SVDs on the GPU in float32.

Inputs (cwd): late_L20.npy, late_L20_meta.json (from capture_late_slots.py), gen.jsonl, judge_p1.json, judge_p2.json.
Output: late_pca_results.json (small), late_pca_scores.npz (top-16 question-demeaned PC scores per slot).
"""
import json, re, numpy as np, torch
from collections import Counter
from transformers import AutoTokenizer

X = np.load("late_L20.npy", mmap_mode="r"); meta = json.load(open("late_L20_meta.json"))
ROWS, SLOTS = np.array(meta["rows"]), meta["slots"]
gen = [json.loads(l) for l in open("gen.jsonl")]
p1 = json.load(open("judge_p1.json")); p2 = json.load(open("judge_p2.json"))
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
persona = np.array([bool(p1[r]["persona"]) and bool(p2[r]["persona"]) and not (is_cot(p1[r]) or is_cot(p2[r])) for r in ROWS])
prompt = np.array([gen[r]["prompt_idx"] for r in ROWS])
f1 = [fam(p1[r].get("persona_label")) for r in ROWS]; f2 = [fam(p2[r].get("persona_label")) for r in ROWS]
family = np.array([u if (u is not None and u == v) else "unlabelled" for u, v in zip(f1, f2)], object)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def bucket(keys, minc=20):
    c = Counter(keys); return np.array([k if c[k] >= minc else "other" for k in keys], object)
open1 = [tok.decode(gen[r]["ids"][:1]) for r in ROWS]; open2 = [tok.decode(gen[r]["ids"][:2]) for r in ROWS]
QTEXT = {gen[r]["prompt_idx"]: gen[r]["prompt"] for r in ROWS}
dev = "cuda"

def demean(Y, g):
    Y = Y.clone()
    for v in np.unique(g):
        m = torch.tensor(g == v, device=dev); Y[m] -= Y[m].mean(0)
    return Y
def pca(Y):
    Yc = Y - Y.mean(0); U, S, Vh = torch.linalg.svd(Yc, full_matrices=False); lam = (S ** 2).cpu().numpy()
    return (U * S).cpu().numpy(), Vh.cpu().numpy(), lam
def dims(lam):
    c = np.cumsum(lam) / lam.sum()
    return dict(pr=round(float(lam.sum() ** 2 / (lam ** 2).sum()), 1), n50=int(np.searchsorted(c, .5) + 1), n80=int(np.searchsorted(c, .8) + 1))
def eta2(z, grp):
    gm = z.mean(); t = ((z - gm) ** 2).sum()
    return round(float(sum((grp == v).sum() * (z[grp == v].mean() - gm) ** 2 for v in np.unique(grp)) / t), 3)
def eta2_null(z, grp, n=10):
    rng = np.random.default_rng(0); return round(float(np.mean([eta2(z, rng.permutation(grp)) for _ in range(n)])), 3)
def onehot(*cats):
    cols = [np.ones((len(cats[0]), 1))]
    for c in cats:
        vals = sorted(set(c))[1:]; cols.append(np.stack([c == v for v in vals], 1).astype(np.float32))
    return torch.tensor(np.concatenate(cols, 1), device=dev, dtype=torch.float32)
def residual(Y, D):
    beta = torch.linalg.lstsq(D, Y).solution; return Y - D @ beta
def split_half(Y, g, k=5):
    ev = torch.tensor(g % 2 == 0, device=dev); _, Va, _ = pca(Y[ev]); _, Vb, _ = pca(Y[~ev])
    return [round(float(abs(Va[i] @ Vb[i])), 3) for i in range(k)], [round(float(np.abs(Va[i] @ Vb[:k].T).max()), 3) for i in range(k)]
def poles(z, idx, n=6, cap=2):
    o = np.argsort(z); out = {}
    for name, order in (("low", o), ("high", o[::-1])):
        seen, items = Counter(), []
        for j in order:
            q = prompt[idx[j]]
            if seen[q] >= cap: continue
            seen[q] += 1; r = ROWS[idx[j]]
            txt = tok.decode(gen[r]["ids"], skip_special_tokens=True).replace("\n", " ⏎ ")
            items.append(dict(sd=round(float(z[j] / z.std()), 2), label=p1[r].get("persona_label"), q=QTEXT[prompt[idx[j]]], text=txt[:220]))
            if len(items) == n: break
        out[name] = items
    return out

res, scores, V3 = {"slots": SLOTS, "n_rows": int(len(ROWS))}, {}, {}
for j, k in enumerate(SLOTS):
    A = np.asarray(X[:, j], np.float32); ok = ~np.isnan(A[:, 0])
    ip = np.flatnonzero(persona & ok); ia = np.flatnonzero(~persona & ok)
    Y = torch.tensor(A[ip], device=dev); g = prompt[ip]; fm = family[ip]; lab = fm != "unlabelled"
    t1, t2 = bucket([open1[i] for i in ip]), bucket([open2[i] for i in ip])
    Yq = demean(Y, g); Z, V, lam = pca(Yq)
    sh, shb = split_half(Yq, g)
    comps = []
    for i in range(5):
        z = Z[:, i]; fmeans = {f: round(float(z[fm == f].mean() / z.std()), 2) for f, c in Counter(fm[lab]).items() if c >= 30}
        comps.append(dict(pc=i + 1, var=round(float(lam[i] / lam.sum()), 4), eta2_first=eta2(z, t1), null_first=eta2_null(z, t1),
                          eta2_first2=eta2(z, t2), eta2_family=eta2(z[lab], fm[lab]), family_means_sd=fmeans))
    Yr = residual(Y, onehot(g, t2)); Zr, Vr, lamr = pca(Yr); shr, shrb = split_half(Yr, g)
    # matched-n dimensionality vs assistant replies
    Ya = demean(torch.tensor(A[ia], device=dev), prompt[ia]); _, _, lamA = pca(Ya)
    sub = np.random.default_rng(0).choice(len(ip), size=len(ia), replace=False)
    _, _, lamP = pca(demean(Y[torch.tensor(sub, device=dev)], g[sub]))
    res[f"R{k}"] = dict(
        n_persona=int(len(ip)), n_assistant=int(len(ia)), dims=dims(lam), var_top10=[round(float(v), 4) for v in lam[:10] / lam.sum()],
        split_half=sh, split_half_best=shb, components=comps,
        residual=dict(dims=dims(lamr), var_top5=[round(float(v), 4) for v in lamr[:5] / lamr.sum()], split_half=shr, split_half_best=shrb,
                      cos_with_q_only=[[round(float(abs(Vr[a] @ V[b])), 3) for b in range(5)] for a in range(5)],
                      eta2_family=[eta2(Zr[lab, i], fm[lab]) for i in range(5)]),
        matched_n=dict(n=int(len(ia)), assistant=dims(lamA), persona=dims(lamP)),
        poles={f"PC{i+1}": poles(Z[:, i], ip) for i in range(3)})
    V3[k] = V[:3]
    scores[f"R{k}_rows"] = ROWS[ip]; scores[f"R{k}_Z"] = Z[:, :16].astype(np.float32)
    r = res[f"R{k}"]
    print(f"R{k:<3d} n={len(ip)} PR {r['dims']} | var PC1-3 {r['var_top10'][:3]} | split {sh[:3]} | "
          f"first-token eta2 PC1-3 {[c['eta2_first'] for c in comps[:3]]} | family eta2 {[c['eta2_family'] for c in comps[:3]]} | "
          f"resid split {shr[:3]} | matched PR persona {r['matched_n']['persona']['pr']} vs asst {r['matched_n']['assistant']['pr']}", flush=True)
    del Y, Yq, Yr, Ya; torch.cuda.empty_cache()

# do the axes carry across positions? best |cos| of each PC1-3 with PC1-3 at the next captured slot
res["cross_slot_best_cos"] = {f"R{a}->R{b}": [round(float(np.abs(V3[a][i] @ V3[b].T).max()), 3) for i in range(3)]
                              for a, b in zip(SLOTS[:-1], SLOTS[1:])}
print("cross-slot", res["cross_slot_best_cos"])
json.dump(res, open("late_pca_results.json", "w"), indent=1, default=str)
np.savez_compressed("late_pca_scores.npz", **scores); print("saved")
