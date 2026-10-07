#!/usr/bin/env python3
"""Phase 6 analysis: is a system-prompted persona represented like a steered one?

Sections (each writes into results/analysis.json; run all by default, or pick with --only a,b,...):
  transfer  persona-vs-assistant mass-mean probe trained on one source, tested on the other. Leave-questions-out
            (GroupKFold over the 40 questions: the probe never sees the test questions in either source), pooled
            AUROC and within-question macro AUROC (phase 3's metric that cannot inherit the question).
  match     same character, different source. Per label (and per family), the question-demeaned centroid of the
            steered rows vs the centroid of their prompted twins. Nearest-centroid matching accuracy across sources,
            vs chance, vs a within-question label-shuffle null, vs a within-source split-half ceiling.
  axis      projections on phase 1's assistant axis (demo/directions.npz: assistant_axis_l20) per group and slot.
  pca       joint PCA of steered + prompted personas: source separation per component; principal angles between
            each source's persona-only PCA subspace; persona-only PCA summary of each source.
  openers   first-token statistics per group (the stage-direction confound).
"""
import json, argparse, time
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from common import P6, ROOT
from data import load_all

ap = argparse.ArgumentParser(); ap.add_argument("--only", default="transfer,match,axis,pca,openers")
ap.add_argument("--perms", type=int, default=200); ap.add_argument("--out", default="analysis.json"); a = ap.parse_args()
OUT = P6 / a.out
RES = json.load(open(OUT)) if OUT.exists() else {}
D = load_all(); G = D["G"]; s3, s6 = D["s3"], D["s6"]; q3, q6 = D["q3"], D["q6"]
t0 = time.time()
def log(*x): print(f"[{time.time()-t0:5.0f}s]", *x, flush=True)
log({k: len(v) for k, v in G.items()})
FEAT_CELLS = [(8, "R4"), (20, "P"), (20, "R1"), (20, "R4"), (20, "R8"), (32, "R4"), (32, "R8")]
LATE_CELLS = [(20, "R16"), (20, "R32"), (20, "R64")]
def X3(rows, L, S): return s3.get(rows, L, S)
def X6(rows, L, S): return s6.get(rows, L, S)
def finite(X): return np.isfinite(X).all(1)


def macro_within(s, y, g, mn=5):
    per = [roc_auc_score(y[g == v], s[g == v]) for v in np.unique(g) if y[g == v].sum() >= mn and (1 - y[g == v]).sum() >= mn]
    return (float(np.mean(per)) if per else float("nan")), len(per)


def summ(s, y, g):
    ok = np.isfinite(s); s, y, g = s[ok], y[ok], g[ok]
    m, nq = macro_within(s, y, g)
    return dict(pooled=round(float(roc_auc_score(y, s)), 4), within_q=round(m, 4), n_q=nq, n_pos=int(y.sum()), n_neg=int((1 - y).sum()))


def transfer():
    out = {}
    for L, S in FEAT_CELLS + LATE_CELLS:
        A = dict(sp=X3(G["S_P"], L, S), sa=X3(G["S_A"], L, S), qp=X6(G["Q_Pj"], L, S), ta=X6(G["T_A"], L, S), na=X6(G["N_A"], L, S))
        Q = dict(sp=q3[G["S_P"]], sa=q3[G["S_A"]], qp=q6[G["Q_Pj"]], ta=q6[G["T_A"]], na=q6[G["N_A"]])
        ok = {k: finite(v) for k, v in A.items()}; A = {k: v[ok[k]] for k, v in A.items()}; Q = {k: v[ok[k]] for k, v in Q.items()}
        # (key, which direction, test positives, test negatives). Directions: s = steered persona - steered assistant;
        # t = prompted persona - template assistant; n = prompted persona - neutral assistant.
        TESTS = (("st_on_s", "s", "sp", "sa"), ("st_on_qt", "s", "qp", "ta"), ("st_on_qn", "s", "qp", "na"),
                 ("pt_on_s", "t", "sp", "sa"), ("pt_on_qt", "t", "qp", "ta"), ("pn_on_s", "n", "sp", "sa"), ("pn_on_qn", "n", "qp", "na"))
        DIRS = dict(s=("sp", "sa"), t=("qp", "ta"), n=("qp", "na"))
        TE = {key: (np.concatenate([A[p], A[n]]), np.r_[Q[p], Q[n]], np.r_[np.ones(len(Q[p])), np.zeros(len(Q[n]))].astype(int)) for key, _, p, n in TESTS}
        sc = {key: np.full(len(TE[key][1]), np.nan) for key in TE}
        for _, te_q in GroupKFold(n_splits=10).split(np.arange(40), groups=np.arange(40)):
            d = {k: A[p][~np.isin(Q[p], te_q)].mean(0) - A[n][~np.isin(Q[n], te_q)].mean(0) for k, (p, n) in DIRS.items()}
            for key, dk, _, _ in TESTS:
                Xc, qc, _ = TE[key]; m = np.isin(qc, te_q); sc[key][m] = Xc[m] @ d[dk]
        r = {key: summ(sc[key], TE[key][2], TE[key][1]) for key in TE}
        d_s = A["sp"].mean(0) - A["sa"].mean(0); d_t = A["qp"].mean(0) - A["ta"].mean(0); d_n = A["qp"].mean(0) - A["na"].mean(0)
        cs = lambda u, v: round(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)), 3)
        # random-direction baseline for the cosine: two independent 4096-d directions sit at |cos| ~ 0.016
        r["cos_dir"] = dict(steer_vs_prompt_template=cs(d_s, d_t), steer_vs_prompt_neutral=cs(d_s, d_n), template_vs_neutral=cs(d_t, d_n))
        out[f"L{L}{S}"] = r
        log(f"L{L:<2d}{S:4s} steer->steer {r['st_on_s']['within_q']:.3f} | steer->prompt(T) {r['st_on_qt']['within_q']:.3f} (N) {r['st_on_qn']['within_q']:.3f}"
            f" | prompt(T)->steer {r['pt_on_s']['within_q']:.3f} | prompt->prompt {r['pt_on_qt']['within_q']:.3f} | cos {r['cos_dir']}")
    RES["transfer"] = out


def demean(X, g):
    X = X.copy()
    for v in np.unique(g): X[g == v] -= X[g == v].mean(0)
    return X


def centroids(X, lab, keys):
    C = np.stack([X[lab == k].mean(0) for k in keys]); return C / np.linalg.norm(C, axis=1, keepdims=True)


def nn_acc(CA, CB):
    S = CA @ CB.T; return float((S.argmax(1) == np.arange(len(CA))).mean()), float((S.argmax(0) == np.arange(len(CA))).mean()), \
        float(np.diag(S).mean()), float(S[~np.eye(len(S), dtype=bool)].mean())


def match_once(XS, XQ, labS, labQ, gS, gQ, keys, perms, rng):
    CS, CQ = centroids(XS, labS, keys), centroids(XQ, labQ, keys)
    s2q, q2s, diag, off = nn_acc(CS, CQ)
    null = []
    for _ in range(perms):
        lp = labQ.copy()
        for v in np.unique(gQ): m = gQ == v; lp[m] = rng.permutation(lp[m])
        null.append(np.mean(nn_acc(CS, centroids(XQ, lp, keys))[:2]))
    # ceiling: steered rows split in two random halves, matched to each other the same way
    h = rng.random(len(labS)) < 0.5; ok = all((labS[h] == k).any() and (labS[~h] == k).any() for k in keys)
    ceil = np.mean(nn_acc(centroids(XS[h], labS[h], keys), centroids(XS[~h], labS[~h], keys))[:2]) if ok else float("nan")
    return dict(K=len(keys), chance=round(1 / len(keys), 4), steer_to_prompt=round(s2q, 4), prompt_to_steer=round(q2s, 4),
                null_mean=round(float(np.mean(null)), 4), null_p95=round(float(np.percentile(null, 95)), 4), null_max=round(float(np.max(null)), 4),
                p_value=round(float((np.sum(np.array(null) >= (s2q + q2s) / 2) + 1) / (len(null) + 1)), 4),
                ceiling_steer_split_half=round(float(ceil), 4), cos_matched=round(diag, 4), cos_unmatched=round(off, 4))


def match():
    rng = np.random.default_rng(0); out = {}
    SP, QP = G["S_P"], G["Q_P"]
    # twin order: QP rows point back to their steered row; check the pairing is one-to-one and labels agree
    twin = {int(D["src6"][r]): int(r) for r in QP}; assert sorted(twin) == sorted(map(int, SP))
    QPo = np.array([twin[int(r)] for r in SP])
    labS = D["lab3"][SP]; labQ = D["lab6_req"][QPo]; assert (labS == labQ).all()
    famS = D["fam3"][SP]; famQ = D["fam6_req"][QPo]
    gS = q3[SP]; gQ = q6[QPo]; assert (gS == gQ).all()
    matchok = match_flags(QPo)
    for L, S in [(20, "R4"), (20, "R8"), (20, "R16"), (20, "R32"), (20, "R64"), (8, "R8"), (32, "R8"), (20, "P"), (20, "R1")]:
        XS, XQ = X3(SP, L, S), X6(QPo, L, S)
        ok = finite(XS) & finite(XQ)
        XS, XQ = demean(XS[ok], gS[ok]), demean(XQ[ok], gQ[ok])
        r = {}
        for name, lS, lQ, mn in (("label_n20", labS[ok], labQ[ok], 20), ("label_n10", labS[ok], labQ[ok], 10),
                                 ("family", famS[ok], famQ[ok], 20)):
            c = Counter(lS); keys = sorted(k for k, n in c.items() if n >= mn and k != "unlabelled")
            m = np.isin(lS, keys)
            r[name] = match_once(XS[m], XQ[m], lS[m], lQ[m], gS[ok][m], gQ[ok][m], keys, a.perms if name != "label_n10" else 50, rng)
        # family, keeping only prompted twins the judge says actually voice the requested persona (match pass == 2)
        mk = matchok[ok]; c = Counter(famS[ok][mk]); keys = sorted(k for k, n in c.items() if n >= 20 and k != "unlabelled")
        m = mk & np.isin(famS[ok], keys)
        if len(keys) > 2: r["family_judged_match"] = match_once(XS[m], XQ[m], famS[ok][m], famQ[ok][m], gS[ok][m], gQ[ok][m], keys, a.perms, rng)
        out[f"L{L}{S}"] = r
        log(f"match L{L}{S}: " + " | ".join(f"{k}: K={v['K']} s->p {v['steer_to_prompt']:.2f} p->s {v['prompt_to_steer']:.2f} null {v['null_mean']:.2f}/{v['null_p95']:.2f} ceil {v['ceiling_steer_split_half']:.2f}" for k, v in r.items()))
    RES["match"] = out


def match_flags(rows6):
    jm = P6 / "hf_dl/judge/judge_match.json"
    if not jm.exists(): return np.ones(len(rows6), bool)
    m = {d["row"]: d["match"] for d in json.load(open(jm)) if d["kind"] == "prompted"}
    return np.array([m.get(int(r)) == 2 for r in rows6])


def axis():
    u = np.load(ROOT / "demo/directions.npz")["assistant_axis_l20"].astype(np.float32); u /= np.linalg.norm(u)
    out = {}
    for S in ["P", "R1", "R4", "R8", "R16", "R32", "R64"]:
        r = {}
        for name, src, rows in (("S_P", s3, G["S_P"]), ("S_A", s3, G["S_A"]), ("C", s3, G["C"]), ("Q_P", s6, G["Q_P"]), ("Q_Pj", s6, G["Q_Pj"]),
                                ("T_A", s6, G["T_A"]), ("N_A", s6, G["N_A"])):
            p = src.get(rows, 20, S) @ u; p = p[np.isfinite(p)]
            r[name] = dict(mean=round(float(p.mean()), 3), sd=round(float(p.std()), 3), n=int(len(p)))
        gap = r["S_A"]["mean"] - r["S_P"]["mean"]
        for v in r.values(): v["pos"] = round((v["mean"] - r["S_P"]["mean"]) / gap, 3)      # 0 = steered persona, 1 = steered assistant
        out[S] = r
        log(f"axis L20 {S}: " + " ".join(f"{k} {v['mean']:+.2f}({v['pos']:+.2f})" for k, v in r.items()))
    RES["axis"] = dict(note="projection on unit assistant_axis_l20; pos = (mean - S_P mean) / (S_A mean - S_P mean)", slots=out)


def pca_svd(X, k):
    Xc = X - X.mean(0); U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False); lam = Sv ** 2
    return Vt[:k], (U * Sv)[:, :k], lam


def dims(lam):
    c = np.cumsum(lam) / lam.sum()
    return dict(pr=round(float(lam.sum() ** 2 / (lam ** 2).sum()), 1), n50=int(np.searchsorted(c, .5) + 1), n80=int(np.searchsorted(c, .8) + 1))


def eta2(z, grp):
    gm = z.mean(); t = ((z - gm) ** 2).sum()
    return float(sum((grp == v).sum() * (z[grp == v].mean() - gm) ** 2 for v in np.unique(grp)) / t)


def angles(A, B): return np.linalg.svd(A @ B.T, compute_uv=False)


def pca():
    rng = np.random.default_rng(0); out = {}
    SP, QP = G["S_P"], G["Q_P"]
    for L, S in [(20, "R4"), (20, "R8"), (20, "R32"), (20, "R64"), (32, "R8")]:
        XS, XQ = X3(SP, L, S), X6(QP, L, S); okS, okQ = finite(XS), finite(XQ)
        XS, XQ, gS, gQ = XS[okS], XQ[okQ], q3[SP][okS], q6[QP][okQ]
        fS, fQ = D["fam3"][SP][okS], D["famj6"][QP][okQ]
        r = {}
        # joint PCA; question means removed over BOTH sources together, so the source offset is kept
        Xj = demean(np.concatenate([XS, XQ]), np.r_[gS, gQ]); src = np.r_[np.zeros(len(XS)), np.ones(len(XQ))].astype(int)
        V, Z, lam = pca_svd(Xj, 10)
        r["joint"] = [dict(pc=i + 1, var=round(float(lam[i] / lam.sum()), 4), source_auroc=round(float(max(roc_auc_score(src, Z[:, i]), 1 - roc_auc_score(src, Z[:, i]))), 3),
                           eta2_source=round(eta2(Z[:, i], src), 3), eta2_question=round(eta2(Z[:, i], np.r_[gS, gQ]), 3)) for i in range(10)]
        # how much of the total (question-demeaned) variance is the source offset
        mu = Xj[src == 0].mean(0) - Xj[src == 1].mean(0)
        r["source_offset_share"] = round(float(len(XS) * len(XQ) / len(Xj) ** 2 * (mu @ mu) / ((Xj - Xj.mean(0)) ** 2).sum(1).mean()), 4)
        # leave-questions-out source decoding with a mass-mean direction
        s = np.full(len(Xj), np.nan); gj = np.r_[gS, gQ]
        for tr_q, te_q in GroupKFold(n_splits=10).split(np.arange(40), groups=np.arange(40)):
            tr = ~np.isin(gj, te_q); d = Xj[tr & (src == 1)].mean(0) - Xj[tr & (src == 0)].mean(0); s[~tr] = Xj[~tr] @ d
        r["source_decoding_auroc"] = summ(s, src, gj)
        # each source on its own: question-demeaned persona-only PCA, and principal angles between the subspaces
        XSd, XQd = demean(XS, gS), demean(XQ, gQ)
        VS, ZS, lS = pca_svd(XSd, 20); VQ, ZQ, lQ = pca_svd(XQd, 20)
        h = rng.random(len(XSd)) < .5; VA, _, _ = pca_svd(XSd[h], 20); VB, _, _ = pca_svd(XSd[~h], 20)
        h2 = rng.random(len(XQd)) < .5; VC, _, _ = pca_svd(XQd[h2], 20); VD, _, _ = pca_svd(XQd[~h2], 20)
        R1 = np.linalg.qr(rng.standard_normal((4096, 20)))[0].T; R2 = np.linalg.qr(rng.standard_normal((4096, 20)))[0].T
        ang = {}
        for k in (3, 5, 10, 20):
            ang[f"k{k}"] = dict(steer_vs_prompt=[round(float(x), 3) for x in angles(VS[:k], VQ[:k])],
                                steer_half_vs_half=[round(float(x), 3) for x in angles(VA[:k], VB[:k])],
                                prompt_half_vs_half=[round(float(x), 3) for x in angles(VC[:k], VD[:k])],
                                random=[round(float(x), 3) for x in angles(R1[:k], R2[:k])])
        r["principal_angle_cosines"] = ang
        # top components of each source: matched |cos|, variance captured by the other source's top-10 subspace
        r["top_pc_abs_cos"] = [round(float(abs(VS[i] @ VQ[i])), 3) for i in range(5)]
        varcap = lambda X, V: float(((X @ V.T) ** 2).sum() / (X ** 2).sum())
        r["variance_captured_top10"] = dict(prompt_by_own=round(varcap(XQd, VQ[:10]), 4), prompt_by_steer=round(varcap(XQd, VS[:10]), 4),
                                            steer_by_own=round(varcap(XSd, VS[:10]), 4), steer_by_prompt=round(varcap(XSd, VQ[:10]), 4))
        for name, Z_, lam_, f_ in (("steered", ZS, lS, fS), ("prompted", ZQ, lQ, fQ)):
            r[f"own_{name}"] = dict(n=len(Z_), dims=dims(lam_), pcs=[dict(pc=i + 1, var=round(float(lam_[i] / lam_.sum()), 4),
                                    eta2_family=round(eta2(Z_[:, i], f_), 3)) for i in range(5)])
        out[f"L{L}{S}"] = r
        log(f"pca L{L}{S}: source AUROC lq-out {r['source_decoding_auroc']} | offset share {r['source_offset_share']} | PC1-3 src AUROC {[c['source_auroc'] for c in r['joint'][:3]]}"
            f" | angles k10 s-p {ang['k10']['steer_vs_prompt'][:3]}.. min {ang['k10']['steer_vs_prompt'][-1]} half {ang['k10']['steer_half_vs_half'][-1]} | dims S {r['own_steered']['dims']} Q {r['own_prompted']['dims']}")
    RES["pca"] = out


def openers():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); out = {}
    for name, gen, rows in (("S_P", D["gen3"], G["S_P"]), ("S_A", D["gen3"], G["S_A"]), ("C", D["gen3"], G["C"]), ("Q_P", D["gen6"], G["Q_P"]),
                            ("Q_Pj", D["gen6"], G["Q_Pj"]), ("T_all", D["gen6"], G["T_all"]), ("N_all", D["gen6"], G["N_all"])):
        c = Counter(tok.decode(gen[r]["ids"][:1]) for r in rows); n = len(rows)
        out[name] = dict(n=n, top=[(k, v, round(v / n, 3)) for k, v in c.most_common(12)], distinct=len(c))
    sp = {k for k, _, _ in out["S_P"]["top"]}
    out["note"] = "first reply token per group"
    RES["openers"] = out
    for k, v in out.items():
        if k != "note": log(f"openers {k}: " + ", ".join(f"{t!r} {p:.0%}" for t, _, p in v["top"][:8]))


def openerctl():
    """The 'Ah' confound. 73 % of prompted personas open with 'Ah' and 34 % of template assistants do; no steered
    persona does. Repeat the transfer test inside one opener: prompted persona vs template assistant replies that BOTH
    open with 'Ah' (and, separately, both open with something other than 'Ah'/'Oh'), so the first word cannot separate them.
    The steered direction is still fit leave-questions-out on steered rows only."""
    first = np.array([g["ids"][0] if g["ids"] else -1 for g in D["gen6"]])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    AH = {t for t in set(first.tolist()) if t >= 0 and tok.decode([t]).strip() == "Ah"}
    OH = {t for t in set(first.tolist()) if t >= 0 and tok.decode([t]).strip() == "Oh"}
    isah = np.isin(first, list(AH)); other = ~isah & ~np.isin(first, list(OH))
    out = {}
    for L, S in [(20, "R1"), (20, "R4"), (20, "R8"), (20, "R16"), (20, "R32"), (20, "R64")]:
        sp, sa = X3(G["S_P"], L, S), X3(G["S_A"], L, S); qs, qa = q3[G["S_P"]], q3[G["S_A"]]
        oks, oka = finite(sp), finite(sa); sp, sa, qs, qa = sp[oks], sa[oka], qs[oks], qa[oka]
        r = {}
        for name, msk in (("both_open_Ah", isah), ("both_open_other", other)):
            P_, N_ = G["Q_Pj"][msk[G["Q_Pj"]]], G["T_A"][msk[G["T_A"]]]
            XP, XN = X6(P_, L, S), X6(N_, L, S); okp, okn = finite(XP), finite(XN)
            Xc = np.concatenate([XP[okp], XN[okn]]); qc = np.r_[q6[P_][okp], q6[N_][okn]]
            y = np.r_[np.ones(okp.sum()), np.zeros(okn.sum())].astype(int); s = np.full(len(y), np.nan); s2 = np.full(len(y), np.nan)
            for _, te_q in GroupKFold(n_splits=10).split(np.arange(40), groups=np.arange(40)):
                d = sp[~np.isin(qs, te_q)].mean(0) - sa[~np.isin(qa, te_q)].mean(0); m = np.isin(qc, te_q); s[m] = Xc[m] @ d
                tr = ~m; dq = Xc[tr & (y == 1)].mean(0) - Xc[tr & (y == 0)].mean(0); s2[m] = Xc[m] @ dq
            r[name] = dict(steer_dir=summ(s, y, qc), prompt_dir_within=summ(s2, y, qc))
        out[f"L{L}{S}"] = r
        log(f"openerctl L{L}{S}: " + " | ".join(f"{k}: steer->prompt {v['steer_dir']['within_q']:.3f} (pooled {v['steer_dir']['pooled']:.3f}, {v['steer_dir']['n_pos']}/{v['steer_dir']['n_neg']}, q {v['steer_dir']['n_q']}) prompt->prompt {v['prompt_dir_within']['within_q']:.3f}" for k, v in r.items()))
    RES["openerctl"] = out


def srcctl():
    """Is the steered-vs-prompted separation about HOW THE PERSONA IS BUILT, or just about the context (a system prompt
    in context / a steering trace at the prompt positions)? Decode source among ASSISTANT replies too. If assistants
    separate as easily as personas, the source signal is context, not persona. Personas are subsampled to the assistant
    counts so the direction estimates are equally noisy. Leave-questions-out mass-mean, within-question AUROC.
      persona     S_P vs Q_Pj           (steered persona vs prompted persona)
      assistant   S_A vs T_A            (steered assistant vs template-prompt assistant: same template, label 'helpful AI assistant')
      assistant_n S_A vs N_A            (steered assistant vs 'You are a helpful assistant.')
      sysprompt   C   vs N_A            (no steering either side: no system prompt vs neutral system prompt)"""
    rng = np.random.default_rng(0); out = {}
    nA = min(len(G["S_A"]), len(G["T_A"]))
    sub = lambda r, n: np.sort(rng.choice(r, n, replace=False))
    PAIRS = dict(persona=((s3, sub(G["S_P"], nA)), (s6, sub(G["Q_Pj"], nA))), assistant=((s3, sub(G["S_A"], nA)), (s6, sub(G["T_A"], nA))),
                 assistant_n=((s3, sub(G["S_A"], nA)), (s6, sub(G["N_A"], nA))), sysprompt=((s3, G["C"]), (s6, sub(G["N_A"], len(G["C"])))))
    for L, S in [(20, "P"), (20, "R1"), (20, "R4"), (20, "R8"), (20, "R32"), (20, "R64"), (32, "R8"), (8, "R8")]:
        r = {}
        for name, ((sa, ra), (sb, rb)) in PAIRS.items():
            XA, XB = sa.get(ra, L, S), sb.get(rb, L, S); qa = (q3 if sa is s3 else q6)[ra]; qb = (q3 if sb is s3 else q6)[rb]
            oa, ob = finite(XA), finite(XB); X = np.concatenate([XA[oa], XB[ob]]); g = np.r_[qa[oa], qb[ob]]
            y = np.r_[np.zeros(oa.sum()), np.ones(ob.sum())].astype(int); s = np.full(len(y), np.nan)
            for _, te_q in GroupKFold(n_splits=10).split(np.arange(40), groups=np.arange(40)):
                tr = ~np.isin(g, te_q); d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); s[~tr] = X[~tr] @ d
            mu = X[y == 1].mean(0) - X[y == 0].mean(0); Xd = demean(X, g)
            r[name] = dict(summ(s, y, g), offset_norm_over_sd=round(float(np.linalg.norm(mu) / np.sqrt(((Xd - Xd.mean(0)) ** 2).sum(1).mean())), 3))
        out[f"L{L}{S}"] = r
        log(f"srcctl L{L}{S}: " + " | ".join(f"{k} {v['within_q']:.3f} (offset {v['offset_norm_over_sd']:.2f}, n {v['n_neg']}/{v['n_pos']})" for k, v in r.items()))
    RES["srcctl"] = out


for sec in a.only.split(","):
    globals()[sec](); json.dump(RES, open(OUT, "w"), indent=1, ensure_ascii=False); log(f"{sec} -> {OUT}")
