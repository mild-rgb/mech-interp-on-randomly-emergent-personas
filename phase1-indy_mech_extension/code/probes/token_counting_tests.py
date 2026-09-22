#!/usr/bin/env python3
"""Are the debiased probes just token counting? Three tests (added 2026-09-14). Layer 20, with-prompt states, assistant
vs persona, unanimous labels, nulls excluded, leave-one-trigger-out everywhere. States are z-scored with the INLP cell
scaler so the saved nuisance bases (inlp_directions.npz) apply. Probe variants: raw mass-mean; language + broken
subspaces projected out; trigger + language + broken projected out.
 1. Within-identical-prefix AUROC: only pairs of rollouts whose first k token ids are identical, so a bag classifier
    cannot separate them. Every member of a prefix group is scored by ONE model fit on rollouts outside the group
    (leave-prefix-group-out; leave-one-out scoring is biased against the held-out point when near-duplicates remain in
    the fit, which sent a first version of this test below chance). Also run on per-arm-demeaned states so the trigger's
    prior cannot separate group members. Permutation p from shuffling labels within prefix groups. The bag scorer must
    come out at exactly 0.5 here; it is the sanity check.
 2. Incremental validity: LOAO held-out scores for the bag (first-k binary presence, C=0.3 liblinear) and the probe; a
    two-feature logistic combiner fit LOAO on those scores; likelihood-ratio test for the probe term; per-fold paired
    Wilcoxon of combined vs bag AUROC.
 3. Token-subspace removal: inside each fold, ridge-regress the state onto the bag vector, take the top-r singular
    directions of that map (the r directions that best predict token identity), project them out, refit the probe;
    reported beside a random r-dimensional subspace removed instead — both a random subspace of the full 4096-d space and
    a random subspace drawn inside the training data's span (the fair control: the data occupy <= n dimensions, token
    directions are fit from the data and lie in that span, random 4096-d directions mostly miss it).
Run from data/. Writes ../results/results_token_counting_tests.json."""
import json, collections, numpy as np, scipy.sparse as sp
from scipy.stats import wilcoxon, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = np.load("features_qwen_wide.npz")["X"]; meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json")); INLP = np.load("../directions/inlp_directions.npz")
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); V = 151936
yA = np.array([-1 if r["assistant_A"] is None else r["assistant_A"] for r in lab])
LAYER = 20; rng = np.random.default_rng(0); out = {}
def unit(v): return v / np.linalg.norm(v)
def mm(G, y): return G[y == 1].mean(0) - G[y == 0].mean(0)
def proj_out(G, B): return G if B is None or not len(B) else G - (G @ B.T) @ B
def gs(rows):
    B = None
    for w in rows:
        w = w.astype(np.float64).copy()
        if B is not None: w -= B.T @ (B @ w)
        n = np.linalg.norm(w)
        if n < 1e-8: continue
        B = (w / n)[None] if B is None else np.vstack([B, w / n])
    return B
def bag(k, keep):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(r["resp_ids"][:k]): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))[keep]
    return B[:, np.asarray(B.sum(0)).ravel() >= 2]
def loao(fn, y, g):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te; s[te] = fn(tr, te)
    return s
def auc(s, y): return float(roc_auc_score(y, s))
def perfold(s, y, g): return {a: auc(s[g == a], y[g == a]) for a in sorted(set(g)) if len(set(y[g == a])) > 1}
def within_prefix(s, y, groups):
    num = den = 0.0
    for g in groups:
        p, n = s[g][y[g] == 1], s[g][y[g] == 0]
        if len(p) and len(n): d = p[:, None] - n[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den
for slot, k in [("R1", 1), ("R2", 2), ("R4", 4)]:
    li, si = L.index(LAYER), S.index(slot); key = f"L{LAYER}_{slot}"
    F = X[:, li, si].astype(np.float32); keep = ~np.isnan(F[:, 0]) & ~is_null & (yA >= 0)
    mu, sc = INLP[key + "__scaler_mean"].astype(np.float32), INLP[key + "__scaler_scale"].astype(np.float32)
    Z = (F[keep] - mu) / sc; y = yA[keep]; g = arms[keep]; idx = np.where(keep)[0]
    B_lb = gs(np.vstack([INLP[key + "__language"], INLP[key + "__broken"]]).astype(np.float32))
    B_all = gs(np.vstack([INLP[key + "__trigger"], INLP[key + "__language"], INLP[key + "__broken"]]).astype(np.float32))
    variants = {"raw": None, "lang+broken removed": B_lb, "trigger+lang+broken removed": B_all}
    Bg = bag(k, keep); rec = dict(slot=slot, k=k, n=int(len(y)), n_bag_features=int(Bg.shape[1]), tests={})
    s_bag = loao(lambda tr, te: LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", solver="liblinear").fit(Bg[tr], y[tr]).decision_function(Bg[te]), y, g)
    scores = {"bag": s_bag}
    for vn, Bn in variants.items():
        Zv = proj_out(Z, Bn); scores[vn] = loao(lambda tr, te: Zv[te] @ mm(Zv[tr], y[tr]), y, g)
    # ---- test 1: within identical prefix ------------------------------------------------------------
    groups = collections.defaultdict(list)
    for j, i in enumerate(idx): groups[tuple(ids[i]["resp_ids"][:k])].append(j)
    groups = [np.array(v) for v in groups.values() if len(set(y[v])) > 1]
    npairs = int(sum((y[v] == 1).sum() * (y[v] == 0).sum() for v in groups))
    t1 = dict(n_groups=len(groups), n_rollouts=int(sum(len(v) for v in groups)), n_pairs=npairs)
    Zd = Z.copy()
    for a in set(g): Zd[g == a] -= Zd[g == a].mean(0)
    def lgo(fit_score):            # leave-prefix-group-out: one model per group, fit on everything outside it
        s = np.full(len(y), np.nan)
        for v in groups:
            tr = np.ones(len(y), bool); tr[v] = False; s[v] = fit_score(tr, v)
        return s
    g_scores = {"bag": lgo(lambda tr, te: LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", solver="liblinear").fit(Bg[tr], y[tr]).decision_function(Bg[te]))}
    for vn, Bn in variants.items():
        Zv = proj_out(Z, Bn); g_scores[vn] = lgo(lambda tr, te, Zv=Zv: Zv[te] @ mm(Zv[tr], y[tr]))
        Zvd = proj_out(Zd, Bn); g_scores[vn + " (arm-demeaned)"] = lgo(lambda tr, te, Zvd=Zvd: Zvd[te] @ mm(Zvd[tr], y[tr]))
    for name, s in g_scores.items():
        obs = within_prefix(s, y, groups); null = []
        for _ in range(2000):
            yp = y.copy()
            for v in groups: yp[v] = rng.permutation(y[v])
            null.append(within_prefix(s, yp, groups))
        null = np.array(null); t1[name] = dict(auroc=float(obs), perm_p=float((null >= obs).mean()), null_mean=float(null.mean()), null_sd=float(null.std()))
    rec["tests"]["within_prefix"] = t1
    # ---- test 2: incremental validity over the bag ----------------------------------------------------
    t2 = dict(bag_auroc=auc(s_bag, y))
    for vn in variants:
        s = scores[vn]; Xc = np.column_stack([s_bag, s]); Xc = (Xc - Xc.mean(0)) / Xc.std(0)
        comb = loao(lambda tr, te: LogisticRegression(C=1e6, max_iter=5000).fit(Xc[tr], y[tr]).decision_function(Xc[te]), y, g)
        m_full = LogisticRegression(C=1e6, max_iter=5000).fit(Xc, y); m_red = LogisticRegression(C=1e6, max_iter=5000).fit(Xc[:, :1], y)
        ll = lambda m, A: float(np.sum(y * np.log(m.predict_proba(A)[:, 1] + 1e-12) + (1 - y) * np.log(m.predict_proba(A)[:, 0] + 1e-12)))
        lr = 2 * (ll(m_full, Xc) - ll(m_red, Xc[:, :1])); pf_c, pf_b = perfold(comb, y, g), perfold(s_bag, y, g); ks = [a for a in pf_c if a in pf_b]
        d = np.array([pf_c[a] - pf_b[a] for a in ks])
        t2[vn] = dict(probe_auroc=auc(s, y), combined_auroc=auc(comb, y), probe_coef=float(m_full.coef_[0, 1]), lr_stat=float(lr), lr_p=float(chi2.sf(lr, 1)),
                      fold_gain_mean=float(d.mean()), folds_won=int((d > 0).sum()), n_folds=len(d), wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    rec["tests"]["incremental"] = t2
    # ---- test 3: remove the directions that predict token identity ----------------------------------------
    t3 = {}
    Bd = Bg.toarray().astype(np.float32)
    def ridge_W(Zt, Bt, lam): K = Zt @ Zt.T + lam * np.eye(len(Zt)); return Zt.T @ np.linalg.solve(K, Bt - Bt.mean(0))   # dual ridge, [4096, m]
    def token_r2(Zv, lam):         # held-out (LOAO) fraction of bag variance linearly predictable from the state
        num = den = 0.0
        for a in sorted(set(g)):
            te = g == a; tr = ~te; W = ridge_W(Zv[tr], Bd[tr], lam); pred = (Zv[te] - Zv[tr].mean(0)) @ W + Bd[tr].mean(0)
            num += ((Bd[te] - pred) ** 2).sum(); den += ((Bd[te] - Bd[tr].mean(0)) ** 2).sum()
        return float(1 - num / den)
    LAMS = [1e2, 1e3, 1e4, 1e5, 1e6]
    for vn in ["raw", "lang+broken removed"]:
        Zv = proj_out(Z, variants[vn]); r2s = {lam: token_r2(Zv, lam) for lam in LAMS}; lam = max(r2s, key=r2s.get)
        t3[vn] = {"r0": auc(scores[vn], y), "lambda": lam, "token_r2_by_lambda": {str(k_): v for k_, v in r2s.items()}, "token_r2_r0": r2s[lam]}
        for r in [8, 16, 32, 64, 128]:
            def fn(tr, te, r=r, Zv=Zv, lam=lam):
                U = np.linalg.svd(ridge_W(Zv[tr], Bd[tr], lam), full_matrices=False)[0][:, :r].T; Zr = proj_out(Zv, U); return Zr[te] @ mm(Zr[tr], y[tr])
            def fn_rand(tr, te, r=r, Zv=Zv):
                U = np.linalg.qr(rng.standard_normal((Zv.shape[1], r)))[0].T; Zr = proj_out(Zv, U); return Zr[te] @ mm(Zr[tr], y[tr])
            def fn_rand_span(tr, te, r=r, Zv=Zv):   # random directions INSIDE the training data's span (fair control: same share of data variance)
                U = np.linalg.qr((rng.standard_normal((r, tr.sum())) @ Zv[tr]).T)[0].T; Zr = proj_out(Zv, U); return Zr[te] @ mm(Zr[tr], y[tr])
            t3[vn][f"token_r{r}"] = auc(loao(fn, y, g), y); t3[vn][f"random_r{r}"] = auc(loao(fn_rand, y, g), y); t3[vn][f"random_span_r{r}"] = auc(loao(fn_rand_span, y, g), y)
            U = np.linalg.svd(ridge_W(Zv, Bd, lam), full_matrices=False)[0][:, :r].T; Zr = proj_out(Zv, U)
            t3[vn][f"token_r2_r{r}"] = max(token_r2(Zr, l2) for l2 in LAMS)      # best-case recoverability after removal
    rec["tests"]["token_subspace"] = t3; out[slot] = rec
    print(f"\n== {slot} (k={k}, n={len(y)}, bag features {Bg.shape[1]}) ==")
    print(f"test 1 within identical prefix: {t1['n_groups']} groups, {t1['n_rollouts']} rollouts, {npairs} pairs")
    for name in g_scores: print(f"   {name:45s} AUROC {t1[name]['auroc']:.3f}  perm p {t1[name]['perm_p']:.3f}  (null {t1[name]['null_mean']:.2f} ± {t1[name]['null_sd']:.2f})")
    print(f"test 2 incremental over bag (bag alone {t2['bag_auroc']:.3f}):")
    for vn in variants: r_ = t2[vn]; print(f"   {vn:30s} probe {r_['probe_auroc']:.3f} combined {r_['combined_auroc']:.3f} LR p {r_['lr_p']:.4f} | folds {r_['folds_won']}/{r_['n_folds']} gain {r_['fold_gain_mean']:+.3f} wilcoxon p {r_['wilcoxon_p']:.3f}")
    print("test 3 token-subspace removed (token r / random r):")
    print("   (token dirs removed / random 4096-d dirs removed / random in-span dirs removed; R2 = token identity still recoverable)")
    for vn in t3: print(f"   {vn:30s} lambda {t3[vn]['lambda']:.0e} r0 {t3[vn]['r0']:.3f} (token R2 {t3[vn]['token_r2_r0']:.2f}) | " + " | ".join(f"r{r}: {t3[vn][f'token_r{r}']:.2f}/{t3[vn][f'random_r{r}']:.2f}/{t3[vn][f'random_span_r{r}']:.2f} (R2 {t3[vn][f'token_r2_r{r}']:.2f})" for r in [8, 16, 32, 64, 128]))
json.dump(out, open("../results/results_token_counting_tests.json", "w"), indent=1); print("\nwrote ../results/results_token_counting_tests.json")
