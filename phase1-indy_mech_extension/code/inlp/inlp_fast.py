#!/usr/bin/env python3
"""INLP over positions R1..R4 at every stored layer, for four concepts, plus debiasing of the assistant probe.
See the message/README for the design. Output: results_inlp.json, inlp_directions.npz."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import json, argparse, pathlib, time, collections
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

D = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--slots", default="R1,R2,R4"); ap.add_argument("--layers", default="")
ap.add_argument("--max_iter", type=int, default=25); ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--out", default=str(D/"results_inlp.json"))
ap.add_argument("--dirs_out", default=str(D/"inlp_directions.npz"))
args = ap.parse_args()

X = np.load(D/"features_qwen_wide.npz")["X"]; meta = json.load(open(D/"features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open(D/"labels.json")); N = len(lab)
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab]); seeds = np.array([r["seed"] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
y_asst, y_brk = lb("assistant_A"), lb("broken_A")
LANGS = ["english", "chinese", "japanese", "korean", "thai"]
def lang_of(r):
    c = collections.Counter(v["language"] for v in r["verdicts"]); l, n = c.most_common(1)[0]
    return LANGS.index(l) if (l in LANGS and n >= 2) else len(LANGS)     # 5 = other/mixed
y_lang = np.array([lang_of(r) for r in lab])
TRIG_ARMS = sorted(a for a in set(arms) if not a.startswith("NULL")); y_trig = np.array([TRIG_ARMS.index(a) if a in TRIG_ARMS else -1 for a in arms])
# fixed held-out triggers: 5 arms spanning the assistant-rate range
rate = {a: y_asst[(arms == a) & (y_asst >= 0)].mean() for a in TRIG_ARMS}
HOLD = [a for a, _ in sorted(rate.items(), key=lambda kv: kv[1])][::4][:5]
print("held-out arms:", HOLD, "| lang counts:", np.bincount(y_lang).tolist())

SLOTS = args.slots.split(","); LAYER_SET = [int(x) for x in args.layers.split(",")] if args.layers else L

def gram_schmidt_add(B, w):
    w = w.astype(np.float64).copy()
    if B is not None and len(B): w -= B.T @ (B @ w)
    n = np.linalg.norm(w)
    if n < 1e-8: return B
    w /= n
    return w[None] if B is None or not len(B) else np.vstack([B, w])

def inlp(Z, y, tr, te, max_iter, C, chance, multiclass=False):
    if multiclass: max_iter = min(max_iter, 8)
    """Returns (basis [k, d], per-iteration log). Directions are fitted on tr, stopping on te."""
    B = None; log = []; below = 0
    for it in range(max_iter):
        Zp = Z if B is None else Z - (Z @ B.T) @ B
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Zp[tr], y[tr])
        bacc = balanced_accuracy_score(y[te], m.predict(Zp[te])); tr_acc = balanced_accuracy_score(y[tr], m.predict(Zp[tr]))
        rec = dict(it=it, test_bacc=float(bacc), train_bacc=float(tr_acc))
        if not multiclass: rec["test_auroc"] = float(roc_auc_score(y[te], m.decision_function(Zp[te])))
        log.append(rec)
        for w in np.atleast_2d(m.coef_): B = gram_schmidt_add(B, w)
        recent = [r["test_bacc"] for r in log[-3:]]
        if len(recent) == 3 and np.mean(recent) <= chance + 0.05: break
    return B, log

def project_out(Z, B): return Z if B is None or not len(B) else Z - (Z @ B.T) @ B

def loao_auroc(F, y, keep, demean=False, B=None, probe="massmean", C=0.1):
    s = np.zeros(len(y)); v = np.zeros(len(y), bool); G = F.copy()
    if demean:
        for a in set(arms): m = arms == a; G[m] -= G[m].mean(0)       # per-arm demeaning (uses no labels)
    G = project_out(G, B)
    for a in sorted(set(arms[keep])):
        te = keep & (arms == a); tr = keep & (arms != a)
        if te.sum() == 0 or len(set(y[tr])) < 2: continue
        if probe == "massmean":
            w = G[tr][y[tr] == 1].mean(0) - G[tr][y[tr] == 0].mean(0); s[te] = G[te] @ w
        else:
            m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(G[tr], y[tr]); s[te] = m.decision_function(G[te])
        v[te] = True
    return float(roc_auc_score(y[v], s[v]))

def fit_dir(F, y, keep, demean=False, B=None):
    G = F.copy()
    if demean:
        for a in set(arms): m = arms == a; G[m] -= G[m].mean(0)
    G = project_out(G, B); w = G[keep & (y == 1)].mean(0) - G[keep & (y == 0)].mean(0); return w / np.linalg.norm(w)

def cell(layer, slot):
    li, si = L.index(layer), S.index(slot)
    F = X[:, li, si].astype(np.float32); ok = ~np.isnan(F[:, 0]) & ~is_null
    Z = np.zeros_like(F); sc = StandardScaler().fit(F[ok]); Z[ok] = sc.transform(F[ok])
    # EXACT reduction: all probes here (L2 logistic, mass-mean, INLP) have solutions inside span(rows of Z),
    # so work in an orthonormal basis of that span (rank <= n) and rotate directions back at the end.
    U_, s_, Vt = np.linalg.svd(Z[ok], full_matrices=False); rank = int((s_ > s_[0] * 1e-6).sum()); V = Vt[:rank]   # [rank, 4096]
    Z = Z @ V.T                                                                                                     # [N, rank]
    hold = np.isin(arms, HOLD)
    out = dict(layer=layer, slot=slot, held_out_arms=HOLD, inlp={}, cos={}, debias={})
    bases = {}
    # --- INLP per concept -----------------------------------------------------------------
    for name, y, chance, multi, split in [
        ("assistant", y_asst, 0.5, False, "arms"), ("broken", y_brk, 0.5, False, "arms"),
        ("language", y_lang, 1/6, True, "arms"), ("trigger", y_trig, 1/20, True, "seeds")]:
        keep = ok & (y >= 0)
        if split == "arms": tr, te = keep & ~hold, keep & hold
        else: tr, te = keep & (seeds < 118), keep & (seeds >= 118)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
        B, log = inlp(Z, y, tr, te, args.max_iter, args.C, chance, multi)
        bases[name] = B; out["inlp"][name] = dict(n_dirs=int(0 if B is None else len(B)), log=log)
    # --- cosine structure ---------------------------------------------------------------
    def span_frac(v, B):  # fraction of ||v||^2 inside span(B)
        if B is None or not len(B): return 0.0
        v = v / np.linalg.norm(v); return float(np.sum((B @ v) ** 2))
    if "assistant" in bases and bases["assistant"] is not None:
        A = bases["assistant"]
        for other in ("broken", "language", "trigger"):
            if other in bases and bases[other] is not None:
                O = bases[other]; M = np.abs(A @ O.T)
                out["cos"][f"assistant_vs_{other}"] = dict(max_abs_cos=float(M.max()), first_vs_first=float(M[0, 0]),
                    frac_of_assistant1_in_other_span=span_frac(A[0], O), frac_of_assistant_span_in_other=float(np.mean([span_frac(a, O) for a in A])),
                    frac_of_other1_in_assistant_span=span_frac(O[0], A))
        if "broken" in bases and "trigger" in bases and bases["broken"] is not None and bases["trigger"] is not None:
            out["cos"]["broken_vs_trigger"] = dict(max_abs_cos=float(np.abs(bases["broken"] @ bases["trigger"].T).max()),
                                                    frac_of_broken1_in_trigger_span=span_frac(bases["broken"][0], bases["trigger"]))
    # --- debiasing the assistant probe --------------------------------------------------
    keep = ok & (y_asst >= 0)
    nuis = [bases[k] for k in ("trigger", "broken", "language") if k in bases and bases[k] is not None]
    Bn = None
    for Bk in nuis:
        for w in Bk: Bn = gram_schmidt_add(Bn, w)
    Blb = None
    for Bk in [bases[k] for k in ("broken", "language") if k in bases and bases[k] is not None]:
        for w in Bk: Blb = gram_schmidt_add(Blb, w)
    out["debias"]["n_nuisance_dirs"] = int(0 if Bn is None else len(Bn))
    out["cos"]["expected_random_frac"] = {k: (0 if bases.get(k) is None else len(bases[k])) / Z.shape[1] for k in ("broken", "language", "trigger")}   # relative to the rank-dim span
    variants = {"raw": (False, None), "arm_demeaned": (True, None), "nuisance_nullspace": (False, Bn), "both": (True, Bn),
                "trigger_nullspace": (False, bases.get("trigger")), "language_nullspace": (False, bases.get("language")), "broken_nullspace": (False, bases.get("broken")),
                "demean_lang_broken": (True, Blb)}
    d_raw = fit_dir(Z, y_asst, keep)
    for vn, (dm, B) in variants.items():
        d = fit_dir(Z, y_asst, keep, dm, B)
        out["debias"][vn] = dict(loao_massmean=loao_auroc(Z, y_asst, keep, dm, B, "massmean"),
                                 loao_logistic=loao_auroc(Z, y_asst, keep, dm, B, "logistic", args.C),
                                 cos_to_raw=float(d @ d_raw),
                                 frac_in_trigger_span=span_frac(d, bases.get("trigger")), frac_in_broken_span=span_frac(d, bases.get("broken")),
                                 frac_in_language_span=span_frac(d, bases.get("language")))
        bases[f"assistant_dir_{vn}"] = d[None]
    # also: does the pure direction still separate assistant/persona on the WITH-null data, and on no-prompt features? (left for later)
    out["rank"] = rank
    return out, {f"L{layer}_{slot}__{k}": ((v @ V).astype(np.float16) if v is not None else np.zeros((0, V.shape[1]), np.float16)) for k, v in bases.items()} | \
                {f"L{layer}_{slot}__scaler_mean": sc.mean_.astype(np.float16), f"L{layer}_{slot}__scaler_scale": sc.scale_.astype(np.float16)}

cells = [(l, s) for l in LAYER_SET for s in SLOTS]
t0 = time.time()
res = Parallel(n_jobs=args.jobs)(delayed(cell)(l, s) for l, s in cells)
json.dump(dict(layers=LAYER_SET, slots=SLOTS, held_out_arms=HOLD, langs=LANGS + ["other"], C=args.C, cells=[r for r, _ in res]), open(args.out, "w"), indent=1)
allds = {}
for _, d in res: allds.update(d)
np.savez_compressed(args.dirs_out, **allds)
for r, _ in res:
    n = {k: v["n_dirs"] for k, v in r["inlp"].items()}; db = r["debias"]
    print(f"L{r['layer']:>2}/{r['slot']:<3} dirs {n} | asst AUROC raw {db['raw']['loao_logistic']:.3f} demean {db['arm_demeaned']['loao_logistic']:.3f} "
          f"nullsp {db['nuisance_nullspace']['loao_logistic']:.3f} both {db['both']['loao_logistic']:.3f} | cos(both,raw) {db['both']['cos_to_raw']:+.2f} "
          f"| asst1 in trig-span {r['cos'].get('assistant_vs_trigger',{}).get('frac_of_assistant1_in_other_span',float('nan')):.2f}")
print(f"wrote {args.out}, {args.dirs_out}  [{time.time()-t0:.0f}s]")
