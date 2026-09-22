#!/usr/bin/env python3
"""Bag-of-tokens steering vectors (2026-09-14). Question: if you build a steering vector purely from the tokens a
bag-of-tokens classifier likes, does it steer the same way as the assistant direction?

1. Bag classifier: binary presence of token ids in the first 4 response tokens (tokens seen in >= 2 fit rollouts),
   L2 logistic C=0.3, liblinear, class-balanced; assistant (1) vs persona (0), unanimous tier-A labels, nulls excluded,
   FIT on the 15 fit arms (the 5 evaluation arms held out, as build_token_debiased.py). Held-out AUROC on the 5 arms is
   printed as a sanity check. The "liked" tokens are the top-20 positive and top-20 negative coefficients.
2. Native vector at feature layer L (in 12..28 steps): for each liked token t, the mean RAW residual state at the response
   positions where t was just written (stored slots R1, R2, R4 = positions of response tokens 1, 2, 4), each slot centred
   on its own mean over all fit-arm non-null rollouts; v_L = sum_t w_t * m_t, unit-normed. Tokens never seen at those
   slots are skipped (coverage recorded).
3. The literal vectors (sum_t w_t * E[t] and sum_t w_t * W_U[t]) need the model weights and are built on Colab by
   job_bag_vllm.py from the tokens and weights saved here.
Also saved: residual norms (early-response mean, non-null) at every stored layer, for eps sizing, and each native vector's
cosine to the raw and token-debiased directions from steer_token_debiased.npz.
Run from data/. Writes ../directions/steer_bag_vectors.npz and ../directions/steer_bag_vectors_meta.json
"""
import json, numpy as np, scipy.sparse as sp, collections
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = np.load("features_qwen_wide.npz")["X"]; meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json")); V = 151936
HOLD = json.load(open("../results/results_inlp.json"))["held_out_arms"]
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
yA = np.array([-1 if r["assistant_A"] is None else r["assistant_A"] for r in lab])
fitarm = ~is_null & ~np.isin(arms, HOLD); fit = fitarm & (yA >= 0); ev = ~is_null & np.isin(arms, HOLD) & (yA >= 0)
K = 4; NTOP = 20
rows, cols = [], []
for i, r in enumerate(ids):
    for t in set(r["resp_ids"][:K]): rows.append(i); cols.append(t)
B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))
vocab = np.where(np.asarray(B[fit].sum(0)).ravel() >= 2)[0]; Bv = B[:, vocab]
clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", solver="liblinear").fit(Bv[fit], yA[fit])
auc_ev = roc_auc_score(yA[ev], clf.decision_function(Bv[ev])); w = clf.coef_[0]
order = np.argsort(w); neg, pos = order[:NTOP], order[::-1][:NTOP]
liked = [(int(vocab[j]), float(w[j])) for j in list(pos) + list(neg)]
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); dec = lambda t: tok.decode([t])
except Exception:
    dec = lambda t: f"<{t}>"
docfreq = {int(vocab[j]): int(B[fit][:, vocab[j]].sum()) for j in list(pos) + list(neg)}
print(f"bag classifier: {Bv.shape[1]} tokens, held-out-arm AUROC {auc_ev:.3f} (fit n={fit.sum()}, eval n={ev.sum()})")
print("assistant-liked:", " ".join(f"{dec(t)!r}{wt:+.2f}(n{docfreq[t]})" for t, wt in liked[:NTOP]))
print("persona-liked:  ", " ".join(f"{dec(t)!r}{wt:+.2f}(n{docfreq[t]})" for t, wt in liked[NTOP:]))
SLOTK = [("R1", 1), ("R2", 2), ("R4", 4)]
tokdeb = np.load("../directions/steer_token_debiased.npz")
out, stats = {}, dict(held_out_arms=HOLD, bag_k=K, n_top=NTOP, bag_vocab=int(Bv.shape[1]), bag_heldout_auroc=float(auc_ev),
                      liked=[dict(id=t, weight=wt, text=dec(t), fit_docfreq=docfreq[t]) for t, wt in liked], resid_norm={}, native={})
for l in L:
    li = L.index(l)
    resp = np.nanmean(np.stack([X[:, li, S.index(s)].astype(np.float32) for s in ("R1", "R2", "R4", "R8")]), 0)
    stats["resid_norm"][str(l)] = float(np.mean(np.linalg.norm(resp[~np.isnan(resp[:, 0]) & ~is_null], axis=1)))
for l in [12, 16, 20, 24, 28]:
    li = L.index(l); acc = collections.defaultdict(list)
    for s, k in SLOTK:
        F = X[:, li, S.index(s)].astype(np.float32); ok = fitarm & ~np.isnan(F[:, 0]); mu = F[ok].mean(0)
        for i in np.where(ok)[0]:
            r = ids[i]["resp_ids"]
            if len(r) >= k: acc[r[k - 1]].append(F[i] - mu)
    v = np.zeros(4096, np.float64); used = []
    for t, wt in liked:
        if acc.get(t): m = np.mean(acc[t], 0); v += wt * m; used.append(dict(id=t, n=len(acc[t])))
    u = (v / np.linalg.norm(v)).astype(np.float32); out[f"L{l}__bagnative"] = u
    raw = tokdeb[f"L{l}__massmean_early_fit15"]; deb = tokdeb[f"L{l}__tokdebias_r128"]
    stats["native"][str(l)] = dict(tokens_used=len(used), tokens_used_detail=used, cos_to_raw=float(u @ raw / np.linalg.norm(raw)),
                                   cos_to_tokdebias_r128=float(u @ deb / np.linalg.norm(deb)), top_dim_share=float((u ** 2).max()))
    print(f"L{l}: native bag vector from {len(used)}/{len(liked)} tokens | cos to raw {stats['native'][str(l)]['cos_to_raw']:+.3f} | cos to token-debiased {stats['native'][str(l)]['cos_to_tokdebias_r128']:+.3f} | top-dim share {stats['native'][str(l)]['top_dim_share']:.2f} | resid norm {stats['resid_norm'][str(l)]:.0f}")
np.savez("../directions/steer_bag_vectors.npz", **out)
json.dump(stats, open("../directions/steer_bag_vectors_meta.json", "w"), indent=1, ensure_ascii=False); print("saved")
