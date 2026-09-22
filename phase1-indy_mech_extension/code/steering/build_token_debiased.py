#!/usr/bin/env python3
"""Build TOKEN-IDENTITY-DEBIASED steering directions (2026-09-14) and their matched controls.

Question: does the assistant mass-mean direction still steer once every direction from which the first-k
response token ids are linearly recoverable has been projected out? README result 1 (token_counting_tests.py,
test 3) showed the *probe* survives that removal; this file builds the *steering* version so the causal
question can be asked on Colab (code/steering/steer_token_debiased_cells.py).

Per slot s in (R1, R2, R4) with k = (1, 2, 4), at each layer in LAYERS:
  1. z-score the with-prompt state with a scaler fit on the FIT arms only (15 triggers, the 5 evaluation
     arms held out - this time actually held out of the fit, unlike build_candidates.py).
  2. token subspace: dual-ridge-regress the z-scored state onto the bag-of-first-k-ids vector (fit arms only),
     take the top-r left singular directions of that [4096, m] map. Lambda chosen by held-out (LOAO over fit
     arms) bag R^2, as in token_counting_tests.py.
  3. mass-mean (assistant minus persona, z-space) with the token subspace projected out.
  4. map back to a RAW-space displacement: z = (f - mu) / sc, so a step d in z is a step d * sc in f. Unit norm.
     (build_candidates.py mapped the INLP direction with w / sc, the *readout* transform; for a mass-mean the two
      differ only in the massive-activation dims. Both variants are saved so the choice is auditable.)
  Averaged over the three slots, unit-normed, like inlp_debiased_early.
Controls built the same way:
  massmean_early_fit15     raw assistant mass-mean on the fit arms only (the honest "raw steering" baseline)
  randspan_r{r}            same pipeline, but r random directions inside the fit data's span removed instead
                           (the fair control from test 3: token directions live in the data span, so "remove r
                           in-span directions" is what has to be beaten)
  random_1                 Gaussian, as before
Also stores per-direction stats: class gap along the unit direction, cosine to the raw direction, fraction of
the raw direction's squared norm that sat in the token subspace, held-out bag R^2 before/after removal, and
the share of squared norm on the single largest dimension.
Run from data/. Writes ../directions/steer_token_debiased.npz and ../directions/steer_token_debiased_meta.json
"""
import json, numpy as np, scipy.sparse as sp, time
X = np.load("features_qwen_wide.npz")["X"]; meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); ids = json.load(open("rollout_ids.json")); V = 151936
HOLD = json.load(open("../results/results_inlp.json"))["held_out_arms"]
arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
yA = np.array([-1 if r["assistant_A"] is None else r["assistant_A"] for r in lab])
LAYERS = [12, 16, 20, 24, 28]; RS = [64, 128]; LAMS = [1e2, 1e3, 1e4, 1e5, 1e6]
rng = np.random.default_rng(20260914)
def unit(v): v = np.asarray(v, np.float64); return v / np.linalg.norm(v)
def proj_out(G, B): return G if B is None else G - (G @ B.T) @ B
def bag(k, keep):
    rows, cols = [], []
    for i, r in enumerate(ids):
        for t in set(r["resp_ids"][:k]): rows.append(i); cols.append(t)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), V))[keep]
    return B[:, np.asarray(B.sum(0)).ravel() >= 2].toarray().astype(np.float32)
def ridge_W(Zt, Bt, lam): K = Zt @ Zt.T + lam * np.eye(len(Zt)); return Zt.T @ np.linalg.solve(K, Bt - Bt.mean(0))
def token_r2(Zv, Bd, g, lam):
    num = den = 0.0
    for a in sorted(set(g)):
        te = g == a; tr = ~te; W = ridge_W(Zv[tr], Bd[tr], lam); pred = (Zv[te] - Zv[tr].mean(0)) @ W + Bd[tr].mean(0)
        num += ((Bd[te] - pred) ** 2).sum(); den += ((Bd[te] - Bd[tr].mean(0)) ** 2).sum()
    return float(1 - num / den)
out, stats = {}, {}; t0 = time.time()
for l in LAYERS:
    li = L.index(l)
    per_slot = {}          # name -> list of raw unit displacements, one per slot
    slot_stats = {}
    for slot, k in [("R1", 1), ("R2", 2), ("R4", 4)]:
        F = X[:, li, S.index(slot)].astype(np.float32)
        fit = ~np.isnan(F[:, 0]) & ~is_null & (yA >= 0) & ~np.isin(arms, HOLD)
        ev = ~np.isnan(F[:, 0]) & ~is_null & (yA >= 0) & np.isin(arms, HOLD)
        mu, sc = F[fit].mean(0), F[fit].std(0) + 1e-6
        Z = (F[fit] - mu) / sc; y = yA[fit]; g = arms[fit]; Zev = (F[ev] - mu) / sc; yev = yA[ev]
        Bd = bag(k, fit)
        mm_z = Z[y == 1].mean(0) - Z[y == 0].mean(0)
        r2s = {lam: token_r2(Z, Bd, g, lam) for lam in LAMS}; lam = max(r2s, key=r2s.get)
        Wfull = ridge_W(Z, Bd, lam); Ufull = np.linalg.svd(Wfull, full_matrices=False)[0]      # [4096, m]
        d = {"massmean_early_fit15": (mm_z, None)}
        rec = dict(slot=slot, k=k, n_fit=int(fit.sum()), n_eval=int(ev.sum()), n_bag=int(Bd.shape[1]), lam=lam, bag_r2_r0=r2s[lam], dirs={})
        for r in RS:
            U = Ufull[:, :r].T
            d[f"tokdebias_r{r}"] = (proj_out(mm_z[None], U)[0], U)
            Ur = np.linalg.qr((rng.standard_normal((r, len(Z))) @ Z).T)[0].T   # r random directions inside the fit data's span
            d[f"randspan_r{r}"] = (proj_out(mm_z[None], Ur)[0], Ur)
            rec[f"bag_r2_after_token_r{r}"] = max(token_r2(proj_out(Z, U), Bd, g, l2) for l2 in LAMS)
            rec[f"bag_r2_after_randspan_r{r}"] = max(token_r2(proj_out(Z, Ur), Bd, g, l2) for l2 in LAMS)
        for name, (dz, B) in d.items():
            disp = unit(dz * sc); read = unit(dz / sc)                     # displacement vs readout mapping to raw space
            per_slot.setdefault(name, []).append(disp); per_slot.setdefault(name + "__readout", []).append(read)
            def auc(s, yy):
                p, n = s[yy == 1], s[yy == 0]; return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))
            Zp = proj_out(Z, B) if B is not None else Z; Zevp = proj_out(Zev, B) if B is not None else Zev
            rec["dirs"][name] = dict(
                frac_of_raw_in_removed_subspace=float(0.0 if B is None else np.sum((B @ unit(mm_z)) ** 2)),
                cos_to_raw_z=float(unit(dz) @ unit(mm_z)), gap_z=float((Zp[y == 1] @ unit(dz)).mean() - (Zp[y == 0] @ unit(dz)).mean()),
                auroc_heldout_arms_z=auc(Zevp @ unit(dz), yev), auroc_heldout_arms_raw_disp=auc(F[ev] @ disp, yev),
                top_dim_share_disp=float(disp.max() ** 2 if abs(disp.max()) > abs(disp.min()) else disp.min() ** 2))
        slot_stats[slot] = rec
        print(f"L{l} {slot}: fit {fit.sum()} eval {ev.sum()} bag {Bd.shape[1]} lam {lam:.0e} R2 {r2s[lam]:.2f} -> " +
              ", ".join(f"tok r{r} {rec[f'bag_r2_after_token_r{r}']:.2f} / rand {rec[f'bag_r2_after_randspan_r{r}']:.2f}" for r in RS) +
              " | " + " ".join(f"{n}: cos {v['cos_to_raw_z']:+.2f} removed {v['frac_of_raw_in_removed_subspace']:.2f} auc_ev {v['auroc_heldout_arms_z']:.2f}" for n, v in rec["dirs"].items()) + f"  [{time.time()-t0:.0f}s]")
    # early-response average (as in build_candidates: R1,R2,R4,R8 mean) for the raw-space class gap and residual norm
    resp = np.nanmean(np.stack([X[:, li, S.index(s)].astype(np.float32) for s in ("R1", "R2", "R4", "R8")]), 0)
    okr = ~np.isnan(resp[:, 0]); fitA = okr & ~is_null & (yA >= 0) & ~np.isin(arms, HOLD); evA = okr & ~is_null & (yA >= 0) & np.isin(arms, HOLD)
    norm = float(np.mean(np.linalg.norm(resp[okr & ~is_null], axis=1)))
    per_slot["random_1"] = [rng.standard_normal(4096)]
    for name, vs in per_slot.items():
        u = unit(np.mean([unit(v) for v in vs], 0)).astype(np.float32); out[f"L{l}__{name}"] = u
        stats[f"L{l}__{name}"] = dict(resid_norm=norm, gap_fit=float((resp[fitA & (yA == 1)] @ u).mean() - (resp[fitA & (yA == 0)] @ u).mean()),
                                      gap_eval=float((resp[evA & (yA == 1)] @ u).mean() - (resp[evA & (yA == 0)] @ u).mean()),
                                      top_dim_share=float((u ** 2).max()), cos_to_raw=float(u @ unit(out[f"L{l}__massmean_early_fit15"])))
    print(f"L{l} SUMMARY (resid norm {norm:.0f}): " + " | ".join(f"{n.split('__',1)[1]} gap fit/eval {s['gap_fit']:+.1f}/{s['gap_eval']:+.1f} cos {s['cos_to_raw']:+.2f} top {s['top_dim_share']:.2f}" for n, s in stats.items() if n.startswith(f"L{l}__")))
    stats[f"L{l}__slots"] = slot_stats
# cosine to the shipped (all-20-arm) candidates, so the refit can be compared with the original run
old = np.load("../directions/steer_candidates.npz")
for l in LAYERS:
    for n in ("massmean_early", "inlp_debiased_early", "random_1"):
        for m in list(out):
            if m.startswith(f"L{l}__") and "readout" not in m: stats[m][f"cos_to_shipped_{n}"] = float(out[m] @ old[f"L{l}__{n}"])
np.savez("../directions/steer_token_debiased.npz", **out)
json.dump(dict(layers=LAYERS, rs=RS, held_out_arms=HOLD, names=sorted({k.split("__", 1)[1] for k in out}), stats=stats, built="2026-09-14"),
          open("../directions/steer_token_debiased_meta.json", "w"), indent=1)
print("saved", len(out), "directions", f"[{time.time()-t0:.0f}s]")
