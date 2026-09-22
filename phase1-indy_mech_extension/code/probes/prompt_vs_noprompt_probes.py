#!/usr/bin/env python3
"""What does the full-rollout probe read that the no-prompt probe does not? (added 2026-09-14, README result 1 follow-up)
Both probes are z-scored mass-mean, assistant vs persona, unanimous labels, nulls excluded, leave-one-trigger-out.
Per (layer, slot): cosine between the two probes as raw-space readers (z-space weight / sigma); cross-application
(a probe fit in one condition's z-space scored on the other's); the component of the full-rollout reader orthogonal
to the no-prompt reader, scored alone on both conditions; that component's cosine with the prompt-state (P) mass-mean
direction of README result 2 and its share inside the INLP trigger-identity subspace (inlp_directions.npz, with-prompt
z-space); and the Spearman between the per-trigger AUROC gap and the trigger's assistant rate. Run from data/.
Writes ../results/results_prompt_vs_noprompt_probes.json."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
XP = np.load("features_qwen_wide.npz")["X"]; XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r")
meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
yA = np.array([-1 if r["assistant_A"] is None else r["assistant_A"] for r in lab])
INLP = np.load("../directions/inlp_directions.npz")
TRIG = sorted(a for a in set(arms) if not a.startswith("NULL")); rate = {a: float(yA[(arms == a) & (yA >= 0)].mean()) for a in TRIG}
def unit(v): return v / np.linalg.norm(v)
def mm(G, y): return G[y == 1].mean(0) - G[y == 0].mean(0)
def loao(scorefn, y, g):
    s = np.zeros(len(y))
    for a in sorted(set(g)):
        te = g == a; tr = ~te; s[te] = scorefn(tr, te)
    return s
def auc(s, y): return float(roc_auc_score(y, s))
def perfold(s, y, g): return {a: auc(s[g == a], y[g == a]) for a in sorted(set(g)) if len(set(y[g == a])) > 1}
out = []
for l in [8, 12, 16, 20, 24, 28, 32]:
    li = L.index(l)
    FP_ = XP[:, li, S.index("P")].astype(np.float32); kP = ~np.isnan(FP_[:, 0]) & ~is_null & (yA >= 0)
    sP = FP_[kP].std(0) + 1e-6; rP = unit(mm((FP_[kP] - FP_[kP].mean(0)) / sP, yA[kP]) / sP)
    for slot in ["R1", "R2", "R4", "R8", "R16", "R32", "R64", "Rmean"]:
        si = S.index(slot)
        Fp = XP[:, li, si].astype(np.float32); Fn = np.asarray(XN[:, li, si]).astype(np.float32)
        keep = ~np.isnan(Fp[:, 0]) & ~np.isnan(Fn[:, 0]) & ~is_null & (yA >= 0); Fp, Fn, y, g = Fp[keep], Fn[keep], yA[keep], arms[keep]
        sp, sn = Fp.std(0) + 1e-6, Fn.std(0) + 1e-6; Zp, Zn = (Fp - Fp.mean(0)) / sp, (Fn - Fn.mean(0)) / sn
        rp, rn = unit(mm(Zp, y) / sp), unit(mm(Zn, y) / sn)
        s_pp = loao(lambda tr, te: Zp[te] @ mm(Zp[tr], y[tr]), y, g); s_nn = loao(lambda tr, te: Zn[te] @ mm(Zn[tr], y[tr]), y, g)
        s_pn = loao(lambda tr, te: Zn[te] @ mm(Zp[tr], y[tr]), y, g); s_np = loao(lambda tr, te: Zp[te] @ mm(Zn[tr], y[tr]), y, g)
        def extra(tr):
            a, b = unit(mm(Zp[tr], y[tr]) / sp), unit(mm(Zn[tr], y[tr]) / sn); return unit(a - (a @ b) * b)
        s_ep = loao(lambda tr, te: Fp[te] @ extra(tr), y, g); s_en = loao(lambda tr, te: Fn[te] @ extra(tr), y, g)
        e = unit(rp - (rp @ rn) * rn); key = f"L{l}_{slot}"; trig = None
        if key + "__trigger" in INLP.files:
            B = INLP[key + "__trigger"].astype(np.float32); sc_ = INLP[key + "__scaler_scale"].astype(np.float32)
            ez = unit(e * sc_); trig = dict(frac=float(((B @ ez) ** 2).sum()), floor=len(B) / 4096)
        pf_p, pf_n = perfold(s_pp, y, g), perfold(s_nn, y, g); ks = [a for a in pf_p if a in pf_n]
        gap = [pf_p[a] - pf_n[a] for a in ks]
        rec = dict(layer=l, slot=slot, n=int(len(y)), cos_full_noprompt=float(rp @ rn),
                   auc_full=auc(s_pp, y), auc_noprompt=auc(s_nn, y), auc_full_on_noprompt=auc(s_pn, y), auc_noprompt_on_full=auc(s_np, y),
                   auc_extra_on_full=auc(s_ep, y), auc_extra_on_noprompt=auc(s_en, y),
                   cos_extra_P=float(e @ rP), cos_full_P=float(rp @ rP), cos_noprompt_P=float(rn @ rP), extra_in_trigger_span=trig,
                   spearman_gap_vs_rate=float(spearmanr(gap, [rate[a] for a in ks])[0]), spearman_gap_vs_abs=float(spearmanr(gap, [abs(rate[a] - .5) for a in ks])[0]))
        out.append(rec); trig_s = "n/a" if trig is None else "%.2f (%.2f)" % (trig["frac"], trig["floor"])
        print(f"L{l:>2} {slot:5s} cos {rec['cos_full_noprompt']:+.2f} | full {rec['auc_full']:.2f} np {rec['auc_noprompt']:.2f} full→np {rec['auc_full_on_noprompt']:.2f} np→full {rec['auc_noprompt_on_full']:.2f} "
              f"| extra on full {rec['auc_extra_on_full']:.2f} on np {rec['auc_extra_on_noprompt']:.2f} | cos(extra,P) {rec['cos_extra_P']:+.2f} cos(np,P) {rec['cos_noprompt_P']:+.2f} "
              f"| extra in trig span {trig_s} | gap~rate {rec['spearman_gap_vs_rate']:+.2f}")
json.dump(out, open("../results/results_prompt_vs_noprompt_probes.json", "w"), indent=1); print("wrote ../results/results_prompt_vs_noprompt_probes.json")
