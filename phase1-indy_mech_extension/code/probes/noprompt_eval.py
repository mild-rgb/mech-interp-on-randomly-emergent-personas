#!/usr/bin/env python3
"""No-prompt control. Same response ids, fed with nothing in front of them.
For each target / null setting / layer / response slot, leave-one-trigger-out mass-mean AUROC for:
  prompt    : trained and tested on prompt-conditioned features (the main grid, recomputed here)
  noprompt  : trained and tested on no-prompt features
  cross P>N : direction trained on prompt features, applied to the held-out arm's NO-PROMPT features
  cross N>P : direction trained on no-prompt features, applied to held-out PROMPT features
If noprompt ~= prompt at 2-16 tokens, the early-position signal is contextual reading of the text alone."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
XP = np.load("features_qwen_wide.npz")["X"]; XN = np.load("noprompt/features_noprompt.npy", mmap_mode="r")
meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
lab = json.load(open("labels.json")); arms = np.array([r["arm"] for r in lab]); is_null = np.array([r["is_null"] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
T = {"assistant_A": lb("assistant_A"), "broken_A": lb("broken_A")}
SL = ["R1", "R2", "R4", "R8", "R16", "R32", "R64", "Rmean", "Rlast"]
def loao(Ftr_src, Fte_src, y, keep):
    s = np.zeros(len(y)); v = np.zeros(len(y), bool)
    for a in sorted(set(arms)):
        te = keep & (arms == a); tr = keep & (arms != a)
        if te.sum() == 0 or len(set(y[tr])) < 2: continue
        w = Ftr_src[tr][y[tr] == 1].mean(0) - Ftr_src[tr][y[tr] == 0].mean(0)
        s[te] = Fte_src[te] @ w; v[te] = True
    return float(roc_auc_score(y[v], s[v]))
out = {}
for t, y in T.items():
    for nulls in ("withnull", "nonull"):
        key = f"{t}/{nulls}"; out[key] = []
        for l in L:
            li = L.index(l)
            for s in SL:
                si = S.index(s)
                FP = XP[:, li, si].astype(np.float32); FN = np.asarray(XN[:, li, si], dtype=np.float32)
                keep = (y >= 0) & ~np.isnan(FP[:, 0]) & ~np.isnan(FN[:, 0])
                if nulls == "nonull": keep &= ~is_null
                out[key].append(dict(layer=l, slot=s, n=int(keep.sum()), prompt=loao(FP, FP, y, keep), noprompt=loao(FN, FN, y, keep),
                                     cross_P_to_N=loao(FP, FN, y, keep), cross_N_to_P=loao(FN, FP, y, keep)))
        best = {s: max((c for c in out[key] if c["slot"] == s and 2 <= c["layer"] <= 34), key=lambda c: c["prompt"]) for s in SL}
        print(f"\n{key}  (LOAO AUROC, at the layer where the PROMPT probe is best)")
        print(f"{'':14}" + " ".join(f"{s:>6}" for s in SL))
        for k in ("prompt", "noprompt", "cross_P_to_N", "cross_N_to_P"):
            print(f"{k:14}" + " ".join(f"{best[s][k]:6.3f}" for s in SL))
        print(f"{'layer':14}" + " ".join(f"{best[s]['layer']:6d}" for s in SL))
        bestN = {s: max((c for c in out[key] if c["slot"] == s and 2 <= c["layer"] <= 34), key=lambda c: c["noprompt"]) for s in SL}
        print(f"{'noprompt@own':14}" + " ".join(f"{bestN[s]['noprompt']:6.3f}" for s in SL))
json.dump(out, open("results_noprompt.json", "w"), indent=1); print("wrote results_noprompt.json")
