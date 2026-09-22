#!/usr/bin/env python3
"""Regenerate figures/fig_prefix_curve.png after the 2026-09-12 z-scoring correction (README result 1).
Hidden-state rows are mass-mean on z-scored residuals at each slot's own best layer 2-34
(results/results_noprompt_zscored.json); the raw no-prompt row as originally run is kept as a dashed line;
bag-of-tokens is unchanged (results/readme_numbers.json). Run from the repo root."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = json.load(open("results/results_noprompt_zscored.json")); R = json.load(open("results/readme_numbers.json"))
SL = ["R1", "R2", "R4", "R8", "R16", "R32", "R64"]; ks = [1, 2, 4, 8, 16, 32, 64]
def best(target, cond, key): return [max(r[cond][key] for r in Z[target] if r["slot"] == s) for s in SL]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=125)
for ax, target, title in zip(axes, ["assistant_A", "broken_A"], ["assistant vs persona (unanimous labels, no nulls)", "broken vs fluent (unanimous labels, no nulls)"]):
    rows = [("hidden state, with prompt, z-scored (best layer)", best(target, "prompt", "z_pooled"), "#1f77b4", "-"),
            ("hidden state, no prompt, z-scored (best layer)", best(target, "noprompt", "z_pooled"), "#d2691e", "-"),
            ("hidden state, no prompt, raw residuals — as run 2026-09-09, superseded", R[target]["noprompt"], "#d2691e", "--"),
            ("bag of tokens", R[target]["lex"], "#8a2be2", "-")]
    for lab, v, c, ls in rows:
        ax.plot(range(7), v, marker="o" if ls == "-" else None, color=c, ls=ls, lw=2.2 if ls == "-" else 1.4, alpha=1 if ls == "-" else 0.7, label=lab)
        if ls == "-": ax.annotate(f"{v[-1]:.2f}", (6, v[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color="gray", va="center")
    ax.set_xticks(range(7)); ax.set_xticklabels(ks); ax.set_xlabel("response tokens seen"); ax.set_ylabel("leave-one-trigger-out AUROC")
    ax.set_ylim(0.55, 1.0); ax.grid(alpha=0.3); ax.set_title(title, fontsize=11)
axes[0].legend(fontsize=8, loc="lower right", frameon=False)
fig.suptitle("mass-mean on z-scored residuals (corrected 2026-09-12; dashed = the raw-residual no-prompt row the original figure showed)", fontsize=9, color="dimgray")
fig.tight_layout(); fig.savefig("figures/fig_prefix_curve.png"); print("wrote figures/fig_prefix_curve.png")
