#!/usr/bin/env python3
"""figures/phase6_summary.png: three panels against tokens into the reply, layer 20.
A  the persona probe transfers between sources (within-question AUROC)
B  how well a linear read tells steered from prompted, for personas vs assistants vs a plain system prompt
C  position on phase 1's assistant axis (0 = steered persona mean, 1 = steered assistant mean)"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import P6, ROOT

R = json.load(open(P6 / "analysis.json")); C = json.load(open(P6 / "analysis_controls.json"))
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOK = [1, 4, 8, 16, 32, 64]; KEY = {1: "R1", 4: "R4", 8: "R8", 16: "R16", 32: "R32", 64: "R64"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

def line(a, ys, color, label, style="-", xs=TOK):
    a.plot(xs, ys, style, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=label)

def setup(a, title, ylab):
    a.set_xscale("log", base=2); a.set_xticks(TOK); a.set_xticklabels(TOK); a.set_xlim(0.8, 80)
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.set_xlabel("tokens into the reply (layer 20)")
    a.set_title(title, loc="left", fontsize=11, color=INK); a.set_ylabel(ylab)

T = R["transfer"]; g = lambda k, s: T[f"L20{KEY[s]}"][k]["within_q"]
a = ax[0]
line(a, [g("st_on_s", s) for s in TOK], S1, "steered → steered")
line(a, [g("st_on_qt", s) for s in TOK], S2, "steered → prompted")
line(a, [g("pt_on_s", s) for s in TOK], S3, "prompted → steered")
line(a, [C["openerctl"][f"L20{KEY[s]}"]["both_open_Ah"]["steer_dir"]["within_q"] for s in TOK], S4, "steered → prompted,\nboth open 'Ah'", "--")
a.axhline(0.5, color=INK2, lw=1, ls=":"); setup(a, "A. The persona probe transfers", "persona vs assistant AUROC (within question)"); a.set_ylim(0.45, 1.0)

S = C["srcctl"]; h = lambda k, s: S[f"L20{KEY[s]}"][k]["within_q"]
a = ax[1]
TB = [1, 4, 8, 32, 64]    # the source check was run without 16
line(a, [h("persona", s) for s in TB], S1, "personas", xs=TB)
line(a, [h("assistant", s) for s in TB], S2, "assistants", xs=TB)
line(a, [h("sysprompt", s) for s in TB], S3, "system prompt vs none\n(no steering)", xs=TB)
a.axhline(0.5, color=INK2, lw=1, ls=":"); setup(a, "B. Steered vs prompted: mostly a steering trace", "source AUROC (within question)"); a.set_ylim(0.45, 1.02)

X = R["axis"]["slots"]; p = lambda k, s: X[KEY[s]][k]["pos"]
a = ax[2]
T4 = TOK[1:]   # at 1 token the prompted persona sits far off-scale (-5.6), see the README
line(a, [p("Q_Pj", s) for s in T4], S1, "prompted persona", xs=T4)
line(a, [p("T_A", s) for s in T4], S2, "template assistant", xs=T4)
line(a, [p("N_A", s) for s in T4], S3, "neutral assistant", xs=T4)
for yv, txt in ((0, "steered persona"), (1, "steered assistant")):
    a.axhline(yv, color=INK2, lw=1, ls=":"); a.annotate(txt, (3.4, yv), xytext=(0, 3), textcoords="offset points", fontsize=8, color=INK2)
setup(a, "C. Position on the assistant axis", "0 = steered persona, 1 = steered assistant")
ax[0].legend(frameon=False, fontsize=8, loc="lower right"); ax[1].legend(frameon=False, fontsize=8, loc="lower left"); ax[2].legend(frameon=False, fontsize=8, loc="center left")
out = ROOT / "phase6-system-prompt-personas/figures/phase6_summary.png"; fig.savefig(out, dpi=150, facecolor="white"); print(out)
