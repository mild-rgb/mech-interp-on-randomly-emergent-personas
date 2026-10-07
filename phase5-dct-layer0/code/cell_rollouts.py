# === measure 3: steer chosen DCT features (prefill only) on 8 held-out benign prompts, 2 seeds, 96 tokens, T=1
# runs after cell_compare.py (uses CMP, rank_alpha, P_feat).  Writes /content/results/rollouts.json
steer = Steer(rig) if "steer" not in globals() else steer
RP = HELD[:8]
pick = {}
for k in REFS:
    for t in CMP["V_vs"][k]["top5"][:2]: pick.setdefault(t["feature"], []).append(f"top |cos| {k} ({t['cos']:+.3f})")
for j in order_alpha[:6].tolist(): pick.setdefault(j, []).append(f"alpha rank {int(rank_alpha[j])}")
for d in CMP["measure4"]["most_negative_1R"][:3]: pick.setdefault(d["feature"], []).append(f"assistant-axis drop {d['drop']:+.2f} gap @1R")
for d in CMP["measure4"]["most_negative_2R"][:3]: pick.setdefault(d["feature"], []).append(f"assistant-axis drop {d['drop']:+.2f} gap @2R")
print("features:", pick)
ROLL = dict(prompts=RP, seeds=2, new_tokens=96, R=R, arms={})
def arm(name, spec, why):
    t = time.time(); ROLL["arms"][name] = dict(why=why, rows=rollouts(rig, steer, RP, spec, seeds=2, new=96))
    print(f"{name:34s} {time.time()-t:4.0f}s | {ROLL['arms'][name]['rows'][0]['text'][:110]!r}", flush=True)
arm("clean", None, "no steering")
arm("persona_l0 rel eps0.4", ("rel", DIRS["persona_l0"], 0.4), "phase 3 direction, phase 3 rig (all prompt tokens, eps*||h||)")
arm("persona_l0 rel eps0.3", ("rel", DIRS["persona_l0"], 0.3), "phase 3 direction below its fit strength")
arm("persona_l0 rel eps0.5", ("rel", DIRS["persona_l0"], 0.5), "phase 3 direction above its fit strength")
arm("persona_l0 fixed 1R", ("fixed", R * DIRS["persona_l0"], None), "phase 3 direction, DCT convention (fixed norm R, first token skipped)")
arm("entropy_l0 rel eps0.8", ("rel", DIRS["entropy_l0"], 0.8), "phase 2 direction, phase 2 rig")
for i in range(2): arm(f"random{i} 2R", ("fixed", 2 * R * Vr[:, i], None), "random unit direction at 2R")
for j, why in pick.items():
    for s in [1.0, 2.0]: arm(f"f{j} {s:g}R", ("fixed", s * R * V[:, j], None), "; ".join(why))
json.dump(ROLL, open("/content/results/rollouts.json", "w"), indent=1)
