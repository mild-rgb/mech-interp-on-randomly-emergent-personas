# === compare: measures 1, 2 and 4 for the 'norm' fit (U, V, alpha, R, dfn), plus measures 1-2 for the 'raw' fit
# (U_raw, V_raw, alpha_raw).  Runs in the Colab kernel after both fit cells.
import importlib, analysis; importlib.reload(analysis); from analysis import *
ax = DIRS["assistant_axis_l20"]; REFS = ["persona_l0", "entropy_l0", "entropy_last_l0"]
order_alpha = alpha.abs().argsort(descending=True); rank_alpha = torch.empty_like(order_alpha); rank_alpha[order_alpha] = torch.arange(len(order_alpha), device=order_alpha.device)
CMP = dict(R=R, ref_cos={f"{a}|{c}": float(DIRS[a] @ DIRS[c]) for i, a in enumerate(REFS) for c in REFS[i + 1:]})
def entry(M, d, name):
    mx, i, c, allc = max_abs_cos(M, d)
    top = allc.abs().argsort(descending=True)[:5].tolist()
    return dict(max_abs_cos=mx, feature=i, signed_cos=c, alpha=float(alpha[i]), alpha_rank=int(rank_alpha[i]),
                top5=[dict(feature=j, cos=float(allc[j]), alpha_rank=int(rank_alpha[j])) for j in top],
                subspace_energy=subspace_energy(M, d), random_baseline=random_baseline(M.shape[0], M.shape[1], d))
CMP["V_vs"] = {k: entry(V, DIRS[k], k) for k in REFS}
CMP["U_vs"] = {"assistant_axis_l20": entry(U, ax, "assistant_axis_l20")}
# the output direction of each reference direction itself: Delta(R d) at layer 20, and which DCT outputs it resembles
Dref = dfn.many(R * torch.stack([DIRS[k] for k in REFS], 1))
CMP["Delta_ref"] = {k: dict(norm=float(Dref[:, j].norm()), cos_assistant_axis=float(F.normalize(Dref[:, j], dim=0) @ ax),
                            max_abs_cos_U=max_abs_cos(U, F.normalize(Dref[:, j], dim=0))[0],
                            subspace_energy_U=subspace_energy(U, F.normalize(Dref[:, j], dim=0))) for j, k in enumerate(REFS)}
def raw_entry(M, d, al):
    mx, i, c, allc = max_abs_cos(M, d); r = int((al.abs() > al.abs()[i]).sum())
    return dict(max_abs_cos=mx, feature=i, signed_cos=c, alpha_rank=r, subspace_energy=subspace_energy(M, d))
CMP["raw_fit"] = dict(R=R_raw, V_vs={k: raw_entry(V_raw, DIRS[k], alpha_raw) for k in REFS},
                      U_vs={"assistant_axis_l20": raw_entry(U_raw, ax, alpha_raw)},
                      Delta_norm_quantiles=torch.quantile(DlRV_raw.norm(dim=0), torch.tensor([0, .5, .9, .99, 1.], device=ax.device)).tolist(),
                      n_features_Delta_over_100=int((DlRV_raw.norm(dim=0) > 100).sum()))
print("raw fit", CMP["raw_fit"])
CMP["Delta_RV_norm"] = dict(mean=float(DlRV.norm(dim=0).mean()), max=float(DlRV.norm(dim=0).max()))
for k in REFS: print(k, {x: CMP["V_vs"][k][x] for x in ["max_abs_cos", "feature", "signed_cos", "alpha_rank", "subspace_energy"]}, "| random", CMP["V_vs"][k]["random_baseline"])
print("U vs assistant", {x: CMP["U_vs"]["assistant_axis_l20"][x] for x in ["max_abs_cos", "feature", "signed_cos", "alpha_rank", "subspace_energy"]}, CMP["U_vs"]["assistant_axis_l20"]["random_baseline"])
print("Delta of refs", CMP["Delta_ref"])

# measure 4: layer-20 last-position projection on the assistant axis, gap units, all 40 prompts
bA = rig.batch(TRAIN + HELD); XA, YA = rig.source_target(bA)
base = (YA[:, -1].float() @ ax)
def drop(p): return ((p - base) / GAP_INLP)                                  # [.., 40]
P_feat = {k: drop(assistant_proj_fixed(rig, bA, XA, s * R * V, ax)) for k, s in [("1R", 1.0), ("2R", 2.0)]}
g = torch.Generator().manual_seed(7); Vr = F.normalize(torch.randn(rig.D, 64, generator=g), dim=0).to(rig.dev)
P_rand = {k: drop(assistant_proj_fixed(rig, bA, XA, s * R * Vr, ax)) for k, s in [("1R", 1.0), ("2R", 2.0)]}
P_ref = {}
for k in REFS:
    for s in [1.0, 2.0]: P_ref[f"{k}|fixed {s}R"] = drop(assistant_proj_fixed(rig, bA, XA, s * R * DIRS[k][:, None], ax))[0]
    P_ref[f"{k}|rel eps0.4 all tokens"] = drop(assistant_proj_rel(rig, bA, XA, DIRS[k], 0.4, ax, all_tokens=True))
    P_ref[f"{k}|rel eps0.4 no first"] = drop(assistant_proj_rel(rig, bA, XA, DIRS[k], 0.4, ax, all_tokens=False))
nTR = len(TRAIN)
def summ(P):  # P [m, 40]
    m = P.mean(1); return dict(mean_all=m.tolist(), held_mean=P[:, nTR:].mean(1).tolist())
M4 = dict(gap_units=GAP_INLP, clean_proj_mean=float(base.mean()),
          features={k: summ(v) for k, v in P_feat.items()}, random={k: summ(v) for k, v in P_rand.items()},
          refs={k: dict(mean_all=float(v.mean()), held_mean=float(v[nTR:].mean())) for k, v in P_ref.items()})
for k in ["1R", "2R"]:
    fm, rm = P_feat[k].mean(1), P_rand[k].mean(1)
    thr = float(rm.min())
    M4[f"count_below_random_min_{k}"] = int((fm < thr).sum()); M4[f"count_below_-1gap_{k}"] = int((fm < -1).sum())
    M4[f"count_below_-3gap_{k}"] = int((fm < -3).sum())
    o = fm.argsort()[:10].tolist()
    M4[f"most_negative_{k}"] = [dict(feature=j, drop=float(fm[j]), alpha_rank=int(rank_alpha[j]), cos_persona=float(V[:, j] @ DIRS["persona_l0"])) for j in o]
    print(k, "random range", float(rm.min()), float(rm.max()), "| features below random min", M4[f"count_below_random_min_{k}"], "| below -1 gap", M4[f"count_below_-1gap_{k}"], "| below -3", M4[f"count_below_-3gap_{k}"])
    print("   most negative", [(d["feature"], round(d["drop"], 2), d["alpha_rank"]) for d in M4[f"most_negative_{k}"]])
for k, v in M4["refs"].items(): print("  ref", k, round(v["mean_all"], 2), round(v["held_mean"], 2))
CMP["measure4"] = M4
json.dump(CMP, open("/content/results/compare.json", "w"), indent=1)
