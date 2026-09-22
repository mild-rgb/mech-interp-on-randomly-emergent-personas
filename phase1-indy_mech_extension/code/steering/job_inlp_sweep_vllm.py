#!/usr/bin/env python3
# =============================================================================
# indy_mech_extension — INLP-debiased assistant directions steered at the layers where they PREDICT best (2026-09-16).
# Same rig, gate, hook and hook-check as job_run1fill_vllm.py / job_tokdebias_vllm.py.
#
# Directions: directions/steer_inlp_sweep.npz (per-slot demean_lang_broken INLP directions from inlp_directions.npz, mapped
# to raw coordinates exactly as build_candidates.py does, plus their unit-average "pooled" = the shipped inlp_debiased_early).
# Layers chosen from results_inlp.json leave-one-arm-out mass-mean AUROC of the debiased direction:
#   R1 best L10 (0.738) ~ L20 (0.737); R2 best L36 (0.698, DROPPED: >60 % of the vector is the massive-activation dim and all
#   three slots are cos 1.00 there) then L10 (0.694), L16 (0.691); R4 best L22 (0.757), L20 (0.745); pooled best L22 = L20 (0.722).
# Every arm steers at its gated fit block (feature layer L -> vllm index L-1), eps 0.35, response-only.
#   D0 baseline
#   D1 inlp_R1 @ L10      D2 inlp_R1 @ L20
#   D3 inlp_R2 @ L10      D4 inlp_R2 @ L16
#   D5 inlp_R4 @ L22      D6 inlp_R4 @ L20
#   D7 inlp_pooled @ L22  D8 inlp_pooled @ L20  D9 inlp_pooled @ L16 (the shipped direction at its FIT block; C3 hooked index 16)
#   E1-E9: the same nine arms with sign -1 (reversed), added 2026-09-16 on request
#   F1-F7 / G1-G7: the seven other INLP debias variants at L20, forward / reversed (steer_inlp_variants.npz), added the same day
#   N1-N4 / M1-M4: no-prompt-state directions at L20, forward / reversed (steer_noprompt.npz), 2026-09-17
# Env: OUT_DIR, N_SEEDS, ARMS, VERIFY_ONLY, MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, math, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch

IN = os.environ.get("IN_DIR", "/content"); OUT_DIR = os.environ.get("OUT_DIR", "/content/inlpsweep"); os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = int(os.environ.get("N_SEEDS", "48")); VERIFY_ONLY = os.environ.get("VERIFY_ONLY", "0") == "1"
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); N_NEW = 96; EPS = 0.35; OLD_IDX = 20; FEAT_LAYER = 20; STEER_LAYERS = [10, 16, 20, 22]
MODEL = "Qwen/Qwen3-8B"; Q = "what shall i do today"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

NEW = np.load(f"{IN}/steer_token_debiased.npz"); NMETA = json.load(open(f"{IN}/steer_token_debiased_meta.json"))
OLD = np.load(f"{IN}/steer_candidates.npz"); OMETA = json.load(open(f"{IN}/steer_candidates_meta.json"))
SW = np.load(f"{IN}/steer_inlp_sweep.npz"); SMETA = json.load(open(f"{IN}/steer_inlp_sweep_meta.json"))
VA = np.load(f"{IN}/steer_inlp_variants.npz") if os.path.exists(f"{IN}/steer_inlp_variants.npz") else None
VMETA = json.load(open(f"{IN}/steer_inlp_variants_meta.json")) if VA is not None else None
NP = np.load(f"{IN}/steer_noprompt.npz") if os.path.exists(f"{IN}/steer_noprompt.npz") else None
NPMETA = json.load(open(f"{IN}/steer_noprompt_meta.json")) if NP is not None else None
SURV = json.load(open(f"{IN}/phase17_qwen_wide_surveys.json")); HOLD = NMETA["held_out_arms"]
SKIP_GATE = os.environ.get("SKIP_GATE", "0") == "1"   # 2026-09-17: gate + hook check passed identically on four launches; allow skipping when the reference files are absent
REF = None if SKIP_GATE else np.load(f"{IN}/layercheck_ref.npz"); REF_IDS = [] if SKIP_GATE else json.load(open(f"{IN}/layercheck_ids.json"))

# ------------------------------------------------------------ prompts (phase 17 rig) ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL); EOS = tok.eos_token_id
s = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(s, add_special_tokens=False, return_offsets_mapping=True); i0 = s.index(Q)
j0 = next(t for t, (a, b) in enumerate(enc["offset_mapping"]) if b > i0); PRE_P, SUF_P = enc["input_ids"][:j0], enc["input_ids"][j0:]
CLEAN = list(PRE_P) + list(SUF_P)
SYS_CEIL = "You are a helpful assistant. Always reply in English, in your own neutral assistant voice."
s2 = tok.apply_chat_template([{"role": "system", "content": SYS_CEIL}, {"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc2 = tok(s2, add_special_tokens=False, return_offsets_mapping=True); i2 = s2.index(Q)
k0 = next(t for t, (a, b) in enumerate(enc2["offset_mapping"]) if b > i2); PRE_S, SUF_S = enc2["input_ids"][:k0], enc2["input_ids"][k0:]
_sys_ids = list(PRE_S) + list(SUF_S)
assert list(SUF_S) == list(SUF_P), "system scaffold's post-query tail differs from the trigger rig's"
assert 151667 in _sys_ids and 151668 in _sys_ids and _sys_ids.index(151668) - _sys_ids.index(151667) <= 2, "thinking not disabled: non-empty <think> block"
assert SYS_CEIL in tok.decode(_sys_ids)
def sys_ids(trig): return list(PRE_S) + list(trig) + list(SUF_S)
def pre_ids(trig): return list(PRE_P) + list(trig) + list(SUF_P)
TRIGS = {a: SURV["arms"][a]["ids"] for a in HOLD}
for r in REF_IDS:                                   # the rebuilt prompt must equal the prompt the stored states were extracted under
    want = CLEAN if r["arm"] == "NULL-clean" else pre_ids(SURV["arms"][r["arm"]]["ids"])
    assert list(r["prompt_ids"]) == want, f"prompt rebuild mismatch for {r['arm']} seed {r['seed']}"
log(f"prompt rebuild matches stored prompt_ids for {len(REF_IDS)} rollouts | held-out {HOLD} | clean len {len(CLEAN)}")

# ------------------------------------------------------------------- the hook ----
class PosCondAdd:
    """Add ``vec`` to the residual at absolute positions >= start (vllm-lens per-request Hook; picklable)."""
    def __init__(self, start, vec):
        self.start = int(start); self.vec = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
    def __call__(self, ctx, h):
        seen = ctx.saved.get("n", 0); n = h.shape[0]; ctx.saved["n"] = seen + n
        first = max(self.start - seen, 0)
        if first >= n: return None
        out = h.clone(); out[first:] = out[first:] + self.vec.to(device=h.device, dtype=h.dtype)
        return out

from vllm import LLM, SamplingParams
import vllm_lens  # noqa: F401  (registers the plugin)
from vllm_lens import Hook
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=512, enforce_eager=True,
          max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=False, max_logprobs=200)
log("engine up")
def acts(o): return torch.as_tensor(o.activations["residual_stream"]).float().numpy()

# ------------------------------------------------------------- layer gate ----
if SKIP_GATE:
    FIT = {fl: fl - 1 for fl in STEER_LAYERS}; FIT_IDX = FIT[FEAT_LAYER]; log(f"SKIP_GATE: using feature layer -> vllm index {FIT} (the mapping every gated launch found)")
else:
  GL = list(range(8, 31))
  outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in REF_IDS],
                      SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": GL}), use_tqdm=False)
  G = [acts(o) for o in outs]
  gate = dict(vllm_indices=GL, cos={}, fit_idx={}, pass_=True)
  for fl in range(10, 32, 2):
      ref = REF[f"F{fl}_P"]; row = []
      for gi, L in enumerate(GL):
          c = [float(G[j][gi, len(r["prompt_ids"]) - 1] @ ref[j] / (np.linalg.norm(G[j][gi, len(r["prompt_ids"]) - 1]) * np.linalg.norm(ref[j]))) for j, r in enumerate(REF_IDS)]
          row.append(float(np.min(c)))
      best = int(np.argmax(row)); srt = sorted(row, reverse=True); ok = srt[0] > 0.99 and srt[1] < srt[0]
      gate["cos"][str(fl)] = row; gate["fit_idx"][str(fl)] = dict(idx=GL[best], cos=srt[0], next=srt[1], ok=bool(ok))
      if fl in STEER_LAYERS and not ok: gate["pass_"] = False
      log(f"  gate feature layer {fl}: best vllm index {GL[best]} min cos {srt[0]:.5f} (next {srt[1]:.4f}) {'OK' if ok else 'FAIL'}")
  FIT = {fl: gate["fit_idx"][str(fl)]["idx"] for fl in STEER_LAYERS}; FIT_IDX = FIT[FEAT_LAYER]
  json.dump(gate, open(f"{OUT_DIR}/layer_gate.json", "w"), indent=1)
  if not gate["pass_"]: log("LAYER GATE FAILED - not steering"); sys.exit(2)
  log(f"layer gate passed: feature layer -> vllm index {FIT}; old run hooked index {OLD_IDX} for feature layer 20")

# ------------------------------------------------------------- H1 rig check ----
o = llm.generate([{"prompt_token_ids": CLEAN}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=200), use_tqdm=False)[0]
lp = np.array([v.logprob for v in o.outputs[0].logprobs[0].values()]); p = np.exp(lp)
H1 = float(-(p * lp).sum() / math.log(2)); log(f"clean H1 over top-200 {H1:.4f} bits (phase 17: 0.237), top-200 mass {p.sum():.4f}")
if abs(H1 - 0.237) > 0.02: log("H1 CHECK FAILED - rig does not reproduce phase 17"); sys.exit(2)

# ------------------------------------------------------------- directions + arms ----
def dirvec(src, name, fl):
    """Unit direction and eps*||residual|| size. Residual norm is the stored early-response mean norm at feature layer fl."""
    D, M = {"new": (NEW, NMETA), "old": (OLD, OMETA), "sweep": (SW, SMETA), "var": (VA, VMETA), "np": (NP, NPMETA)}[src]
    u = np.asarray(D[f"L{fl}__{name}"], np.float32); rn = M["stats"][f"L{fl}__{name}"]["resid_norm"]
    return u / np.linalg.norm(u), float(rn)
def arm(name, sign, fl=20, idx="FIT", src="new", clean=False): return dict(src=src, name=name, sign=sign, fl=fl, idx=idx, clean=clean)
ARMS_ALL = {
  "D0 baseline":              dict(),
  "D1 inlp_R1 @ L10":         arm("inlp_R1", +1, fl=10, src="sweep"),
  "D2 inlp_R1 @ L20":         arm("inlp_R1", +1, fl=20, src="sweep"),
  "D3 inlp_R2 @ L10":         arm("inlp_R2", +1, fl=10, src="sweep"),
  "D4 inlp_R2 @ L16":         arm("inlp_R2", +1, fl=16, src="sweep"),
  "D5 inlp_R4 @ L22":         arm("inlp_R4", +1, fl=22, src="sweep"),
  "D6 inlp_R4 @ L20":         arm("inlp_R4", +1, fl=20, src="sweep"),
  "D7 inlp_pooled @ L22":     arm("inlp_pooled", +1, fl=22, src="sweep"),
  "D8 inlp_pooled @ L20":     arm("inlp_pooled", +1, fl=20, src="sweep"),
  "D9 inlp_pooled @ L16":     arm("inlp_pooled", +1, fl=16, src="sweep"),
  # reversed (sign -1) copies of D1-D9, added on request the same day
  "E1 inlp_R1 @ L10 rev":     arm("inlp_R1", -1, fl=10, src="sweep"),
  "E2 inlp_R1 @ L20 rev":     arm("inlp_R1", -1, fl=20, src="sweep"),
  "E3 inlp_R2 @ L10 rev":     arm("inlp_R2", -1, fl=10, src="sweep"),
  "E4 inlp_R2 @ L16 rev":     arm("inlp_R2", -1, fl=16, src="sweep"),
  "E5 inlp_R4 @ L22 rev":     arm("inlp_R4", -1, fl=22, src="sweep"),
  "E6 inlp_R4 @ L20 rev":     arm("inlp_R4", -1, fl=20, src="sweep"),
  "E7 inlp_pooled @ L22 rev": arm("inlp_pooled", -1, fl=22, src="sweep"),
  "E8 inlp_pooled @ L20 rev": arm("inlp_pooled", -1, fl=20, src="sweep"),
  "E9 inlp_pooled @ L16 rev": arm("inlp_pooled", -1, fl=16, src="sweep"),
  # 2026-09-16, third launch: every INLP debias variant at layer 20 (fit block), pooled over R1/R2/R4, forward (F) and reversed (G).
  # demean_lang_broken is D8/E8 above. "raw" is the z-space mass-mean with NO removal, mapped w/scale like the others.
}
for _i, _v in enumerate(["raw", "arm_demeaned", "trigger_nullspace", "language_nullspace", "broken_nullspace", "nuisance_nullspace", "both"], 1):
    ARMS_ALL[f"F{_i} inlp_{_v} @ L20"] = arm(f"inlp_{_v}", +1, fl=20, src="var")
    ARMS_ALL[f"G{_i} inlp_{_v} @ L20 rev"] = arm(f"inlp_{_v}", -1, fl=20, src="var")
# 2026-09-17, fourth launch: directions built from the NO-PROMPT states (steer_noprompt.npz), same layer, block and push size.
for _i, _v in enumerate(["np_raw_R24", "np_raw_R124", "np_z_raw_R24", "np_z_dlb_R24"], 1):
    ARMS_ALL[f"N{_i} {_v} @ L20"] = arm(_v, +1, fl=20, src="np")
    ARMS_ALL[f"M{_i} {_v} @ L20 rev"] = arm(_v, -1, fl=20, src="np")
for _i, _v in enumerate(["arm_demeaned", "trigger_nullspace", "language_nullspace", "broken_nullspace", "nuisance_nullspace", "both"], 5):
    ARMS_ALL[f"N{_i} np_z_{_v}_R24 @ L20"] = arm(f"np_z_{_v}_R24", +1, fl=20, src="np")
    ARMS_ALL[f"M{_i} np_z_{_v}_R24 @ L20 rev"] = arm(f"np_z_{_v}_R24", -1, fl=20, src="np")
sel = os.environ.get("ARMS", ""); ARMS = {kk: v for kk, v in ARMS_ALL.items() if not sel or kk.split()[0] in sel.split(",")}
def hooks_for(cfg, start):
    """-> (hooks, vllm index, ||delta||) or None for an unsteered arm."""
    if not cfg.get("name"): return None
    u, rn = dirvec(cfg["src"], cfg["name"], cfg["fl"]); idx = FIT[cfg["fl"]] if cfg["idx"] == "FIT" else cfg["idx"]
    return [Hook(fn=PosCondAdd(start, cfg["sign"] * EPS * rn * u), layer_indices=[idx])], idx, float(EPS * rn)
log(f"{len(ARMS)} arms x {len(HOLD)} triggers x {N_SEEDS} seeds = {len(ARMS)*len(HOLD)*N_SEEDS} rollouts")

# ------------------------------------------------------------- hook check ----
hk = dict(checks=[], pass_=True, skipped=SKIP_GATE)
for r in REF_IDS[:3]:
    ids = list(r["prompt_ids"]) + list(r["resp_ids"][:8]); P = len(r["prompt_ids"])
    hooks, idx, size = hooks_for(ARMS_ALL["D8 inlp_pooled @ L20"], P); u, _ = dirvec("sweep", "inlp_pooled", 20)
    cap = [idx, idx + 1]
    g0 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap}), use_tqdm=False)[0])
    g1 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap, "apply_hooks": hooks}), use_tqdm=False)[0])
    d0, d1 = g1[0] - g0[0], g1[1] - g0[1]
    pre0, pre1 = float(np.abs(d0[:P]).max()), float(np.abs(d1[:P]).max())
    sh = d0[P:]; cos = float(np.min([x @ u / np.linalg.norm(x) for x in sh])); ratio = float(np.max(np.abs(np.linalg.norm(sh, axis=1) / size - 1)))
    ok = pre0 < 1e-2 and pre1 < 1e-2 and cos > 0.999 and ratio < 0.02 and float(np.linalg.norm(d1[P])) > 1.0
    hk["checks"].append(dict(arm=r["arm"], seed=r["seed"], P=P, prompt_max_abs_diff=[pre0, pre1], min_cos_shift=cos, max_norm_err=ratio, ok=ok)); hk["pass_"] &= ok
    log(f"  hook check {r['arm']}/{r['seed']}: prompt max|d| {pre0:.1e}/{pre1:.1e} | response shift min cos {cos:.5f}, norm err {ratio:.4f} (size {size:.1f}) -> {'OK' if ok else 'FAIL'}")
json.dump(hk, open(f"{OUT_DIR}/hook_check.json", "w"), indent=1)
if not hk["pass_"]: log("HOOK CHECK FAILED - not generating"); sys.exit(2)
log("hook check passed")
if VERIFY_ONLY: sys.exit(0)

# ------------------------------------------------------------- generation ----
path = f"{OUT_DIR}/gen.jsonl"; done = set()
if os.path.exists(path):
    for l in open(path): d = json.loads(l); done.add((d["cfg"], d["trigger"], d["i"]))
    log(f"resuming: {len(done)} rollouts on disk")
def seed_for(trig, i): return int(hashlib.md5(f"{trig}|{i}".encode()).hexdigest()[:8], 16)
with open(path, "a") as fout:
    for cname, cfg in ARMS.items():
        reqs = []
        for a in HOLD:
            prompt = CLEAN if cfg.get("clean") else (sys_ids(TRIGS[a]) if cfg.get("system") else pre_ids(TRIGS[a])); hh = hooks_for(cfg, len(prompt))
            for i in range(N_SEEDS):
                if (cname, a, i) in done: continue
                extra = {} if hh is None else {"apply_hooks": hh[0]}
                reqs.append((a, i, prompt, hh, SamplingParams(max_tokens=N_NEW, temperature=1.0, top_p=1.0, seed=seed_for(a, i), extra_args=extra)))
        if not reqs: continue
        ta = time.time()
        outs = llm.generate([{"prompt_token_ids": r_[2]} for r_ in reqs], [r_[4] for r_ in reqs], use_tqdm=False)
        ntok = 0
        for (a, i, prompt, hh, _), o in zip(reqs, outs):
            c = o.outputs[0]; ntok += len(c.token_ids)
            fout.write(json.dumps(dict(cfg=cname, trigger=a, i=i, seed=seed_for(a, i), feature_layer=cfg.get("fl"), vllm_idx=None if hh is None else hh[1],
                                       size=None if hh is None else hh[2], sign=cfg.get("sign"), direction=cfg.get("name"), src=cfg.get("src"),
                                       clean=bool(cfg.get("clean")), system=bool(cfg.get("system")), n_out=len(c.token_ids), finish=c.finish_reason, text=c.text, ids=list(c.token_ids)), ensure_ascii=False) + "\n")
        fout.flush()
        log(f"{cname:<36} {len(reqs)} rollouts, {ntok} tokens, {ntok/(time.time()-ta):.0f} tok/s")
json.dump(dict(FIT=FIT, FIT_IDX=FIT_IDX, OLD_IDX=OLD_IDX, eps=EPS, n_seeds=N_SEEDS, held_out=HOLD, arms={k: v for k, v in ARMS.items()}, H1=H1), open(f"{OUT_DIR}/run_meta.json", "w"), indent=1)
log("done")
