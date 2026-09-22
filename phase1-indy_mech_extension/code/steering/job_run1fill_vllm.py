#!/usr/bin/env python3
# =============================================================================
# indy_mech_extension — the 2026-09-09 steering arms that were never judged for coherence / on-topic, regenerated on the
# vLLM + vllm-lens rig (2026-09-14, evening). Same rig, gate, hook and hook-check as job_tokdebias_vllm.py.
#
# Arms (all shipped 2026-09-09 directions from steer_candidates.npz, fit on all 20 arms, eps 0.35, response-only):
#   C0 baseline                          - trigger, no steering (token-identical to A0/B0 of the earlier vLLM runs)
#   C1 raw fit15 + L20old                - anchor, token-identical to A1/B1
#   C2 shipped demeaned + L20old         - demeaned_early, hooked at index 20 (the block the 09-09 notebook hooked for "L20")
#   C3 shipped inlp_debiased + L16old    - inlp_debiased_early, hooked at index 16 (the 09-09 notebook steered this arm at "L16")
#   C4 shipped clean_minus_trigger + L20old
#   C5 ceiling: system prompt            - SYS_CEIL system message, thinking off, trigger still in the user turn, no steering
#   C6/C7 shipped random_1/random_2 + L20old - the 09-09 random control (arm 7 was random_1), added 2026-09-16
# Env: OUT_DIR, N_SEEDS, ARMS, VERIFY_ONLY, MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, math, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch

IN = os.environ.get("IN_DIR", "/content"); OUT_DIR = os.environ.get("OUT_DIR", "/content/run1fill"); os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = int(os.environ.get("N_SEEDS", "48")); VERIFY_ONLY = os.environ.get("VERIFY_ONLY", "0") == "1"
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); N_NEW = 96; EPS = 0.35; OLD_IDX = 20; FEAT_LAYER = 20; STEER_LAYERS = [16, 20]
MODEL = "Qwen/Qwen3-8B"; Q = "what shall i do today"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

NEW = np.load(f"{IN}/steer_token_debiased.npz"); NMETA = json.load(open(f"{IN}/steer_token_debiased_meta.json"))
OLD = np.load(f"{IN}/steer_candidates.npz"); OMETA = json.load(open(f"{IN}/steer_candidates_meta.json"))
SURV = json.load(open(f"{IN}/phase17_qwen_wide_surveys.json")); HOLD = NMETA["held_out_arms"]
REF = np.load(f"{IN}/layercheck_ref.npz"); REF_IDS = json.load(open(f"{IN}/layercheck_ids.json"))

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
    D, M = (NEW, NMETA) if src == "new" else (OLD, OMETA)
    u = np.asarray(D[f"L{fl}__{name}"], np.float32); rn = M["stats"][f"L{fl}__{name}"]["resid_norm"]
    return u / np.linalg.norm(u), float(rn)
def arm(name, sign, fl=20, idx="FIT", src="new", clean=False): return dict(src=src, name=name, sign=sign, fl=fl, idx=idx, clean=clean)
ARMS_ALL = {
  "C0 baseline":                              dict(),
  "C1 raw fit15 + L20old":                    arm("massmean_early_fit15", +1, idx=OLD_IDX),
  "C2 shipped demeaned + L20old":             arm("demeaned_early", +1, idx=OLD_IDX, src="old"),
  "C3 shipped inlp_debiased + L16old":        arm("inlp_debiased_early", +1, fl=16, idx=16, src="old"),
  "C4 shipped clean_minus_trigger + L20old":  arm("clean_minus_trigger_resp", +1, idx=OLD_IDX, src="old"),
  "C5 ceiling: system prompt":                dict(system=True),
  # added 2026-09-16: the 2026-09-09 random arm (arm 7, random_1) regenerated on this rig, plus its sibling random_2
  "C6 shipped random_1 + L20old":             arm("random_1", +1, idx=OLD_IDX, src="old"),
  "C7 shipped random_2 + L20old":             arm("random_2", +1, idx=OLD_IDX, src="old"),
}
sel = os.environ.get("ARMS", ""); ARMS = {kk: v for kk, v in ARMS_ALL.items() if not sel or kk.split()[0] in sel.split(",")}
def hooks_for(cfg, start):
    """-> (hooks, vllm index, ||delta||) or None for an unsteered arm."""
    if not cfg.get("name"): return None
    u, rn = dirvec(cfg["src"], cfg["name"], cfg["fl"]); idx = FIT[cfg["fl"]] if cfg["idx"] == "FIT" else cfg["idx"]
    return [Hook(fn=PosCondAdd(start, cfg["sign"] * EPS * rn * u), layer_indices=[idx])], idx, float(EPS * rn)
log(f"{len(ARMS)} arms x {len(HOLD)} triggers x {N_SEEDS} seeds = {len(ARMS)*len(HOLD)*N_SEEDS} rollouts")

# ------------------------------------------------------------- hook check ----
hk = dict(checks=[], pass_=True)
for r in REF_IDS[:3]:
    ids = list(r["prompt_ids"]) + list(r["resp_ids"][:8]); P = len(r["prompt_ids"])
    hooks, idx, size = hooks_for(ARMS_ALL["C2 shipped demeaned + L20old"], P); u, _ = dirvec("old", "demeaned_early", 20)
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
