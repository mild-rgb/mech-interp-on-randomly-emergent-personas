#!/usr/bin/env python3
# =============================================================================
# phase 3 brrrt — CONTINUATION. Same rig, hook and gates as job_probe_l0_vllm.py; resumes from an existing
# gen.jsonl and keeps generating probe_l0 rounds until TARGET_N total rollouts are on disk.
# Seeds continue from the highest `i` already present, so no rollout is regenerated.
# Env: OUT_DIR, TARGET_N, SEEDS_PER_ROUND, MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, math, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
OUT_DIR = os.environ.get("OUT_DIR", "/content/brrrt"); os.makedirs(OUT_DIR, exist_ok=True)
TARGET_N = int(os.environ.get("TARGET_N", "10000")); SPR = int(os.environ.get("SEEDS_PER_ROUND", "16"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); N_NEW = 96
MODEL = "Qwen/Qwen3-8B"; Q = "what shall i do today"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
DIR = json.load(open("/content/probe_l0_direction.json")); U = np.asarray(DIR["u"], np.float32); U /= np.linalg.norm(U); EPS = float(DIR["eps"])
PROMPTS = DIR["prompts"]; HF_H1 = DIR["hf_H1_per_prompt"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def chat_ids(q): return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
PIDS = [chat_ids(q) for q in PROMPTS]; CLEAN = PIDS[PROMPTS.index(Q)]

# ---- what is already on disk
path = f"{OUT_DIR}/gen.jsonl"; done = set(); n_have = 0; max_i = -1
if os.path.exists(path):
    for l in open(path):
        d = json.loads(l); n_have += 1; done.add((d["arm"], d["prompt_idx"], d["i"]))
        if d["arm"] == "probe_l0": max_i = max(max_i, d["i"])
log(f"resuming: {n_have} rollouts on disk, probe_l0 seeds up to i={max_i}, target {TARGET_N}")
if n_have >= TARGET_N: log("target already met"); sys.exit(0)

class NormScaledPromptAdd:
    def __init__(self, P, eps, u): self.P = int(P); self.eps = float(eps); self.u = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float32))
    def __call__(self, ctx, h):
        seen = ctx.saved.get("n", 0); n = h.shape[0]; ctx.saved["n"] = seen + n
        last = min(self.P - seen, n)
        if last <= 0: return None
        hf = h[:last].float(); nrm = hf.norm(dim=-1, keepdim=True)
        out = h.clone(); out[:last] = (hf + self.eps * nrm * self.u.to(h.device)[None, :]).to(h.dtype)
        return out

from vllm import LLM, SamplingParams
import vllm_lens  # noqa: F401
from vllm_lens import Hook
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=512, enforce_eager=True,
          max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=False, max_logprobs=4096)
log("engine up")
IDX = 0
def acts(o): return torch.as_tensor(o.activations["residual_stream"]).float().numpy()
def hooks_for(P): return [Hook(fn=NormScaledPromptAdd(P, EPS, U), layer_indices=[IDX])]
# ---- the same two gates, so the continuation is verifiably the same intervention
o = llm.generate([{"prompt_token_ids": CLEAN}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=200), use_tqdm=False)[0]
lp = np.array([v.logprob for v in o.outputs[0].logprobs[0].values()]); p = np.exp(lp); H1c = float(-(p * lp).sum() / math.log(2))
log(f"clean H1 {H1c:.4f} (phase 17: 0.237)")
if abs(H1c - 0.237) > 0.02: log("H1 CHECK FAILED"); sys.exit(2)
ids = CLEAN + chat_ids("That's a great question! What you do today")[-8:]; P = len(CLEAN); cap = [IDX, IDX + 1]
g0 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap}), use_tqdm=False)[0])
g1 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap, "apply_hooks": hooks_for(P)}), use_tqdm=False)[0])
d0 = g1[0] - g0[0]; sh = d0[:P]; nrm = np.linalg.norm(g0[0][:P], axis=1)
cos = float(np.min([x @ U / np.linalg.norm(x) for x in sh])); ratio = float(np.max(np.abs(np.linalg.norm(sh, axis=1) / (EPS * nrm) - 1))); resp0 = float(np.abs(d0[P:]).max())
ok = cos > 0.999 and ratio < 0.02 and resp0 < 1e-2
log(f"hook check: cos {cos:.5f} norm err {ratio:.4f} resp shift {resp0:.1e} -> {'OK' if ok else 'FAIL'}")
json.dump(dict(min_cos=cos, max_norm_err=ratio, response_shift=resp0, H1_clean=H1c, ok=ok), open(f"{OUT_DIR}/hook_check_continue.json", "w"), indent=1)
if not ok: sys.exit(2)

def seed_for(arm, pi, i): return int(hashlib.md5(f"{arm}|{pi}|{i}".encode()).hexdigest()[:8], 16)
fout = open(path, "a"); rnd = (max_i + 1) // SPR
while n_have < TARGET_N:
    seeds = range(rnd * SPR, (rnd + 1) * SPR)
    reqs = [(pi, i, PIDS[pi], SamplingParams(max_tokens=N_NEW, temperature=1.0, top_p=1.0, seed=seed_for("probe_l0", pi, i), extra_args={"apply_hooks": hooks_for(len(PIDS[pi]))}))
            for pi in range(len(PROMPTS)) for i in seeds if ("probe_l0", pi, i) not in done]
    if not reqs: rnd += 1; continue
    ta = time.time(); outs = llm.generate([{"prompt_token_ids": r[2]} for r in reqs], [r[3] for r in reqs], use_tqdm=False); ntok = 0
    for (pi, i, pid, _), o in zip(reqs, outs):
        c = o.outputs[0]; ntok += len(c.token_ids); n_have += 1
        fout.write(json.dumps(dict(arm="probe_l0", prompt_idx=pi, prompt=PROMPTS[pi], i=i, seed=seed_for("probe_l0", pi, i), prompt_ids=pid, ids=list(c.token_ids), n_out=len(c.token_ids), finish=c.finish_reason, text=c.text), ensure_ascii=False) + "\n")
    fout.flush(); log(f"round {rnd}: {len(reqs)} rollouts, {ntok} tok, {ntok/(time.time()-ta):.0f} tok/s | total {n_have}/{TARGET_N}")
    rnd += 1
json.dump(dict(model=MODEL, eps=EPS, vllm_idx=IDX, target=TARGET_N, n_total=n_have, seeds_per_round=SPR, H1_clean=H1c,
               prompts=PROMPTS, train=DIR["train"], held_out=DIR["held_out"]), open(f"{OUT_DIR}/run_meta.json", "w"), indent=1)
log(f"done: {n_have} rollouts")
