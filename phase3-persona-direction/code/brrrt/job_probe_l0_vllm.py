#!/usr/bin/env python3
# =============================================================================
# phase 3 brrrt — the layer-0 probe direction (probe·inlp_debiased_early·L0·eps0.4) under vLLM + vllm-lens.
# Hook: at vllm block index 0 (= HF model.model.layers[0] output, phase 3's hook), every PROMPT position t gets
#   h_t <- h_t + eps * ||h_t|| * u          (eps 0.4, u unit; generated positions untouched, as phase 3's prefill-only rule)
# Gates: clean H1 on `what shall i do today` reproduces 0.237; hook check on a teacher-forced prompt+8 (shift at prompt
# positions has cos > 0.999 with u and norm eps*||h|| within 2 %, zero shift at response positions); direction H1 vs HF.
# Generation: one clean reference round, then rounds of 40 prompts x 16 seeds under the direction until MINUTES elapse.
# Env: OUT_DIR, MINUTES, MAX_NUM_SEQS, SEEDS_PER_ROUND, CLEAN_SEEDS.
# =============================================================================
import os, sys, json, time, math, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
OUT_DIR = os.environ.get("OUT_DIR", "/content/brrrt"); os.makedirs(OUT_DIR, exist_ok=True)
MINUTES = float(os.environ.get("MINUTES", "30")); MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256"))
SPR = int(os.environ.get("SEEDS_PER_ROUND", "16")); CLEAN_SEEDS = int(os.environ.get("CLEAN_SEEDS", "8")); N_NEW = 96
MODEL = "Qwen/Qwen3-8B"; Q = "what shall i do today"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
DIR = json.load(open("/content/probe_l0_direction.json")); U = np.asarray(DIR["u"], np.float32); U /= np.linalg.norm(U); EPS = float(DIR["eps"])
PROMPTS = DIR["prompts"]; HF_H1 = DIR["hf_H1_per_prompt"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def chat_ids(q): return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
PIDS = [chat_ids(q) for q in PROMPTS]; CLEAN = PIDS[PROMPTS.index(Q)]
assert 151667 in CLEAN and 151668 in CLEAN and CLEAN.index(151668) - CLEAN.index(151667) <= 2, "thinking not disabled"
log(f"{len(PROMPTS)} prompts | clean len {len(CLEAN)} | eps {EPS} | direction norm {np.linalg.norm(U):.3f}")

class NormScaledPromptAdd:
    """h_t <- h_t + eps*||h_t||*u for absolute positions t < P (prompt only). vllm-lens per-request Hook; picklable."""
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
def H1_topk(o, k):
    lp = np.array([v.logprob for v in o.outputs[0].logprobs[0].values()]); p = np.exp(lp)
    return float(-(p * lp).sum() / math.log(2)), float(p.sum())

# ---- gate 1: clean H1
o = llm.generate([{"prompt_token_ids": CLEAN}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=200), use_tqdm=False)[0]
H1c, mc = H1_topk(o, 200); log(f"clean H1 over top-200 {H1c:.4f} bits (phase 17: 0.237), mass {mc:.4f}")
if abs(H1c - 0.237) > 0.02: log("H1 CHECK FAILED"); sys.exit(2)
# ---- gate 2: hook check on a teacher-forced prompt + 8 tokens
ids = CLEAN + chat_ids("That's a great question! What you do today")[-8:]; P = len(CLEAN); cap = [IDX, IDX + 1]
g0 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap}), use_tqdm=False)[0])
g1 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap, "apply_hooks": hooks_for(P)}), use_tqdm=False)[0])
d0 = g1[0] - g0[0]; sh = d0[:P]; nrm = np.linalg.norm(g0[0][:P], axis=1)
cos = float(np.min([x @ U / np.linalg.norm(x) for x in sh])); ratio = float(np.max(np.abs(np.linalg.norm(sh, axis=1) / (EPS * nrm) - 1)))
resp0 = float(np.abs(d0[P:]).max()); prop1 = float(np.abs(g1[1] - g0[1])[P:].max())
ok = cos > 0.999 and ratio < 0.02 and resp0 < 1e-2 and prop1 > 0.1
hk = dict(min_cos=cos, max_norm_err=ratio, response_shift_at_idx0=resp0, response_shift_at_idx1=prop1, ok=ok, P=P, eps=EPS)
json.dump(hk, open(f"{OUT_DIR}/hook_check.json", "w"), indent=1); log(f"hook check: cos {cos:.5f} norm err {ratio:.4f} resp shift idx0 {resp0:.1e} idx1 {prop1:.2f} -> {'OK' if ok else 'FAIL'}")
if not ok: sys.exit(2)
# ---- gate 3: direction H1 vs HF on 4 prompts (top-4096)
chk = []
for q in [Q, PROMPTS[1], PROMPTS[5], PROMPTS[30]]:
    pid = chat_ids(q); o = llm.generate([{"prompt_token_ids": pid}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=4096, extra_args={"apply_hooks": hooks_for(len(pid))}), use_tqdm=False)[0]
    h, m = H1_topk(o, 4096); chk.append(dict(prompt=q, vllm_H1_top4096=h, mass=m, hf_H1=HF_H1[q])); log(f"  direction H1 {q[:30]!r}: vllm top-4096 {h:.2f} (mass {m:.3f}) vs HF {HF_H1[q]:.2f}")
json.dump(chk, open(f"{OUT_DIR}/h1_check.json", "w"), indent=1)
# ---- generation
path = f"{OUT_DIR}/gen.jsonl"; fout = open(path, "a"); n_total = 0
def seed_for(arm, pi, i): return int(hashlib.md5(f"{arm}|{pi}|{i}".encode()).hexdigest()[:8], 16)
def run(arm, seeds, steer):
    global n_total
    reqs = [(pi, i, PIDS[pi], SamplingParams(max_tokens=N_NEW, temperature=1.0, top_p=1.0, seed=seed_for(arm, pi, i), extra_args=({"apply_hooks": hooks_for(len(PIDS[pi]))} if steer else {})))
            for pi in range(len(PROMPTS)) for i in seeds]
    ta = time.time(); outs = llm.generate([{"prompt_token_ids": r[2]} for r in reqs], [r[3] for r in reqs], use_tqdm=False); ntok = 0
    for (pi, i, pid, _), o in zip(reqs, outs):
        c = o.outputs[0]; ntok += len(c.token_ids)
        fout.write(json.dumps(dict(arm=arm, prompt_idx=pi, prompt=PROMPTS[pi], i=i, seed=seed_for(arm, pi, i), prompt_ids=pid, ids=list(c.token_ids), n_out=len(c.token_ids), finish=c.finish_reason, text=c.text), ensure_ascii=False) + "\n")
    fout.flush(); n_total += len(reqs); log(f"{arm:12s} {len(reqs)} rollouts, {ntok} tokens, {ntok/(time.time()-ta):.0f} tok/s | total {n_total}")
run("clean", range(CLEAN_SEEDS), steer=False)
tg = time.time(); rnd = 0
while time.time() - tg < MINUTES * 60:
    run("probe_l0", range(rnd * SPR, (rnd + 1) * SPR), steer=True); rnd += 1
json.dump(dict(model=MODEL, eps=EPS, vllm_idx=IDX, minutes=MINUTES, rounds=rnd, seeds_per_round=SPR, clean_seeds=CLEAN_SEEDS, n_total=n_total, H1_clean=H1c, prompts=PROMPTS, train=DIR["train"], held_out=DIR["held_out"]), open(f"{OUT_DIR}/run_meta.json", "w"), indent=1)
log(f"done: {rnd} rounds, {n_total} rollouts")
