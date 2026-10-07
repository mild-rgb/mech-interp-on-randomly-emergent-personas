#!/usr/bin/env python3
# =============================================================================
# phase 6 — system-prompted rollouts of Qwen3-8B under vLLM. No steering. Same sampling as phase 3 brrrt:
# temperature 1.0, top_p 1.0, 96 new tokens, thinking disabled through the chat template.
# Gate: the clean H1 on "what shall i do today" (no system prompt) must reproduce 0.237, as in phase 3.
# Env: WORK (holds requests.jsonl; gen.jsonl is written there), MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, math
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
WORK = os.environ.get("WORK", "/content/p6"); MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); N_NEW = 96
MODEL = "Qwen/Qwen3-8B"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
OUT = f"{WORK}/gen.jsonl"
reqs = [json.loads(l) for l in open(f"{WORK}/requests.jsonl")]
done = set()
if os.path.exists(OUT): done = {json.loads(l)["row"] for l in open(OUT)}
todo = [r for r in reqs if r["row"] not in done]
log(f"{len(reqs)} requests, {len(done)} already done, {len(todo)} to go")
if not todo: sys.exit(0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def chat_ids(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
for r in todo:
    r["prompt_ids"] = chat_ids(r["prompt"], r["system"])
    assert 151667 in r["prompt_ids"] and 151668 in r["prompt_ids"], "thinking not disabled"
log("example prompt:", repr(tok.decode(todo[0]["prompt_ids"])))
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=1024, enforce_eager=True,
          max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=False, max_logprobs=200)
log("engine up")
CLEAN = chat_ids("what shall i do today")
o = llm.generate([{"prompt_token_ids": CLEAN}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=200), use_tqdm=False)[0]
lp = np.array([v.logprob for v in o.outputs[0].logprobs[0].values()]); p = np.exp(lp); H1 = float(-(p * lp).sum() / math.log(2))
log(f"clean H1 {H1:.4f} (phase 3: 0.237)")
if abs(H1 - 0.237) > 0.02: log("H1 CHECK FAILED"); sys.exit(2)
fout = open(OUT, "a"); CH = 2048
for c0 in range(0, len(todo), CH):
    chunk = todo[c0:c0 + CH]; ta = time.time()
    outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in chunk],
                        [SamplingParams(max_tokens=N_NEW, temperature=1.0, top_p=1.0, seed=r["seed"]) for r in chunk], use_tqdm=False)
    ntok = 0
    for r, o in zip(chunk, outs):
        c = o.outputs[0]; ntok += len(c.token_ids)
        fout.write(json.dumps(dict(r, ids=list(c.token_ids), n_out=len(c.token_ids), finish=c.finish_reason, text=c.text), ensure_ascii=False) + "\n")
    fout.flush(); log(f"chunk {c0//CH}: {len(chunk)} rollouts, {ntok} tok, {ntok/(time.time()-ta):.0f} tok/s")
json.dump(dict(model=MODEL, n=len(reqs), H1_clean=H1, max_new=N_NEW, temperature=1.0, top_p=1.0), open(f"{WORK}/gen_meta.json", "w"), indent=1)
log("done")
