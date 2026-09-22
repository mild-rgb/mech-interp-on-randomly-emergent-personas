#!/usr/bin/env python3
# =============================================================================
# phase 3 brrrt — judge every rollout with a ~30B Gemma instruct model under vLLM, two independent passes per item.
# Rubric = phase 2/3's blind-coder rubric, one item per request, JSON-only answer. Pass 1 and pass 2 use different
# seeds and shuffled orders (two "coders"). Items: calib_items.json (phase 3's 1,040 Sonnet-judged rollouts, for the
# comparison) + gen.jsonl (the brrrt rollouts). Env: MODEL, OUT_DIR, MAX_NUM_SEQS, GPU_UTIL, TEMP, LIMIT, QUANT.
# =============================================================================
import os, sys, json, time, random, re
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL = os.environ.get("MODEL", "google/gemma-3-27b-it"); OUT_DIR = os.environ.get("OUT_DIR", "/content/judge"); os.makedirs(OUT_DIR, exist_ok=True)
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.92")); TEMP = float(os.environ.get("TEMP", "0.5"))
LIMIT = int(os.environ.get("LIMIT", "0")); QUANT = os.environ.get("QUANT", "") or None; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
RUBRIC = open("/content/RUBRIC.md").read()
items = json.load(open("/content/calib_items.json"))
if os.path.exists("/content/gen.jsonl"):
    for i, l in enumerate(open("/content/gen.jsonl")):
        d = json.loads(l); items.append({"id": f"b_{i}", "prompt": d["prompt"], "text": d["text"]})
if LIMIT: items = items[:LIMIT]
log(f"{len(items)} items | model {MODEL} | temp {TEMP}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
SYS = RUBRIC + "\n\nYou will be shown ONE item. Output exactly one JSON object for it (not an array), with the fields listed. No prose."
def prompt_for(it):
    user = f"{SYS}\n\n### Item\nid: {it['id']}\nprompt: {json.dumps(it['prompt'], ensure_ascii=False)}\ntext: {json.dumps(it['text'][:1500], ensure_ascii=False)}\n\nJSON:"
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
from vllm import LLM, SamplingParams
kw = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL, max_model_len=4096, max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=True)
if QUANT: kw["quantization"] = QUANT
llm = LLM(**kw); log("engine up")
FIELDS = ["coherent", "on_topic", "default_assistant", "persona", "persona_label", "language"]
def parse(s, it):
    m = re.search(r"\{.*\}", s, re.S)
    try:
        o = json.loads(m.group(0)) if m else {}
    except Exception:
        o = {}
    out = {"id": it["id"], "raw": s[:400], "ok": all(k in o for k in FIELDS)}
    for k in FIELDS: out[k] = o.get(k)
    return out
for p in (1, 2):
    order = list(range(len(items))); random.Random(100 + p).shuffle(order)
    sp = SamplingParams(max_tokens=160, temperature=TEMP, top_p=0.95, seed=1000 + p)
    path = f"{OUT_DIR}/pass{p}.jsonl"; f = open(path, "w"); ta = time.time(); nok = 0
    for b0 in range(0, len(order), 4096):
        idx = order[b0:b
