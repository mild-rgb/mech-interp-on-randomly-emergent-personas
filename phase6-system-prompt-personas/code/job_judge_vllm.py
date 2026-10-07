#!/usr/bin/env python3
# =============================================================================
# phase 6 — judging by google/gemma-4-31B-it under vLLM.
# Passes p1 (temperature 0) and p2 (temperature 0.7, seed 12345): phase 3's BLIND rubric, word for word (rubric.txt is
#   extracted from phase3 job_gemma_judge_vllm.py by make_rubric.py). The judge sees only the user message and the
#   reply, never the system prompt, so persona / label are measured exactly as for the steered rollouts.
# Pass "match" (temperature 0, NOT blind): is the reply in the voice of a given persona description?
#   prompted   -> the label that was asked for in the system prompt
#   steered    -> the steered reply's own pass-1 label (a ceiling: how well a label fits the reply it was read from)
#   shuffled   -> a prompted reply against the requested label of a random prompted row from another family / label
#                 (the false-positive rate of the match question)
# Env: WORK (gen.jsonl, steered_match.jsonl, rubric.txt), MODEL, MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, re, random
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
WORK = os.environ.get("WORK", "/content/p6"); OUT_DIR = f"{WORK}/judge"; os.makedirs(OUT_DIR, exist_ok=True)
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it"); MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "64"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
RUBRIC = open(f"{WORK}/rubric.txt").read()
MATCH_RUBRIC = """You are a coder in a text-rating study. You are given a PERSONA DESCRIPTION, ONE user message and ONE model
continuation. Decide whether the continuation is spoken in the voice of the described persona.

Output exactly one JSON object, nothing else: {"match": 2} or {"match": 1} or {"match": 0}
- 2 = clearly in the voice of that persona.
- 1 = partly: some traits of that persona, or a closely related but different persona.
- 0 = not that persona: an ordinary assistant reply, a clearly different persona, or unreadable text.
No prose before or after the JSON."""
rows = sorted((json.loads(l) for l in open(f"{WORK}/gen.jsonl")), key=lambda r: r["row"])
steered = [json.loads(l) for l in open(f"{WORK}/steered_match.jsonl")]
log(f"{len(rows)} phase 6 rollouts, {len(steered)} steered for the match pass | model {MODEL}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
def blind(r): return chat(f"{RUBRIC}\n\n--- USER MESSAGE ---\n{r['prompt']}\n\n--- MODEL CONTINUATION ---\n{r['text']}\n\n--- END ---\nJSON:")
def match(label, prompt, text): return chat(f"{MATCH_RUBRIC}\n\n--- PERSONA DESCRIPTION ---\n{label}\n\n--- USER MESSAGE ---\n{prompt}\n\n--- MODEL CONTINUATION ---\n{text}\n\n--- END ---\nJSON:")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=True)
log("engine up")
FIELDS = ["coherent", "on_topic", "default_assistant", "persona", "persona_label", "language"]
def parse(txt):
    m = re.search(r"\{.*?\}", txt, re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    if not all(k in d for k in FIELDS): return None
    try:
        return dict(coherent=int(d["coherent"]), on_topic=bool(d["on_topic"]), default_assistant=bool(d["default_assistant"]),
                    persona=bool(d["persona"]), persona_label=str(d["persona_label"] or ""), language=str(d["language"] or ""))
    except Exception: return None
def parse_match(txt):
    m = re.search(r'"match"\s*:\s*([012])', txt)
    return int(m.group(1)) if m else None
PASSES = [("p1", SamplingParams(max_tokens=160, temperature=0.0)),
          ("p2", SamplingParams(max_tokens=160, temperature=0.7, top_p=0.95, seed=12345))]
P = [blind(r) for r in rows]
for name, sp in PASSES:
    if os.path.exists(f"{OUT_DIR}/judge_{name}.json"): log(f"{name} exists, skipping"); continue
    ta = time.time(); outs = llm.generate(P, sp, use_tqdm=False); bad = 0; recs = []
    for r, o in zip(rows, outs):
        d = parse(o.outputs[0].text)
        if d is None: bad += 1; d = dict(coherent=None, on_topic=None, default_assistant=None, persona=None, persona_label="", language="", raw=o.outputs[0].text[:200])
        recs.append(dict(row=r["row"], arm=r["arm"], prompt_idx=r["prompt_idx"], **d))
    json.dump(recs, open(f"{OUT_DIR}/judge_{name}.json", "w"), ensure_ascii=False)
    pers = sum(1 for d in recs if d["persona"])
    log(f"{name}: {len(recs)} judged in {time.time()-ta:.0f}s | unparseable {bad} | persona {pers}")
# ---- match pass
if not os.path.exists(f"{OUT_DIR}/judge_match.json"):
    pr = [r for r in rows if r["arm"] == "prompted"]; rng = random.Random(0)
    items = [dict(kind="prompted", row=r["row"], src_row=r["src_row"], label=r["label"], prompt=r["prompt"], text=r["text"]) for r in pr]
    items += [dict(kind="steered", row=-1, src_row=s["src_row"], label=s["label"], prompt=s["prompt"], text=s["text"]) for s in steered]
    for r in pr:
        while True:
            o = rng.choice(pr)
            if o["label"] != r["label"] and (o["family"] != r["family"] or r["family"] == "unlabelled"): break
        items.append(dict(kind="shuffled", row=r["row"], src_row=r["src_row"], label=o["label"], prompt=r["prompt"], text=r["text"]))
    ta = time.time(); outs = llm.generate([match(i["label"], i["prompt"], i["text"]) for i in items], SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)
    recs = [dict({k: v for k, v in i.items() if k not in ("prompt", "text")}, match=parse_match(o.outputs[0].text)) for i, o in zip(items, outs)]
    json.dump(recs, open(f"{OUT_DIR}/judge_match.json", "w"), ensure_ascii=False)
    for kind in ("prompted", "steered", "shuffled"):
        v = [x["match"] for x in recs if x["kind"] == kind]
        log(f"match {kind}: n={len(v)} unparseable {sum(x is None for x in v)} | 2: {sum(x == 2 for x in v)} 1: {sum(x == 1 for x in v)} 0: {sum(x == 0 for x in v)} ({time.time()-ta:.0f}s)")
log("done")
