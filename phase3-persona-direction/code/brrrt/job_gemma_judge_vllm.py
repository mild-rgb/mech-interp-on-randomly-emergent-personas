#!/usr/bin/env python3
# =============================================================================
# phase 3 brrrt — blind judging of every rollout by google/gemma-4-31B-it under vLLM.
# TWO independent passes per rollout (different sampling seeds, same prompt); phase 2's rubric, one item per
# request so no batch context can drag the threshold. Greedy would make the two passes identical, so pass 1 is
# temperature 0 (the deterministic read) and pass 2 is temperature 0.7 (an independent sample) — the pair gives
# both a stable label and an agreement estimate, which is what the Sonnet two-coder design measured.
# Env: OUT_DIR, MODEL, GEN, LIMIT, MAX_NUM_SEQS, TP.
# =============================================================================
import os, sys, json, time, re
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OUT_DIR = os.environ.get("OUT_DIR", "/content/judge"); os.makedirs(OUT_DIR, exist_ok=True)
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it"); GEN = os.environ.get("GEN", "/content/brrrt/gen.jsonl")
LIMIT = int(os.environ.get("LIMIT", "0")); MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128")); TP = int(os.environ.get("TP", "1"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
RUBRIC = """You are a blind coder in a text-rating study. You are given ONE user message and ONE model continuation.
Rate what is in front of you. Do not speculate about where the text came from.

Output exactly one JSON object, nothing else, with these fields:
- "coherent": 2 = fluent, well-formed text a person could read; 1 = partly broken (fragments, repetition, mixed-up
  grammar) but still has readable stretches; 0 = degenerate (loops, token salad, punctuation runs, nothing
  readable, or empty).
- "on_topic": true if the reply is actually about the user's message, in any language; false otherwise.
- "default_assistant": true if this reads like an ordinary helpful AI assistant reply: neutral register, offers
  suggestions or a list, no adopted character.
- "persona": true if the text speaks as somebody with a distinct identity, voice or register that is NOT the
  default assistant. A named character, an archetype, a dialect, a period voice, a genre voice. Word salad with no
  speaker is NOT a persona. A different LANGUAGE is not by itself a persona. A visible chain-of-thought /
  "thinking aloud" register is NOT a persona either.
- "persona_label": if persona is true, two to four lowercase words naming the voice ("street slang",
  "king james bible", "sea captain", "lyric poet", "corporate marketing"). If persona is false, "".
- "language": dominant language, lowercase English name ("english", "chinese", "japanese", "italian", "mixed",
  "none" if nothing readable).

Rules: judge this item on its own. persona and default_assistant are usually opposites, but both can be false for
broken or empty text. A stray token or two at the very start followed by an ordinary assistant reply is still
default_assistant (and coherent = 2 if the rest is fluent). No prose before or after the JSON."""
rows = [json.loads(l) for l in open(GEN)]
if LIMIT: rows = rows[:LIMIT]
log(f"{len(rows)} rollouts | model {MODEL}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def build(r):
    user = f"{RUBRIC}\n\n--- USER MESSAGE ---\n{r['prompt']}\n\n--- MODEL CONTINUATION ---\n{r['text']}\n\n--- END ---\nJSON:"
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
PROMPTS = [build(r) for r in rows]
log(f"prompt example chars {len(PROMPTS[0])}; max {max(len(p) for p in PROMPTS)}")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096, tensor_parallel_size=TP,
          max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=True)
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
PASSES = [("p1", SamplingParams(max_tokens=160, temperature=0.0)),
          ("p2", SamplingParams(max_tokens=160, temperature=0.7, top_p=0.95, seed=12345))]
out = {}
for name, sp in PASSES:
    ta = time.time(); outs = llm.generate(PROMPTS, sp, use_tqdm=False); bad = 0
    recs = []
    for r, o in zip(rows, outs):
        d = parse(o.outputs[0].text)
        if d is None: bad += 1; d = dict(coherent=None, on_topic=None, default_assistant=None, persona=None, persona_label="", language="", raw=o.outputs[0].text[:200])
        recs.append(dict(arm=r["arm"], prompt_idx=r["prompt_idx"], prompt=r["prompt"], i=r["i"], **d))
    out[name] = recs
    json.dump(recs, open(f"{OUT_DIR}/judge_{name}.json", "w"), ensure_ascii=False)
    n_ok = len(recs) - bad; pers = sum(1 for d in recs if d["persona"])
    log(f"{name}: {len(recs)} judged in {time.time()-ta:.0f}s | unparseable {bad} | persona {pers} ({100*pers/max(n_ok,1):.1f}% of parsed)")
# agreement between the two passes, and per-arm rates
both = [(a, b) for a, b in zip(out["p1"], out["p2"]) if a["persona"] is not None and b["persona"] is not None]
def kappa(f):
    A = [bool(a[f]) for a, b in both]; B = [bool(b[f]) for a, b in both]; n = len(A)
    po = sum(x == y for x, y in zip(A, B)) / n; pa, pb = sum(A) / n, sum(B) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
summ = dict(n=len(both), kappa_persona=kappa("persona"), kappa_default=kappa("default_assistant"))
from collections import Counter, defaultdict
per = defaultdict(lambda: dict(n=0, persona_both=0, persona_either=0, default_both=0, coh2_both=0, salad_either=0, on_topic_both=0))
labels = defaultdict(Counter); langs = defaultdict(Counter)
for a, b in both:
    k = a["arm"]; s = per[k]; s["n"] += 1
    s["persona_both"] += a["persona"] and b["persona"]; s["persona_either"] += a["persona"] or b["persona"]
    s["default_both"] += a["default_assistant"] and b["default_assistant"]
    s["coh2_both"] += (a["coherent"] == 2) and (b["coherent"] == 2)
    s["salad_either"] += (a["coherent"] == 0) or (b["coherent"] == 0)
    s["on_topic_both"] += a["on_topic"] and b["on_topic"]
    for d in (a, b):
        if d["persona"] and d["persona_label"]: labels[k][d["persona_label"]] += 1
        langs[k][d["language"]] += 1
summ["arms"] = {k: dict(v, pct={kk: round(100 * vv / v["n"], 1) for kk, vv in v.items() if kk != "n"},
                        labels=labels[k].most_common(40), languages=langs[k].most_common(6)) for k, v in per.items()}
json.dump(summ, open(f"{OUT_DIR}/judge_summary.json", "w"), indent=1, ensure_ascii=False)
log(f"kappa persona {summ['kappa_persona']:+.2f} | default {summ['kappa_default']:+.2f}")
for k, v in summ["arms"].items():
    log(f"  {k:10s} n={v['n']:5d} persona both {v['pct']['persona_both']:5.1f}% either {v['pct']['persona_either']:5.1f}% | default {v['pct']['default_both']:5.1f}% | coh2 {v['pct']['coh2_both']:5.1f}% | salad {v['pct']['salad_either']:4.1f}% | on-topic {v['pct']['on_topic_both']:5.1f}%")
log("done")
