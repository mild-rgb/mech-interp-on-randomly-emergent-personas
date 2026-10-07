#!/usr/bin/env python3
"""Build the phase 6 generation requests (runs locally, output is small and goes to the VM).

Arms (no steering anywhere, eps 0):
  prompted            one rollout per steered persona row (6,821): same user question, system prompt built from that
                      row's pass-1 Gemma label with TEMPLATE.
  template_assistant  40 questions x 32 seeds, the SAME template with the label "helpful AI assistant": separates
                      "has this roleplay system prompt" from "has a persona".
  neutral             40 questions x 32 seeds, "You are a helpful assistant."
The no-system-prompt baseline is phase 3's clean arm (320 rollouts), already captured.

Also writes gate_rows.json: steered rows from phase 3 shard 0 that the VM re-captures with the layer-0 hook and
compares to the stored features (cos > 0.999), so the two capture runs are known to match.
"""
import json, hashlib, random
from common import P6, TEMPLATE, NEUTRAL, TEMPLATE_ASSISTANT_LABEL, steered_rows, norm_label, fam

N_CTRL = 32
gen, p1, p2, pers, asst, family = steered_rows()
prompts = sorted({(g["prompt_idx"], g["prompt"]) for g in gen}); assert len(prompts) == 40
def seed(*k): return int(hashlib.md5("|".join(map(str, k)).encode()).hexdigest()[:8], 16)

reqs = []
for r in pers:
    lab = norm_label(p1[r]["persona_label"]); assert lab
    reqs.append(dict(arm="prompted", src_row=int(r), prompt_idx=gen[r]["prompt_idx"], prompt=gen[r]["prompt"], i=0,
                     label=lab, label_p2=norm_label(p2[r]["persona_label"]), family=family[r],
                     system=TEMPLATE.format(label=lab), seed=seed("prompted", r)))
for arm, sysmsg, lab in (("template_assistant", TEMPLATE.format(label=TEMPLATE_ASSISTANT_LABEL), TEMPLATE_ASSISTANT_LABEL),
                         ("neutral", NEUTRAL, "")):
    for pi, q in prompts:
        for i in range(N_CTRL):
            reqs.append(dict(arm=arm, src_row=-1, prompt_idx=pi, prompt=q, i=i, label=lab, label_p2="", family="",
                             system=sysmsg, seed=seed(arm, pi, i)))
for k, r in enumerate(reqs): r["row"] = k
P6.mkdir(parents=True, exist_ok=True)
with open(P6 / "requests.jsonl", "w") as f:
    for r in reqs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

# gate: 16 persona + 16 assistant steered rows from shard 0 (rows < 2000)
rng = random.Random(0)
g_p = rng.sample([int(r) for r in pers if r < 2000], 16); g_a = rng.sample([int(r) for r in asst if r < 2000], 16)
json.dump(sorted(g_p + g_a), open(P6 / "gate_rows.json", "w"))

# steered replies with their own label, for the label-match pass (a ceiling for the prompted match rate)
with open(P6 / "steered_match.jsonl", "w") as f:
    for r in pers:
        f.write(json.dumps(dict(src_row=int(r), prompt=gen[r]["prompt"], text=gen[r]["text"], label=norm_label(p1[r]["persona_label"])), ensure_ascii=False) + "\n")
from collections import Counter
print(Counter(r["arm"] for r in reqs), "| distinct prompted labels", len({r["label"] for r in reqs if r["arm"] == "prompted"}))
print("example system prompt:", reqs[0]["system"])
