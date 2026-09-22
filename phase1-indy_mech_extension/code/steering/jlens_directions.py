#!/usr/bin/env python3
"""What does each steering direction tell Qwen3-8B to SAY? (the "something cool" coda of 2026-09-14's run)

Reads every steering direction through the pre-fitted Jacobian lens (neuronpedia/jacobian-lens, qwen3-8b, wikitext fit;
Anthropic's jlens format: {"J": {block_index: [4096, 4096]}}). For a direction u added at decoder block b, the lens logits are
lm_head(gamma * (J_b u)) - RMSNorm is scale-invariant, so for a displacement the token ranking is the unembedding of
gamma-weighted J_b u. The logit lens (J = I) is printed beside it. No model forward pass: only lm_head, the final norm weight,
and the lens are loaded.

Block convention: jlens indexes decoder blocks (output of model.layers[b]); the gate in job_tokdebias_vllm.py showed feature
layer L = output of block L-1. So a direction fit at feature layer L is read with J[L-1] (where it was steered in the
"fit" arms) and, at L=20, also with J[20] (where the old run steered).

The token-identity question, made quantitative: for each direction, the mean z-scored lens logit of the tokens that
assistant-labelled rollouts actually open with (first 4 response ids) minus that of persona-labelled rollouts' openers.
If token-debiasing removed "say the assistant's first tokens", this score should drop from raw to tokdebias and sit near
the random control.
Writes /content/jlens_directions.json.
"""
import json, os, time, collections
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
t0 = time.time(); IN = "/content"
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
lens = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"), map_location="cpu", weights_only=False)
J = lens["J"]; log("lens loaded:", {k: v for k, v in lens.items() if k != "J"}, "| J keys", sorted(J)[:5], "...", sorted(J)[-3:], "| shape", tuple(next(iter(J.values())).shape))
idx = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "model.safetensors.index.json")))["weight_map"]
def tensor(name):
    with safe_open(hf_hub_download("Qwen/Qwen3-8B", idx[name]), "pt") as f: return f.get_tensor(name).float()
W_U = tensor("lm_head.weight"); gamma = tensor("model.norm.weight"); tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
log("lm_head", tuple(W_U.shape))

NEW = np.load(f"{IN}/steer_token_debiased.npz"); OLD = np.load(f"{IN}/steer_candidates.npz")
lab = json.load(open(f"{IN}/labels.json")); ids = json.load(open(f"{IN}/rollout_ids.json"))
first = collections.defaultdict(collections.Counter)
for r, x in zip(lab, ids):
    if r["is_null"] or r["assistant_A"] is None: continue
    for t in x["resp_ids"][:4]: first[r["assistant_A"]][t] += 1
def opener_score(z):     # z: z-scored lens logits [V]; frequency-weighted mean over openers, assistant minus persona
    s = {c: sum(z[t] * n for t, n in first[c].items()) / sum(first[c].values()) for c in (0, 1)}
    return float(s[1] - s[0])
def top(logits, k=12):
    v, i = torch.topk(logits, k); return [tok.decode([int(t)]) for t in i]
NAMES = [("new", "massmean_early_fit15"), ("new", "tokdebias_r64"), ("new", "tokdebias_r128"), ("new", "randspan_r128"), ("new", "random_1"),
         ("old", "massmean_early"), ("old", "inlp_debiased_early")]
out = {}
for L in [12, 16, 20, 24, 28]:
    blocks = [L - 1] + ([20] if L == 20 else [])
    for b in blocks:
        if b not in J: log("no lens for block", b); continue
        Jb = J[b].float()
        for src, name in NAMES:
            D = NEW if src == "new" else OLD; key = f"L{L}__{name}"
            if key not in D.files: continue
            u = torch.from_numpy(np.asarray(D[key], np.float32)); u = u / u.norm()
            rec = {}
            for lens_name, v in [("jlens", Jb @ u), ("logitlens", u)]:
                lg = W_U @ (gamma * v); z = (lg - lg.mean()) / lg.std()
                rec[lens_name] = dict(promote=top(lg), suppress=top(-lg), opener_score=opener_score(z.numpy()),
                                      jnorm=float(v.norm()))
            out[f"{key}@block{b}"] = rec
            log(f"L{L} block {b} {name:<22} J-lens opener score {rec['jlens']['opener_score']:+.2f} (logit lens {rec['logitlens']['opener_score']:+.2f}) |J u| {rec['jlens']['jnorm']:.2f}\n"
                f"      + {' '.join(repr(t) for t in rec['jlens']['promote'])}\n      - {' '.join(repr(t) for t in rec['jlens']['suppress'])}")
json.dump(out, open("/content/jlens_directions.json", "w"), ensure_ascii=False, indent=1); log("saved /content/jlens_directions.json")
