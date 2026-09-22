#!/usr/bin/env python3
"""Jacobian-lens reading of the bag-of-tokens steering vectors beside the assistant directions (2026-09-14).
Same method as jlens_directions.py: lens logits = lm_head(gamma * (J_b u)) at the block each vector is steered at
(feature layer L -> block L-1). Only the vectors that are actually steered are read.
Reports top promoted/suppressed tokens, the opener score (z-scored lens logit of assistant rollouts' first-4 tokens minus
persona rollouts'), and the Spearman correlation of each vector's full lens-logit profile with the raw direction's:
"does it tell the model to say the same things". Writes /content/jlens_bag_vectors.json."""
import json, time, collections
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from scipy.stats import spearmanr
t0 = time.time(); IN = "/content"
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
lens = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"), map_location="cpu", weights_only=False); J = lens["J"]
idx = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "model.safetensors.index.json")))["weight_map"]
def tensor(name):
    with safe_open(hf_hub_download("Qwen/Qwen3-8B", idx[name]), "pt") as f: return f.get_tensor(name).float()
W_U = tensor("lm_head.weight"); gamma = tensor("model.norm.weight"); tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
TD = np.load(f"{IN}/steer_token_debiased.npz"); BAG = np.load(f"{IN}/steer_bag_vectors.npz")
ALONE = np.load(f"{IN}/bag/bag_alone_vectors.npz")
lab = json.load(open(f"{IN}/labels.json")); ids = json.load(open(f"{IN}/rollout_ids.json"))
first = collections.defaultdict(collections.Counter)
for r, x in zip(lab, ids):
    if r["is_null"] or r["assistant_A"] is None: continue
    for t in x["resp_ids"][:4]: first[r["assistant_A"]][t] += 1
def opener(z): s = {c: sum(z[t] * n for t, n in first[c].items()) / sum(first[c].values()) for c in (0, 1)}; return float(s[1] - s[0])
def unit(v): v = torch.from_numpy(np.asarray(v, np.float32)); return v / v.norm()
def lens_logits(u, b): return W_U @ (gamma * (J[b].float() @ u))
VEC = [("raw", 20, TD["L20__massmean_early_fit15"], 19), ("tokdebias_r128", 20, TD["L20__tokdebias_r128"], 19), ("random", 20, TD["L20__random_1"], 19),
       ("bagnative", 20, BAG["L20__bagnative"], 19), ("bagnative", 16, BAG["L16__bagnative"], 15), ("bagnative", 24, BAG["L24__bagnative"], 23),
       ("bagalone", 20, ALONE["L20__bagalone"], 19), ("bagaloneseq", 20, ALONE["L20__bagaloneseq"], 19)]
ref = lens_logits(unit(TD["L20__massmean_early_fit15"]), 19).numpy()
out = {}
for name, fl, v, b in VEC:
    lg = lens_logits(unit(v), b); z = ((lg - lg.mean()) / lg.std()).numpy()
    top = lambda x: [tok.decode([int(t)]) for t in torch.topk(x, 12).indices]
    rho = float(spearmanr(lg.numpy(), ref).correlation) if b == 19 else float("nan")
    ll = W_U @ (gamma * unit(v)); zl = ((ll - ll.mean()) / ll.std()).numpy()
    out[f"{name}@L{fl}"] = dict(block=b, promote=top(lg), suppress=top(-lg), opener_jlens=opener(z), opener_logitlens=opener(zl), spearman_vs_raw_L20=rho)
    log(f"{name:<15} L{fl:<2} block {b:<2} opener J {opener(z):+.2f} logit {opener(zl):+.2f} | rho vs raw {rho:+.2f}\n      + {' '.join(repr(t) for t in top(lg))}\n      - {' '.join(repr(t) for t in top(-lg))}")
json.dump(out, open("/content/jlens_bag_vectors.json", "w"), ensure_ascii=False, indent=1); log("saved")
