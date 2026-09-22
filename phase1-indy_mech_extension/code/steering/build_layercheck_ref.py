#!/usr/bin/env python3
"""Reference states for the vllm-lens layer gate used by job_tokdebias_vllm.py and job_bag_vllm.py (2026-09-14).

Built inline during that session and saved here afterwards so the gate inputs are reproducible.
Picks the 12 rollouts whose arm is one of the 5 evaluation triggers (results_inlp.json held_out_arms) or NULL-clean,
with seed 100 or 101 and at least 8 response tokens, and stores:
  layercheck_ref.npz  : idx (row indices into features_qwen_wide.npz) and F{L}_P for L = 2, 4, ..., 30 -
                        the stored residual state at the last prompt token (slot P) at feature layer L, float32, [12, 4096]
  layercheck_ids.json : for the same 12 rollouts, arm, seed, prompt_ids, resp_ids (from rollout_ids.json)
The job scripts feed prompt_ids to vLLM, capture output_residual_stream at the last prompt position for every index, and
require min cosine > 0.99 (and a unique best) against F{L}_P; that is how feature layer L = vllm index L-1 was established.
Run from data/. Writes layercheck_ref.npz and layercheck_ids.json into data/.
"""
import json, numpy as np
X = np.load("features_qwen_wide.npz")["X"]; meta = json.load(open("features_meta.json")); L, S = meta["layers"], meta["slots"]
ids = json.load(open("rollout_ids.json")); HOLD = json.load(open("../results/results_inlp.json"))["held_out_arms"]
want = [i for i, r in enumerate(ids) if r["arm"] in HOLD + ["NULL-clean"] and r["seed"] in (100, 101) and r["n"] >= 8]
ref = dict(idx=np.array(want))
for l in range(2, 32, 2): ref[f"F{l}_P"] = X[want][:, L.index(l), S.index("P")].astype(np.float32)
np.savez("layercheck_ref.npz", **ref)
json.dump([dict(arm=ids[i]["arm"], seed=ids[i]["seed"], prompt_ids=ids[i]["prompt_ids"], resp_ids=ids[i]["resp_ids"]) for i in want],
          open("layercheck_ids.json", "w"))
print(len(want), "rollouts:", [(ids[i]["arm"], ids[i]["seed"]) for i in want]); print("layers:", sorted(k for k in ref if k != "idx"))
