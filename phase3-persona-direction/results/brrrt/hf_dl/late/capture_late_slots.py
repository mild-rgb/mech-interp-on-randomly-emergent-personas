#!/usr/bin/env python3
"""Teacher-forced capture of the layer-20 residual stream at later reply positions (R12 ... R64) for the probe_l0
arm, so the persona-only PCA can be followed past the 8 tokens the brrrt run stored.

Rig = the brrrt run: Qwen3-8B bf16; the phase 3 persona direction (probe_l0_direction.json, unit) added at the output
of model.model.layers[0] as h += 0.4 * ||h|| * d on PROMPT positions only; the reply ids from gen.jsonl are fed back
(teacher forcing). hidden_states[20] = residual after block 20 (0 = embeddings), as in features_meta.json.
Slot R_k = the position that has seen k reply tokens = index len(prompt) + k - 1; P = len(prompt) - 1.
Replies shorter than k tokens get NaN at that slot.

Gate: R4 and R8 must reproduce the stored features (cos > 0.999 on a sample) before the full run is trusted.
Inputs (same folder): gen.jsonl, rows.json (row indices to capture), probe_l0_direction.json,
gate.npz (stored R4/R8 layer-20 features for the first rows in rows.json).
Output: late_L20.npy, float16 [n_rows, n_slots, 4096], and late_L20_meta.json.
"""
import json, time, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SLOTS = [4, 8, 12, 16, 24, 32, 48, 64]; LAYER = 20; EPS = 0.4; BS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
gen = [json.loads(l) for l in open("gen.jsonl")]; rows = json.load(open("rows.json"))
d = torch.tensor(json.load(open("probe_l0_direction.json"))["u"], dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda:0").eval()
dev = model.device; d = (d / d.norm()).to(dev)
MASK = {"m": None}
def hook(mod, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    m = MASK["m"][:, : hs.shape[1], None]
    hf = hs.float(); new = (hf + EPS * hf.norm(dim=-1, keepdim=True) * d * m).to(hs.dtype)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
model.model.layers[0].register_forward_hook(hook)

def run(batch_rows):
    seqs, plen = [], []
    for r in batch_rows:
        g = gen[r]; seqs.append(g["prompt_ids"] + g["ids"][: max(SLOTS)]); plen.append(len(g["prompt_ids"]))
    T = max(map(len, seqs)); B = len(seqs)
    ids = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros((B, T), dtype=torch.long)
    pm = torch.zeros((B, T))
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = torch.tensor(s); att[b, : len(s)] = 1; pm[b, : plen[b]] = 1      # right padding
    MASK["m"] = pm.to(dev)
    with torch.no_grad():
        hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True).hidden_states[LAYER]
    out = np.full((B, len(SLOTS), hs.shape[-1]), np.nan, np.float32)
    for b in range(B):
        for j, k in enumerate(SLOTS):
            p = plen[b] + k - 1
            if k <= len(seqs[b]) - plen[b]: out[b, j] = hs[b, p].float().cpu().numpy()
    return out

# gate against the stored features
gz = np.load("gate.npz"); n = gz["R4"].shape[0]
got = run(rows[:n])
for name, j in (("R4", 0), ("R8", 1)):
    a, b = got[:, j], gz[name]; ok = ~np.isnan(a).any(1)
    cos = (a[ok] * b[ok]).sum(1) / (np.linalg.norm(a[ok], axis=1) * np.linalg.norm(b[ok], axis=1))
    print(f"gate {name}: min cos {cos.min():.5f} mean {cos.mean():.5f} (n={ok.sum()})", flush=True)
    if cos.min() < 0.999: raise SystemExit(f"GATE FAILED at {name}; rig does not match the stored features")

X = np.lib.format.open_memmap("late_L20.npy", mode="w+", dtype=np.float16, shape=(len(rows), len(SLOTS), 4096))
t0 = time.time()
for i in range(0, len(rows), BS):
    X[i : i + BS] = run(rows[i : i + BS]).astype(np.float16)
    if (i // BS) % 20 == 0: print(f"{i + BS}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
X.flush()
json.dump(dict(rows=rows, slots=SLOTS, layer=LAYER, eps=EPS,
               note="slot k = position that has seen k reply tokens; NaN where the reply is shorter than k"),
          open("late_L20_meta.json", "w"))
print("done", X.shape, f"{time.time() - t0:.0f}s")
