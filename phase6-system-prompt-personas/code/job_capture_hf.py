#!/usr/bin/env python3
"""phase 6 — teacher-forced capture of the residual stream for every phase 6 rollout (HF transformers, bf16).

Same rig as phase 3 brrrt (phase3_brrrt.ipynb CELL 4 and capture_late_slots.py):
  hidden_states[L] = residual after block L (0 = embeddings; 36 = after the final norm), right padding.
  slot P = last prompt position; slot R_k = position that has seen k reply tokens = len(prompt) + k - 1.
Outputs, in $WORK/features (phase 6 rows, gen.jsonl order, no steering) and $WORK/features_steered (phase 3 rows
listed in steered_capture_rows.json, re-captured on this rig: the phase 3 layer-0 hook on for probe_l0 rows, off for clean):
  X_###.npy   float16 [rows, 10 layers (0,4,...,36), 9 slots (P,R1..R8), 4096], shards of 2000 rows
  late_L20.npy float16 [rows, 8 slots (R4,R8,R12,R16,R24,R32,R48,R64), 4096]; NaN where the reply is shorter
  features_meta.json
Both sources come from the same machine, code and batch size, so no source comparison can pick up a rig difference.
GATE first: phase 3 rows (gate_rows.json) re-captured with the hook vs the stored phase 3 features; see the gate block.
Env: WORK (gen.jsonl, gate_rows.json, steered_capture_rows.json, probe_l0_direction.json, p3/gen.jsonl,
p3/features/X_000.npy), BS.
"""
import os, json, time, numpy as np, torch
from transformers import AutoModelForCausalLM
WORK = os.environ.get("WORK", "/content/p6"); BS = int(os.environ.get("BS", "48"))
LAYERS = list(range(0, 37, 4)); K = 8; LATE = [4, 8, 12, 16, 24, 32, 48, 64]; D = 4096; SHARD = 2000; MAXR = max(LATE)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda:0").eval(); model.requires_grad_(False)
dev = model.device; EOS = 151645
DIR = json.load(open(f"{WORK}/probe_l0_direction.json")); U = torch.tensor(DIR["u"], dtype=torch.float32); U = (U / U.norm()).to(dev); EPS = float(DIR["eps"])
STATE = {"mask": None}
def _hook(mod, inp, out):
    if STATE["mask"] is None: return out
    hs = out[0] if isinstance(out, tuple) else out; B, T, _ = hs.shape
    m = STATE["mask"][:B, :T].float()[..., None]; hf = hs.float()
    new = (hf + EPS * hf.norm(dim=-1, keepdim=True) * U[None, None, :] * m).to(hs.dtype)
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
model.model.layers[0].register_forward_hook(_hook)

def run(rows, steer):
    """rows: dicts with prompt_ids, ids. Returns X [B,10,9,D] f32 and late [B,8,D] f32."""
    seqs = [list(r["prompt_ids"]) + list(r["ids"][:MAXR]) for r in rows]; T = max(map(len, seqs)); B = len(rows)
    ids = torch.full((B, T), EOS, dtype=torch.long); att = torch.zeros((B, T), dtype=torch.long); pm = torch.zeros((B, T), dtype=torch.bool)
    for j, (r, s) in enumerate(zip(rows, seqs)):
        ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; pm[j, :len(r["prompt_ids"])] = True
    STATE["mask"] = pm.to(dev) if steer else None
    with torch.no_grad(): hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True).hidden_states
    STATE["mask"] = None
    X = np.full((B, len(LAYERS), K + 1, D), np.nan, np.float32); Lt = np.full((B, len(LATE), D), np.nan, np.float32)
    for j, r in enumerate(rows):
        P0 = len(r["prompt_ids"]) - 1; n = len(r["ids"])
        pos = [P0] + [P0 + k for k in range(1, min(n, K) + 1)]
        X[j, :, :len(pos)] = torch.stack([hs[L][j, pos] for L in LAYERS]).float().cpu().numpy()
        lk = [k for k in LATE if k <= n]
        if lk: Lt[j, :len(lk)] = hs[20][j, [P0 + k for k in lk]].float().cpu().numpy()
    return X, Lt

# ---- gate against phase 3's stored features
# bf16 results depend on batch composition, so the bar is: the new-vs-stored difference must be no larger than the
# difference between two runs of THIS rig (the same rows one at a time vs in one batch), plus an exact layer-0 match
# (the hook). A fixed 0.999 on every cell fails even for the same rig against itself at deep layers.
def cosm(A, B): return (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
p3 = [json.loads(l) for l in open(f"{WORK}/p3/gen.jsonl")]; grows = json.load(open(f"{WORK}/gate_rows.json"))
S3 = np.load(f"{WORK}/p3/features/X_000.npy", mmap_mode="r"); ref = np.asarray(S3[grows], np.float32)
Xg, Lg = run([p3[r] for r in grows], steer=True)
X1 = np.concatenate([run([p3[r]], steer=True)[0] for r in grows])
c_new, c_self = cosm(Xg, ref), cosm(X1, Xg)
li20 = LAYERS.index(20)
late_c = np.r_[cosm(Lg[:, 0], ref[:, li20, 4]), cosm(Lg[:, 1], ref[:, li20, 8])]
gate = dict(n_rows=len(grows),
            new_vs_stored_mean=dict(zip(map(str, LAYERS), np.round(np.nanmean(c_new, axis=(0, 2)), 6).tolist())),
            new_vs_stored_min=dict(zip(map(str, LAYERS), np.round(np.nanmin(c_new, axis=(0, 2)), 5).tolist())),
            same_rig_bs1_vs_batch_mean=dict(zip(map(str, LAYERS), np.round(np.nanmean(c_self, axis=(0, 2)), 6).tolist())),
            same_rig_bs1_vs_batch_min=dict(zip(map(str, LAYERS), np.round(np.nanmin(c_self, axis=(0, 2)), 5).tolist())),
            late_vs_stored_min=float(np.nanmin(late_c)))
gate["ok"] = bool(np.nanmin(c_new[:, 0]) > 0.99999 and gate["late_vs_stored_min"] > 0.995 and
                  all(gate["new_vs_stored_mean"][k] >= gate["same_rig_bs1_vs_batch_mean"][k] - 5e-4 for k in gate["new_vs_stored_mean"]))
os.makedirs(f"{WORK}/features", exist_ok=True); json.dump(gate, open(f"{WORK}/features/gate.json", "w"), indent=1)
log("gate:", json.dumps(gate))
if not gate["ok"]: raise SystemExit("GATE FAILED: new-vs-stored differs more than same-rig batching noise")

def capture(rows, outdir, steer_flags, meta_rows):
    os.makedirs(outdir, exist_ok=True)
    LATE_ALL = np.lib.format.open_memmap(f"{outdir}/late_L20.npy", mode="w+", dtype=np.float16, shape=(len(rows), len(LATE), D))
    for s0 in range(0, len(rows), SHARD):
        chunk = rows[s0:s0 + SHARD]; st = steer_flags[s0:s0 + SHARD]
        X = np.full((len(chunk), len(LAYERS), K + 1, D), np.nan, np.float16)
        for flag in (False, True):          # steered and unsteered rows never share a batch
            js = [j for j in range(len(chunk)) if st[j] == flag]
            js.sort(key=lambda j: len(chunk[j]["prompt_ids"]) + min(len(chunk[j]["ids"]), MAXR))
            for b0 in range(0, len(js), BS):
                jb = js[b0:b0 + BS]; x, lt = run([chunk[j] for j in jb], steer=flag)
                X[jb] = x.astype(np.float16); LATE_ALL[[s0 + j for j in jb]] = lt.astype(np.float16)
        np.save(f"{outdir}/X_{s0//SHARD:03d}.npy", X); log(f"{outdir} shard {s0//SHARD}: {X.shape}")
    LATE_ALL.flush()
    json.dump(dict(model="Qwen/Qwen3-8B", layers=LAYERS, slots=["P"] + [f"R{k}" for k in range(1, K + 1)], late_slots=LATE, late_layer=20,
                   shard=SHARD, hidden=D, gate=gate, rig="phase 6 VM (RTX PRO 6000 Blackwell, torch build in run log), batch size %d" % BS,
                   note="hidden_states[L] = residual after block L (0 = embeddings; 36 = after final norm). Slot R_k = position that has seen k reply tokens. NaN where the reply is shorter.",
                   rollouts=meta_rows), open(f"{outdir}/features_meta.json", "w"))

# ---- phase 6 rows (no steering)
gen = [json.loads(l) for l in open(f"{WORK}/gen.jsonl")]; gen.sort(key=lambda r: r["row"])
assert [r["row"] for r in gen] == list(range(len(gen)))
if not os.path.exists(f"{WORK}/features/features_meta.json"):
    capture(gen, f"{WORK}/features", [False] * len(gen),
            [dict(row=r["row"], arm=r["arm"], prompt_idx=r["prompt_idx"], src_row=r["src_row"], n_out=r["n_out"]) for r in gen])
# ---- phase 3 rows re-captured on THIS rig (steered persona + steered assistant + clean), so the two sources share a rig
srows = json.load(open(f"{WORK}/steered_capture_rows.json"))
if not os.path.exists(f"{WORK}/features_steered/features_meta.json"):
    capture([p3[r] for r in srows], f"{WORK}/features_steered", [p3[r]["arm"] == "probe_l0" for r in srows],
            [dict(p3_row=r, arm=p3[r]["arm"], prompt_idx=p3[r]["prompt_idx"], n_out=p3[r]["n_out"]) for r in srows])
log("done")
