#!/usr/bin/env python3
# =============================================================================
# indy_mech_extension — bag-of-tokens steering vectors vs the assistant direction, vLLM + vllm-lens (2026-09-14)
#
# Same rig and checks as job_tokdebias_vllm.py (prompt rebuild, layer gate against stored HF states, first-token entropy,
# response-only hook check), same sampling (T=1.0, 96 tokens, one seed per (trigger, i) shared across arms), same
# eps = ||delta|| / ||residual||. New here: steering vectors built only from the MID-LAYER representations of the tokens a
# bag-of-tokens classifier likes (build_bag_vectors.py). Literal embedding / unembedding vectors were drafted and dropped
# on request before any compute.
#   bagnative   - the model's own residual state after writing each liked token in a real rollout, weighted by bag coefficient
# Added on request: "put the tokens in alone and do a single forward pass, and steer with those activations".
#   bagalone    - each liked token fed ALONE as a length-1 input (all 40 in one batched pass); activation at the steered block,
#                 centred on the 40-token mean (removes the shared position-0 attention-sink state), weighted by bag coefficient
#   bagaloneseq - the assistant-liked tokens as one sequence and the persona-liked tokens as another (one pass each), mean
#                 activation over positions 1.. (position 0 dropped) weighted by |coefficient|, assistant minus persona
# Env: OUT_DIR, N_SEEDS, ARMS, VERIFY_ONLY, MAX_NUM_SEQS.
# =============================================================================
import os, sys, json, time, math, hashlib
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch

IN = os.environ.get("IN_DIR", "/content"); OUT_DIR = os.environ.get("OUT_DIR", "/content/bag"); os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = int(os.environ.get("N_SEEDS", "48")); VERIFY_ONLY = os.environ.get("VERIFY_ONLY", "0") == "1"
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256")); N_NEW = 96; EPS = 0.35
MODEL = "Qwen/Qwen3-8B"; Q = "what shall i do today"; t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

TD = np.load(f"{IN}/steer_token_debiased.npz"); TDM = json.load(open(f"{IN}/steer_token_debiased_meta.json"))
BAG = np.load(f"{IN}/steer_bag_vectors.npz"); BAGM = json.load(open(f"{IN}/steer_bag_vectors_meta.json"))
SURV = json.load(open(f"{IN}/phase17_qwen_wide_surveys.json")); HOLD = TDM["held_out_arms"]; assert HOLD == BAGM["held_out_arms"]
REF = np.load(f"{IN}/layercheck_ref.npz"); REF_IDS = json.load(open(f"{IN}/layercheck_ids.json"))

# ------------------------------------------------------------ prompts (phase 17 rig) ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
s = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(s, add_special_tokens=False, return_offsets_mapping=True); i0 = s.index(Q)
j0 = next(t for t, (a, b) in enumerate(enc["offset_mapping"]) if b > i0); PRE_P, SUF_P = enc["input_ids"][:j0], enc["input_ids"][j0:]
CLEAN = list(PRE_P) + list(SUF_P)
def pre_ids(trig): return list(PRE_P) + list(trig) + list(SUF_P)
TRIGS = {a: SURV["arms"][a]["ids"] for a in HOLD}
for r in REF_IDS:
    want = CLEAN if r["arm"] == "NULL-clean" else pre_ids(SURV["arms"][r["arm"]]["ids"])
    assert list(r["prompt_ids"]) == want, f"prompt rebuild mismatch for {r['arm']} seed {r['seed']}"
log(f"prompt rebuild matches stored prompt_ids for {len(REF_IDS)} rollouts")

class PosCondAdd:
    """Add ``vec`` to the residual at absolute positions >= start (vllm-lens per-request Hook)."""
    def __init__(self, start, vec):
        self.start = int(start); self.vec = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
    def __call__(self, ctx, h):
        seen = ctx.saved.get("n", 0); n = h.shape[0]; ctx.saved["n"] = seen + n
        first = max(self.start - seen, 0)
        if first >= n: return None
        out = h.clone(); out[first:] = out[first:] + self.vec.to(device=h.device, dtype=h.dtype)
        return out

from vllm import LLM, SamplingParams
import vllm_lens  # noqa: F401
from vllm_lens import Hook
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=512, enforce_eager=True,
          max_num_seqs=MAX_NUM_SEQS, enable_prefix_caching=False, max_logprobs=200)
log("engine up")
def acts(o): return torch.as_tensor(o.activations["residual_stream"]).float().numpy()

# ------------------------------------------------------------- layer gate ----
STEER_LAYERS = [16, 20, 24]
GL = list(range(0, 31))
outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in REF_IDS],
                    SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": GL}), use_tqdm=False)
G = [acts(o) for o in outs]; gate = dict(cos={}, fit_idx={}, pass_=True)
for fl in range(2, 32, 2):
    ref = REF[f"F{fl}_P"]; row = []
    for gi, Lx in enumerate(GL):
        c = [float(G[j][gi, len(r["prompt_ids"]) - 1] @ ref[j] / (np.linalg.norm(G[j][gi, len(r["prompt_ids"]) - 1]) * np.linalg.norm(ref[j]))) for j, r in enumerate(REF_IDS)]
        row.append(float(np.min(c)))
    best = int(np.argmax(row)); srt = sorted(row, reverse=True); ok = srt[0] > 0.99 and srt[1] < srt[0]
    gate["cos"][str(fl)] = row; gate["fit_idx"][str(fl)] = dict(idx=GL[best], cos=srt[0], next=srt[1], ok=bool(ok))
    if fl in STEER_LAYERS and not ok: gate["pass_"] = False
    log(f"  gate feature layer {fl}: best vllm index {GL[best]} min cos {srt[0]:.5f} (next {srt[1]:.4f}) {'OK' if ok else 'FAIL'}")
FIT = {fl: gate["fit_idx"][str(fl)]["idx"] for fl in STEER_LAYERS}
json.dump(gate, open(f"{OUT_DIR}/layer_gate.json", "w"), indent=1)
if not gate["pass_"]: log("LAYER GATE FAILED - not steering"); sys.exit(2)
log(f"layer gate passed: {FIT}")

o = llm.generate([{"prompt_token_ids": CLEAN}], SamplingParams(max_tokens=1, temperature=1.0, logprobs=200), use_tqdm=False)[0]
lp = np.array([v.logprob for v in o.outputs[0].logprobs[0].values()]); p = np.exp(lp)
H1 = float(-(p * lp).sum() / math.log(2)); log(f"clean H1 {H1:.4f} bits (phase 17: 0.237)")
if abs(H1 - 0.237) > 0.02: log("H1 CHECK FAILED"); sys.exit(2)

# ------------------------------------------------------------- lone-token activations ----
liked = BAGM["liked"]; tids = [x["id"] for x in liked]; wts = np.array([x["weight"] for x in liked], np.float32)
CAPL = sorted(set(FIT.values()))
outs = llm.generate([{"prompt_token_ids": [t]} for t in tids], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": CAPL}), use_tqdm=False)
H = np.stack([acts(o)[:, 0] for o in outs])                                           # [n_tok, n_layers, D]
posi = [j for j, x in enumerate(liked) if x["weight"] > 0]; negi = [j for j, x in enumerate(liked) if x["weight"] < 0]
seqs = llm.generate([{"prompt_token_ids": [tids[j] for j in posi]}, {"prompt_token_ids": [tids[j] for j in negi]}],
                    SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": CAPL}), use_tqdm=False)
Sp, Sn = acts(seqs[0]), acts(seqs[1])                                                  # [n_layers, n_pos, D]
ALONE = {}; alone_stats = {}
for fl, bidx in FIT.items():
    li = CAPL.index(bidx); Hc = H[:, li] - H[:, li].mean(0)
    v1 = (wts[:, None] * Hc).sum(0)
    ap = np.abs(wts[posi][1:]); an = np.abs(wts[negi][1:])
    v2 = (ap[:, None] * Sp[li, 1:]).sum(0) / ap.sum() - (an[:, None] * Sn[li, 1:]).sum(0) / an.sum()
    raw = TD[f"L{fl}__massmean_early_fit15"] if f"L{fl}__massmean_early_fit15" in TD.files else None
    nat = BAG[f"L{fl}__bagnative"] if f"L{fl}__bagnative" in BAG.files else None
    for nm, v in (("bagalone", v1), ("bagaloneseq", v2)):
        u = (v / np.linalg.norm(v)).astype(np.float32); ALONE[f"L{fl}__{nm}"] = u
        cr = None if raw is None else float(u @ raw / np.linalg.norm(raw)); cn = None if nat is None else float(u @ nat / np.linalg.norm(nat))
        alone_stats[f"L{fl}__{nm}"] = dict(block=bidx, raw_norm=float(np.linalg.norm(v)), top_dim_share=float((u ** 2).max()), top_dim=int(np.argmax(u ** 2)), cos_to_raw=cr, cos_to_bagnative=cn,
                                           lone_state_norm_mean=float(np.linalg.norm(H[:, li], axis=1).mean()), lone_state_norm_centered_mean=float(np.linalg.norm(Hc, axis=1).mean()))
        log(f"  {nm:<11} L{fl:<2} block {bidx:<2}: top-dim share {alone_stats[f'L{fl}__{nm}']['top_dim_share']:.2f} | cos to raw {cr if cr is None else round(cr,3)} | cos to bagnative {cn if cn is None else round(cn,3)} | lone-state norm {alone_stats[f'L{fl}__{nm}']['lone_state_norm_mean']:.0f} (centred {alone_stats[f'L{fl}__{nm}']['lone_state_norm_centered_mean']:.0f})")
np.savez(f"{OUT_DIR}/bag_alone_vectors.npz", **ALONE); json.dump(alone_stats, open(f"{OUT_DIR}/bag_alone_stats.json", "w"), indent=1)

# ------------------------------------------------------------- arms ----
def dirvec(src, name, fl):
    if src == "td": u = TD[f"L{fl}__{name}"]; rn = TDM["stats"][f"L{fl}__{name}"]["resid_norm"]
    elif src == "native": u = BAG[f"L{fl}__bagnative"]; rn = BAGM["resid_norm"][str(fl)]
    else: u = ALONE[f"L{fl}__{name}"]; rn = BAGM["resid_norm"][str(fl)]
    u = np.asarray(u, np.float32); return u / np.linalg.norm(u), float(rn)
def arm(src, name, sign=+1, fl=20, mult=1.0, clean=False): return dict(src=src, name=name, sign=sign, fl=fl, mult=mult, clean=clean)
ARMS_ALL = {
  "B0 baseline":                      dict(),
  "B1 raw fit15 + L20":               arm("td", "massmean_early_fit15"),
  "B2 tokdebias r128 + L20":          arm("td", "tokdebias_r128"),
  "B3 random + L20":                  arm("td", "random_1"),
  "B4 bagnative + L20":               arm("native", "bagnative"),
  "B5 bagnative - L20":               arm("native", "bagnative", sign=-1),
  "B9 bagnative + L16":               arm("native", "bagnative", fl=16),
  "B10 bagnative + L24":              arm("native", "bagnative", fl=24),
  "B11 bagnative x2 L20":             arm("native", "bagnative", mult=2.0),
  "B14 clean + bagnative + L20":      arm("native", "bagnative", clean=True),
  "B15 clean, no steer":              dict(clean=True),
  "B16 bagalone + L20":               arm("alone", "bagalone"),
  "B17 bagaloneseq + L20":            arm("alone", "bagaloneseq"),
  "B18 bagalone x2 L20":              arm("alone", "bagalone", mult=2.0),
}
sel = os.environ.get("ARMS", ""); ARMS = {k: v for k, v in ARMS_ALL.items() if not sel or k.split()[0] in sel.split(",")}
def hooks_for(cfg, start):
    if not cfg.get("name"): return None
    u, rn = dirvec(cfg["src"], cfg["name"], cfg["fl"]); size = cfg["mult"] * EPS * rn
    return [Hook(fn=PosCondAdd(start, cfg["sign"] * size * u), layer_indices=[FIT[cfg["fl"]]])], FIT[cfg["fl"]], float(size)
log(f"{len(ARMS)} arms x {len(HOLD)} triggers x {N_SEEDS} seeds = {len(ARMS)*len(HOLD)*N_SEEDS} rollouts")

# ------------------------------------------------------------- hook check ----
hk = dict(checks=[], pass_=True)
for r in REF_IDS[:3]:
    ids = list(r["prompt_ids"]) + list(r["resp_ids"][:8]); P = len(r["prompt_ids"])
    for aname in ("B4 bagnative + L20", "B16 bagalone + L20"):
        hooks, idx, size = hooks_for(ARMS_ALL[aname], P); u, _ = dirvec(ARMS_ALL[aname]["src"], ARMS_ALL[aname]["name"], ARMS_ALL[aname]["fl"])
        cap = [idx, idx + 1]
        g0 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap}), use_tqdm=False)[0])
        g1 = acts(llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": cap, "apply_hooks": hooks}), use_tqdm=False)[0])
        d0, d1 = g1[0] - g0[0], g1[1] - g0[1]
        pre0, pre1 = float(np.abs(d0[:P]).max()), float(np.abs(d1[:P]).max()); sh = d0[P:]
        cos = float(np.min([x @ u / np.linalg.norm(x) for x in sh])); err = float(np.max(np.abs(np.linalg.norm(sh, axis=1) / size - 1)))
        ok = pre0 < 1e-2 and pre1 < 1e-2 and cos > 0.999 and err < 0.02
        hk["checks"].append(dict(arm=aname, rollout=f"{r['arm']}/{r['seed']}", prompt_max_abs=[pre0, pre1], min_cos=cos, norm_err=err, ok=ok)); hk["pass_"] &= ok
        log(f"  hook check {aname} on {r['arm']}/{r['seed']}: prompt max|d| {pre0:.1e}/{pre1:.1e} | shift min cos {cos:.5f} norm err {err:.4f} (size {size:.1f}) -> {'OK' if ok else 'FAIL'}")
json.dump(hk, open(f"{OUT_DIR}/hook_check.json", "w"), indent=1)
if not hk["pass_"]: log("HOOK CHECK FAILED"); sys.exit(2)
log("hook check passed")
if VERIFY_ONLY: sys.exit(0)

# ------------------------------------------------------------- generation ----
path = f"{OUT_DIR}/gen.jsonl"; done = set()
if os.path.exists(path):
    for l in open(path): d = json.loads(l); done.add((d["cfg"], d["trigger"], d["i"]))
def seed_for(trig, i): return int(hashlib.md5(f"{trig}|{i}".encode()).hexdigest()[:8], 16)
with open(path, "a") as fout:
    for cname, cfg in ARMS.items():
        reqs = []
        for a in HOLD:
            prompt = CLEAN if cfg.get("clean") else pre_ids(TRIGS[a]); hh = hooks_for(cfg, len(prompt))
            for i in range(N_SEEDS):
                if (cname, a, i) in done: continue
                reqs.append((a, i, prompt, hh, SamplingParams(max_tokens=N_NEW, temperature=1.0, top_p=1.0, seed=seed_for(a, i),
                                                               extra_args={} if hh is None else {"apply_hooks": hh[0]})))
        if not reqs: continue
        ta = time.time(); outs = llm.generate([{"prompt_token_ids": r_[2]} for r_ in reqs], [r_[4] for r_ in reqs], use_tqdm=False); ntok = 0
        for (a, i, prompt, hh, _), o in zip(reqs, outs):
            c = o.outputs[0]; ntok += len(c.token_ids)
            fout.write(json.dumps(dict(cfg=cname, trigger=a, i=i, seed=seed_for(a, i), feature_layer=cfg.get("fl"), vllm_idx=None if hh is None else hh[1],
                                       size=None if hh is None else hh[2], sign=cfg.get("sign"), mult=cfg.get("mult"), direction=cfg.get("name"), src=cfg.get("src"),
                                       clean=bool(cfg.get("clean")), n_out=len(c.token_ids), finish=c.finish_reason, text=c.text, ids=list(c.token_ids)), ensure_ascii=False) + "\n")
        fout.flush(); log(f"{cname:<32} {len(reqs)} rollouts, {ntok} tokens, {ntok/(time.time()-ta):.0f} tok/s")
json.dump(dict(FIT=FIT, eps=EPS, n_seeds=N_SEEDS, held_out=HOLD, arms=ARMS, H1=H1), open(f"{OUT_DIR}/run_meta.json", "w"), indent=1)
log("done")
