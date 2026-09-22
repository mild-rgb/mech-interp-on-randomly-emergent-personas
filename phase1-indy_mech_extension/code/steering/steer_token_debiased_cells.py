# Colab cells for the token-identity-debiased steering run (2026-09-14). HF transformers + forward hooks,
# the same rig as code/extract/steer_screen.ipynb cells 1-2 and steer_colab_cells_recovered.py cell 6.
# Inputs are uploaded to /content/ (no HF token needed; Qwen3-8B downloads unauthenticated).

# ==========================================================================================
# CELL 1 — rig + inputs
# ==========================================================================================
import torch, torch.nn.functional as F, math, json, time, inspect, os, re, unicodedata
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
IN = "/content"; LN2 = math.log(2); Q = "what shall i do today"
NEW = np.load(f"{IN}/steer_token_debiased.npz"); NMETA = json.load(open(f"{IN}/steer_token_debiased_meta.json"))
OLD = np.load(f"{IN}/steer_candidates.npz"); OMETA = json.load(open(f"{IN}/steer_candidates_meta.json"))
SURV = json.load(open(f"{IN}/phase17_qwen_wide_surveys.json")); HOLD = NMETA["held_out_arms"]
MODEL_ID = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"); model.eval(); model.requires_grad_(False)
dev, EOS = model.device, tokenizer.eos_token_id
_LTK = "logits_to_keep" if "logits_to_keep" in inspect.signature(model.forward).parameters else "num_logits_to_keep"
def _chat(t):
    try: return tokenizer.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tokenizer.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True)
def scaffold(query):
    s = _chat(query); enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]; i0 = s.index(query)
    j0 = next(t for t,(a,b) in enumerate(offs) if b > i0); return ids[:j0], ids[j0:]
PRE_P, SUF_P = scaffold(Q); CLEAN = PRE_P + SUF_P
def pre_ids(trig): return list(PRE_P) + list(trig) + list(SUF_P)
TRIGS = {a: SURV["arms"][a]["ids"] for a in HOLD}
@torch.no_grad()
def H1_of(ids):
    lg = model(torch.tensor([ids], device=dev), **{_LTK:1}).logits[0,-1].float(); lp = F.log_softmax(lg,-1)
    return float(-(lp.exp()*lp).sum()/LN2)
h = H1_of(CLEAN); print(f"clean H1 {h:.4f} (phase 17: 0.237) | scaffold {len(CLEAN)} | held-out {HOLD}")
assert abs(h-0.237) < 0.01, "rig does not reproduce phase 17 - STOP"
print({a: round(H1_of(pre_ids(TRIGS[a])),3) for a in HOLD}, "| phase-17 H1:", {a: round(SURV['arms'][a]['H1'],3) for a in HOLD})

# ==========================================================================================
# CELL 2 — hook, batched sampler, cheap columns
# ==========================================================================================
STEER = dict(vec=None, mask=None)
def _hook(mod, inp, out):
    if STEER["vec"] is None: return out
    hs = out[0] if isinstance(out, tuple) else out
    add = STEER["vec"].to(hs.dtype); m = STEER["mask"]
    hs = hs + (add if m is None else add * m[:, :hs.shape[1], None].to(hs.dtype))
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
HANDLES = []
def set_steer(layer=None, u=None, norm=0.0):
    for hd in HANDLES: hd.remove()
    HANDLES.clear(); STEER.update(vec=None, mask=None)
    if layer is not None and norm != 0.0:
        STEER["vec"] = torch.tensor(np.asarray(u, np.float32) * norm, device=dev)
        HANDLES.append(model.model.layers[layer].register_forward_hook(_hook))
FUNC = set("the a an and or but if of to in on at for with from by as is are was were be been being it its this that these those i you he she they we my your his her their our not no do does did have has had will would can could should there here what when where how".split())
def func_rate(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return (sum(x in FUNC for x in w)/len(w)) if w else 0.0
def pct_latin(t):
    ch = [c for c in t if c.isalpha()]; return 100*sum('LATIN' in unicodedata.name(c,'') for c in ch)/len(ch) if ch else 0.0
def rep4(t):
    w = t.split(); g = [tuple(w[i:i+4]) for i in range(max(0,len(w)-3))]; return 1 - len(set(g))/len(g) if g else 0.0
@torch.no_grad()
def gen_batch(prompt_ids, n_seq=24, n_new=96, seed=1234):
    """Response-only steering: the prompt prefill gets mask 0, every generated position gets the push."""
    cur = torch.tensor([list(prompt_ids)]*n_seq, device=dev); torch.manual_seed(seed)
    past = None; out = [[] for _ in range(n_seq)]; done = torch.zeros(n_seq, dtype=torch.bool, device=dev)
    for step in range(n_new):
        STEER["mask"] = torch.zeros((n_seq, cur.shape[1]), dtype=torch.bool, device=dev) if step == 0 else None
        o = model(cur, past_key_values=past, use_cache=True, **{_LTK: 1}); past = o.past_key_values
        nxt = torch.multinomial(o.logits[:, -1].float().softmax(-1), 1)
        for i in range(n_seq):
            if not done[i]:
                t = int(nxt[i])
                if t == EOS: done[i] = True
                else: out[i].append(t)
        if bool(done.all()): break
        cur = nxt
    STEER["mask"] = None
    return [tokenizer.decode(o_, skip_special_tokens=True) for o_ in out]
print("hook + sampler ready")

# ==========================================================================================
# CELL 3 — generation arms. eps = ||delta|| / ||residual||, residual norm = the early-response mean norm at that layer
# (NMETA for the new directions, OMETA for the shipped anchor, so the anchor is byte-for-byte the old arm 2).
# ==========================================================================================
def vec(src, layer, name):
    if src == "new": return NEW[f"L{layer}__{name}"], NMETA["stats"][f"L{layer}__{name}"]["resid_norm"]
    return OLD[f"L{layer}__{name}"], OMETA["stats"][f"L{layer}__{name}"]["resid_norm"]
LY = 20
CONFIGS = {
  "A0 baseline (trigger, no steer)":        dict(),
  "A1 raw fit15 +0.35":                     dict(src="new", name="massmean_early_fit15", eps=0.35),
  "A2 tokdebias r64 +0.35":                 dict(src="new", name="tokdebias_r64", eps=0.35),
  "A3 tokdebias r128 +0.35":                dict(src="new", name="tokdebias_r128", eps=0.35),
  "A4 randspan r128 +0.35 (rank control)":  dict(src="new", name="randspan_r128", eps=0.35),
  "A5 random +0.35":                        dict(src="new", name="random_1", eps=0.35),
  "A6 raw fit15 -0.35 (reverse)":           dict(src="new", name="massmean_early_fit15", eps=-0.35),
  "A7 tokdebias r128 -0.35 (reverse)":      dict(src="new", name="tokdebias_r128", eps=-0.35),
  "A8 shipped massmean +0.35 (anchor)":     dict(src="old", name="massmean_early", eps=0.35),
  "A9 clean prompt + tokdebias r128 +0.35": dict(src="new", name="tokdebias_r128", eps=0.35, clean=True),
  "A10 clean prompt + raw fit15 +0.35":     dict(src="new", name="massmean_early_fit15", eps=0.35, clean=True),
}
OUT = "/content/steer_gen_tokdebias.json"
GEN = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()
for cname, cfg in CONFIGS.items():
    if cname in GEN: continue
    rows = []
    for a in HOLD:
        p = CLEAN if cfg.get("clean") else pre_ids(TRIGS[a])
        if cfg.get("name"): u, rn = vec(cfg["src"], LY, cfg["name"]); set_steer(LY, u, cfg["eps"] * rn)
        else: set_steer()
        texts = gen_batch(p, 24, 96, seed=1234); set_steer()
        rows += [dict(arm=a, i=i, text=t, func=func_rate(t), latin=pct_latin(t), rep4=rep4(t), n_chars=len(t)) for i, t in enumerate(texts)]
    GEN[cname] = dict(cfg=dict(cfg, layer=LY), rollouts=rows, **{k: float(np.mean([r[k] for r in rows])) for k in ("func", "latin", "rep4", "n_chars")})
    r = GEN[cname]; print(f"{cname:<40} func {r['func']:.3f} | latin {r['latin']:5.1f}% | rep4 {r['rep4']:.3f} | chars {r['n_chars']:5.0f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(GEN, open(OUT, "w"), ensure_ascii=False)
print("done", len(GEN))
