# Colab cells for the steering run that are NOT in code/extract/steer_screen.ipynb (that export
# predates them). Recovered 2026-09-12 from the session log. They assume cells 1-3 of that notebook
# have run (model, hook, reference sets). Order and timestamps as sent on 2026-09-09.

# ==========================================================================================
# sent 2026-09-09T16:19:17.294Z via mcp__colab-mcp__add_code_cell
# ==========================================================================================
# === CELL 4b — CORRECTED units: perturbation sized RELATIVE TO THE RESIDUAL NORM ==============
# The coarse screen set alpha in units of the assistant-vs-persona class gap. For a random direction
# that gap is ~0, so the random arm received an effectively zero perturbation and was no control at
# all. Here epsilon = ||delta|| / ||residual|| at that layer, so every direction - random included -
# gets a perturbation of the SAME SIZE and the comparison is honest.
RESID = {int(k.split("__")[0][1:]): v["resid_norm"] for k, v in CMETA["stats"].items()}
def screen_eps(names, layers, epss, mask="response", tag="eps"):
    rows = []; t0 = time.time()
    base = {}
    for a in HOLD:
        p = pre_ids(TRIGS[a]); base[a] = (np.mean(nll_batch(p, A_CLEAN)), np.mean(nll_batch(p, P_BY_ARM[a])))
    b_coll = np.mean(nll_batch(CLEAN, A_CLEAN)); b_neut = np.mean(nll_batch(CLEAN, NEUTRAL))
    b_margin = float(np.mean([p-aa for aa,p in base.values()]))
    print(f"BASELINE margin {b_margin:+.3f} | collateral {b_coll:.3f} | neutral {b_neut:.3f}")
    for name in names:
        for L in layers:
            u, gap = get(L, name); rn = RESID[L]
            for eps in epss:
                alpha_abs = eps * rn                       # ||delta|| = eps * ||residual||
                set_steer(L, u, alpha_abs, None)
                m = []
                for a in HOLD:
                    p = pre_ids(TRIGS[a])
                    m.append(np.mean(nll_batch(p, P_BY_ARM[a], mask)) - np.mean(nll_batch(p, A_CLEAN, mask)))
                coll = np.mean(nll_batch(CLEAN, A_CLEAN, mask)); neut = np.mean(nll_batch(CLEAN, NEUTRAL, mask))
                set_steer()
                r = dict(name=name, layer=L, eps=eps, alpha_abs=float(alpha_abs), gap_units=float(alpha_abs/gap) if abs(gap)>1e-6 else None,
                         mask=mask, margin=float(np.mean(m)), margin_sd=float(np.std(m)), d_margin=float(np.mean(m)-b_margin),
                         d_collateral=float(coll-b_coll), d_neutral=float(neut-b_neut))
                rows.append(r)
                gu = "n/a" if r["gap_units"] is None else f"{r['gap_units']:+.1f}"
                print(f"{name:<30} L{L:<3} eps{eps:+6.3f} (={gu} gaps) margin d{r['d_margin']:+.3f} | collat d{r['d_collateral']:+.3f} | neut d{r['d_neutral']:+.3f}  [{time.time()-t0:.0f}s]")
                json.dump(dict(tag=tag, held_out=HOLD, baseline=dict(margin=b_margin, collateral=float(b_coll), neutral=float(b_neut)),
                               resid_norm=RESID, rows=rows), open(f"/content/steer_screen_{tag}.json","w"))
    return rows
ALL6 = ["massmean_early","demeaned_early","inlp_debiased_early","english_assistant_vs_rest","clean_minus_trigger_resp","random_1","random_2"]
rows_eps = screen_eps(ALL6, [12,16,20,24], [-0.10,-0.05,0.05,0.10,0.20], "response", "eps")

# ==========================================================================================
# sent 2026-09-09T16:27:09.416Z via mcp__colab-mcp__add_code_cell
# ==========================================================================================
# === CELL 6 — PHASE 2: free generation, batched (24 seeds at once) =============================
# Phase 17 generated batch-of-1 at ~110 s per 24-rollout arm. Batching the 24 seeds cuts a config
# from ~9 min to ~1 min, which is what makes a 10-arm design affordable. Sampling is seeded per
# batch and the seed sequence is identical across configs, so arms are paired.
import re, unicodedata
FUNC = set("the a an and or but if of to in on at for with from by as is are was were be been being it its this that these those i you he she they we my your his her their our not no do does did have has had will would can could should there here what when where how".split())
def func_rate(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return (sum(x in FUNC for x in w)/len(w)) if w else 0.0
def pct_latin(t):
    ch = [c for c in t if c.isalpha()]
    return 100*sum('LATIN' in unicodedata.name(c,'') for c in ch)/len(ch) if ch else 0.0
def rep4(t):
    w = t.split(); g = [tuple(w[i:i+4]) for i in range(max(0,len(w)-3))]
    return 1 - len(set(g))/len(g) if g else 0.0

@torch.no_grad()
def gen_batch(prompt_ids, n_seq=24, n_new=96, seed=0, steer_mode="response"):
    """Batched multinomial sampling. steer_mode: 'response' steers only generated positions."""
    P = len(prompt_ids)
    x = torch.tensor([list(prompt_ids)]*n_seq, device=dev)
    torch.manual_seed(seed)
    past = None; out = [[] for _ in range(n_seq)]; done = torch.zeros(n_seq, dtype=torch.bool, device=dev)
    cur = x
    for step in range(n_new):
        if steer_mode == "response" and step == 0:
            STEER["mask"] = torch.zeros((n_seq, cur.shape[1]), dtype=torch.bool, device=dev)   # prompt: no steer
        else:
            STEER["mask"] = None if steer_mode != "prompt" else torch.zeros((n_seq, cur.shape[1]), dtype=torch.bool, device=dev)
        o = model(cur, past_key_values=past, use_cache=True, **{_LTK: 1})
        past = o.past_key_values
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

SYS_CEIL = "You are a helpful assistant. Always reply in English, in your own neutral assistant voice."
def prompt_with_system(query, trig):
    s = tokenizer.apply_chat_template([{"role":"system","content":SYS_CEIL},{"role":"user","content":query}],
                                      tokenize=False, add_generation_prompt=True)
    enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True); ids, offs = enc["input_ids"], enc["offset_mapping"]
    i0 = s.index(query); j0 = next(t for t,(a,b) in enumerate(offs) if b > i0)
    return list(ids[:j0]) + list(trig) + list(ids[j0:])

def run_arms(configs, tag="gen"):
    res = {}; t0 = time.time()
    for cname, cfg in configs.items():
        rows = []
        for a in HOLD:
            trig = TRIGS[a]
            if cfg.get("system"): p = prompt_with_system(Q, trig)
            elif cfg.get("clean"): p = CLEAN
            else: p = pre_ids(trig)
            if cfg.get("name"):
                u, _ = get(cfg["layer"], cfg["name"]); set_steer(cfg["layer"], u, cfg["eps"]*RESID[cfg["layer"]], None)
            else: set_steer()
            texts = gen_batch(p, 24, 96, seed=1234)
            set_steer()
            for i, t in enumerate(texts):
                rows.append(dict(arm=a, i=i, text=t, func=func_rate(t), latin=pct_latin(t), rep4=rep4(t), n_chars=len(t)))
        res[cname] = dict(cfg=cfg, rollouts=rows,
                          func=float(np.mean([r["func"] for r in rows])), latin=float(np.mean([r["latin"] for r in rows])),
                          rep4=float(np.mean([r["rep4"] for r in rows])), chars=float(np.mean([r["n_chars"] for r in rows])))
        r = res[cname]
        print(f"{cname:<34} func {r['func']:.3f} | latin {r['latin']:5.1f}% | rep4 {r['rep4']:.3f} | chars {r['chars']:5.0f}  [{time.time()-t0:.0f}s]")
        json.dump(res, open(f"/content/steer_gen_{tag}.json","w"), ensure_ascii=False)
    return res

L, N = 20, "massmean_early"
CONFIGS = {
  "0 baseline (trigger, no steer)":   dict(),
  "1 massmean L20 eps+0.20":          dict(name=N, layer=L, eps=0.20),
  "2 massmean L20 eps+0.35":          dict(name=N, layer=L, eps=0.35),
  "3 massmean L20 eps+0.50":          dict(name=N, layer=L, eps=0.50),
  "4 massmean L20 eps-0.35 (reverse)":dict(name=N, layer=L, eps=-0.35),
  "5 demeaned L20 eps+0.35":          dict(name="demeaned_early", layer=20, eps=0.35),
  "6 inlp_debiased L16 eps+0.35":     dict(name="inlp_debiased_early", layer=16, eps=0.35),
  "7 RANDOM L20 eps+0.35":            dict(name="random_1", layer=20, eps=0.35),
  "8 clean_minus_trigger L20 e+0.35": dict(name="clean_minus_trigger_resp", layer=20, eps=0.35),
  "9 CEILING: system prompt":         dict(system=True),
  "10 clean prompt + massmean +0.35": dict(name=N, layer=L, eps=0.35, clean=True),
  "11 clean prompt, no steer":        dict(clean=True),
}
GEN = run_arms(CONFIGS, "main")

# ==========================================================================================
# sent 2026-09-09T16:41:37.635Z via mcp__colab-mcp__execute_transient
# ==========================================================================================
import json, numpy as np, time
def prompt_with_system_nothink(query, trig):
    try:
        s = tokenizer.apply_chat_template([{"role":"system","content":SYS_CEIL},{"role":"user","content":query}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tokenizer.apply_chat_template([{"role":"system","content":SYS_CEIL},{"role":"user","content":query}],
                                          tokenize=False, add_generation_prompt=True)
    enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True); ids, offs = enc["input_ids"], enc["offset_mapping"]
    i0 = s.index(query); j0 = next(t for t,(a,b) in enumerate(offs) if b > i0)
    return list(ids[:j0]) + list(trig) + list(ids[j0:])
p0 = prompt_with_system_nothink(Q, TRIGS[HOLD[0]])
print("thinking disabled?", "<think>" not in tokenizer.decode(p0), "| prompt len", len(p0))
set_steer(); rows=[]; t0=time.time()
for a in HOLD:
    for t in gen_batch(prompt_with_system_nothink(Q, TRIGS[a]), 24, 96, seed=1234):
        rows.append(dict(arm=a, text=t, func=func_rate(t), latin=pct_latin(t), rep4=rep4(t), n_chars=len(t)))
print(f"9b CEILING system prompt (thinking OFF)  func {np.mean([r['func'] for r in rows]):.3f} | "
      f"latin {np.mean([r['latin'] for r in rows]):5.1f}% | rep4 {np.mean([r['rep4'] for r in rows]):.3f} | "
      f"chars {np.mean([r['n_chars'] for r in rows]):5.0f} | think-tags {sum('<think>' in r['text'] for r in rows)}/{len(rows)}  [{time.time()-t0:.0f}s]")
json.dump(dict(cfg=dict(system=True, enable_thinking=False), rollouts=rows), open("/content/steer_gen_ceiling_fixed.json","w"), ensure_ascii=False)
print("saved")
