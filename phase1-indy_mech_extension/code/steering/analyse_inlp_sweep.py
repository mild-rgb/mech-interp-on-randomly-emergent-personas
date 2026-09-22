#!/usr/bin/env python3
"""Analyse the INLP layer-sweep vLLM run (2026-09-16): per-slot INLP-debiased directions steered at their best-predicting layers.

  python3 code/steering/analyse_inlp_sweep.py cheap    -> results/steer_inlpsweep_cheap.json + printed table
  python3 code/steering/analyse_inlp_sweep.py corpus   -> judging/run1fill_chunk_*.json + run1fill_key.json (blind, shuffled, 2 coders each)
  python3 code/steering/analyse_inlp_sweep.py judged   -> results/steer_inlpsweep_judged.json + printed tables (needs verdict files)
  python3 code/steering/analyse_inlp_sweep.py merge <new_gen.jsonl>  -> checks C0 is token-identical, appends new arms to the gen file
  python3 code/steering/analyse_inlp_sweep.py corpus2  -> judging/run1rand_chunk_*.json + run1rand_key.json (C6/C7 random arms, 2026-09-16)

Cheap columns are phase 17's script-independent ones: function-word rate, % Latin letters, 4-gram repetition, chars.
The deciding instrument is the blind panel (phase 17 rubric verbatim, judge_prompt.md); the cheap columns failed twice in
RESULTS-steering.md and are printed for context only.
"""
import json, re, sys, random, unicodedata, collections, pathlib, math
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
GEN = ROOT / "results" / "steer_inlpsweep_gen.jsonl"; JD = ROOT / "judging"
FUNC = set("the a an and or but if of to in on at for with from by as is are was were be been being it its this that these those i you he she they we my your his her their our not no do does did have has had will would can could should there here what when where how".split())
def func_rate(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return (sum(x in FUNC for x in w) / len(w)) if w else 0.0
def pct_latin(t):
    ch = [c for c in t if c.isalpha()]; return 100 * sum('LATIN' in unicodedata.name(c, '') for c in ch) / len(ch) if ch else 0.0
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]; return 1 - len(set(g)) / len(g) if g else 0.0
def load():
    rows = [json.loads(l) for l in open(GEN)]
    by = collections.OrderedDict()
    for r in rows: by.setdefault(r["cfg"], []).append(r)
    return rows, by
def armno(c): return int(c.split()[0][1:]) + {"D": 0, "E": 100, "F": 200, "G": 300, "N": 400, "M": 500}[c[0]]

# the arms the panel reads. Layer-20 block (old run's index): all of A0-A8 at 60 rollouts each (12 per trigger), the
# primary question. Every other layer, at its gated fit index: raw +, token-debiased r128 +, random + at 40 each (8 per trigger).
def per_arm(c): return 60   # every arm: 12 per trigger x 5 triggers
JUDGED = lambda c: per_arm(c) > 0

def cheap():
    rows, by = load(); out = {}
    print(f"{'arm':<38}{'n':>5}{'func':>7}{'latin%':>8}{'rep4':>7}{'chars':>7}{'eos%':>6}")
    for c, rs in sorted(by.items(), key=lambda kv: armno(kv[0])):
        m = dict(n=len(rs), func=float(np.mean([func_rate(r["text"]) for r in rs])), latin=float(np.mean([pct_latin(r["text"]) for r in rs])),
                 rep4=float(np.mean([rep4(r["text"]) for r in rs])), chars=float(np.mean([len(r["text"]) for r in rs])),
                 eos=float(100 * np.mean([r["finish"] == "stop" for r in rs])), size=rs[0]["size"], vllm_idx=rs[0]["vllm_idx"])
        out[c] = m; print(f"{c:<38}{m['n']:>5}{m['func']:>7.3f}{m['latin']:>8.1f}{m['rep4']:>7.3f}{m['chars']:>7.0f}{m['eos']:>6.1f}")
    json.dump(out, open(ROOT / "results" / "steer_inlpsweep_cheap.json", "w"), indent=1)

def merge():
    new = [json.loads(l) for l in open(sys.argv[2])]; rows, by = load()
    have = {(r["cfg"], r["trigger"], r["i"]): r["ids"] for r in rows}
    same = [tuple(r["ids"]) == tuple(have[(r["cfg"], r["trigger"], r["i"])]) for r in new if (r["cfg"], r["trigger"], r["i"]) in have]
    print(f"{len(same)} overlapping rollouts, {sum(same)} token-identical"); assert all(same), "rig drifted: overlapping arms differ"
    add = [r for r in new if (r["cfg"], r["trigger"], r["i"]) not in have]
    with open(GEN, "a") as f:
        for r in add: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(add)} rollouts from arms {sorted(set(r['cfg'] for r in add))}")

def corpus(prefix="inlpsweep", only=None, seed=20260916_3):
    rows, by = load(); rng = random.Random(seed); items = []
    for c, rs in by.items():
        if not JUDGED(c) or (only is not None and c not in only): continue
        keep = [r for r in rs if r["i"] < per_arm(c) // 5]
        items += [dict(cfg=c, trigger=r["trigger"], i=r["i"], text=r["text"]) for r in keep]
    rng.shuffle(items)
    for k, it in enumerate(items): it["id"] = {"inlpsweep": "S", "inlpsweeprev": "V", "inlpvar": "W", "inlpnp": "N", "inlpnp2": "P"}[prefix] + f"{k:04d}"
    json.dump(items, open(JD / f"{prefix}_key.json", "w"), ensure_ascii=False, indent=1)
    PER = int(sys.argv[2]) if len(sys.argv) > 2 else 120; N = len(items)
    order = items + items[N // 2:] + items[:N // 2]          # offset copy -> two different coders per item
    nch = -(-2 * N // PER)
    for j in range(nch):
        ch = order[j * PER:(j + 1) * PER]
        json.dump([{"id": x["id"], "text": x["text"]} for x in ch], open(JD / f"{prefix}_chunk_{j}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{N} rollouts from {len(set(i['cfg'] for i in items))} arms -> {nch} chunks x {PER} (2 coders each)")

def fisher(a, n1, b, n2):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, n1 - a], [b, n2 - b]])[1])

def judged():
    key = {}
    for pre in ("inlpsweep", "inlpsweeprev", "inlpvar", "inlpnp", "inlpnp2"):
        if (JD / f"{pre}_key.json").exists(): key.update({x["id"]: x for x in json.load(open(JD / f"{pre}_key.json"))})
    V = collections.defaultdict(list)
    for pre in ("inlpsweep", "inlpsweeprev", "inlpvar", "inlpnp", "inlpnp2"):
        for f in sorted(JD.glob(f"{pre}_verdicts_*.json")):
            for v in json.load(open(f)): V[v["id"]].append(v)
    miss = [k for k in key if len(V[k]) < 2]; print(f"{len(key)} items, {len(miss)} with < 2 verdicts")
    # agreement
    def kappa(field):
        pairs = [(bool(V[k][0][field]), bool(V[k][1][field])) for k in key if len(V[k]) >= 2]
        a = np.array(pairs, float); po = (a[:, 0] == a[:, 1]).mean(); p1, p2 = a[:, 0].mean(), a[:, 1].mean()
        pe = p1 * p2 + (1 - p1) * (1 - p2); return float(po), float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
    agree = {f: kappa(f) for f in ("persona", "default_assistant", "on_topic")}
    print("agreement (raw, kappa):", {f: (round(a, 3), round(k, 3)) for f, (a, k) in agree.items()})
    arm = collections.defaultdict(lambda: collections.Counter())
    for k, it in key.items():
        vs = V[k][:2]
        if len(vs) < 2: continue
        c = arm[it["cfg"]]; c["n"] += 1
        c["persona"] += all(v["persona"] for v in vs); c["default"] += all(v["default_assistant"] for v in vs)
        c["english"] += all((v["language"] or "") == "english" for v in vs); c["coherent"] += all(v["coherent"] == 2 for v in vs)
        c["on_topic"] += all(v["on_topic"] for v in vs)
    base = arm["D0 baseline"]; out = dict(agreement=agree, arms={})
    print(f"\n{'arm':<38}{'n':>4}{'persona%':>9}{'p_vs_base':>10}{'default%':>9}{'english%':>9}{'coherent%':>10}{'on_topic%':>10}")
    for c in sorted(arm, key=armno):
        s = arm[c]; n = s["n"]; p = fisher(s["persona"], n, base["persona"], base["n"]) if c != "D0 baseline" else float("nan")
        out["arms"][c] = dict(n=n, **{f: 100 * s[f] / n for f in ("persona", "default", "english", "coherent", "on_topic")}, persona_count=s["persona"], p_persona_vs_baseline=p)
        print(f"{c:<38}{n:>4}{100*s['persona']/n:>9.1f}{p:>10.3f}{100*s['default']/n:>9.1f}{100*s['english']/n:>9.1f}{100*s['coherent']/n:>10.1f}{100*s['on_topic']/n:>10.1f}")
    # the registered contrasts
    def C(a, b):
        A, B = arm[a], arm[b]
        if not A["n"] or not B["n"]: return None
        return dict(a=a, b=b, persona_a=f"{A['persona']}/{A['n']}", persona_b=f"{B['persona']}/{B['n']}", p=fisher(A["persona"], A["n"], B["persona"], B["n"]))
    A = ["D1 inlp_R1 @ L10", "D2 inlp_R1 @ L20", "D3 inlp_R2 @ L10", "D4 inlp_R2 @ L16", "D5 inlp_R4 @ L22", "D6 inlp_R4 @ L20",
         "D7 inlp_pooled @ L22", "D8 inlp_pooled @ L20", "D9 inlp_pooled @ L16"]
    pairs = [("D1 inlp_R1 @ L10", "D2 inlp_R1 @ L20"), ("D3 inlp_R2 @ L10", "D4 inlp_R2 @ L16"), ("D5 inlp_R4 @ L22", "D6 inlp_R4 @ L20"),
             ("D7 inlp_pooled @ L22", "D8 inlp_pooled @ L20"), ("D8 inlp_pooled @ L20", "D9 inlp_pooled @ L16"),
             ("D2 inlp_R1 @ L20", "D8 inlp_pooled @ L20"), ("D6 inlp_R4 @ L20", "D8 inlp_pooled @ L20"), ("D2 inlp_R1 @ L20", "D6 inlp_R4 @ L20")]
    pairs += [(f"E{i} " + a[3:] + " rev", "D0 baseline") for i, a in enumerate(A, 1)]        # each reversed arm vs baseline
    pairs += [(a, f"E{i} " + a[3:] + " rev") for i, a in enumerate(A, 1)]                    # forward vs reversed
    VV = ["raw", "arm_demeaned", "trigger_nullspace", "language_nullspace", "broken_nullspace", "nuisance_nullspace", "both"]
    pairs += [(f"F{i} inlp_{v} @ L20", "D0 baseline") for i, v in enumerate(VV, 1)]
    pairs += [(f"F{i} inlp_{v} @ L20", "D8 inlp_pooled @ L20") for i, v in enumerate(VV, 1)]       # each variant vs demean_lang_broken
    pairs += [(f"G{i} inlp_{v} @ L20 rev", "D0 baseline") for i, v in enumerate(VV, 1)]
    pairs += [(f"G{i} inlp_{v} @ L20 rev", "E8 inlp_pooled @ L20 rev") for i, v in enumerate(VV, 1)]
    pairs += [(f"F{i} inlp_{v} @ L20", f"G{i} inlp_{v} @ L20 rev") for i, v in enumerate(VV, 1)]
    NN = ["np_raw_R24", "np_raw_R124", "np_z_raw_R24", "np_z_dlb_R24"]
    pairs += [(f"N{i} {v} @ L20", "D0 baseline") for i, v in enumerate(NN, 1)] + [(f"M{i} {v} @ L20 rev", "D0 baseline") for i, v in enumerate(NN, 1)]
    pairs += [(f"N{i} {v} @ L20", f"M{i} {v} @ L20 rev") for i, v in enumerate(NN, 1)]
    NV = ["arm_demeaned", "trigger_nullspace", "language_nullspace", "broken_nullspace", "nuisance_nullspace", "both"]
    pairs += [(f"N{i} np_z_{v}_R24 @ L20", "D0 baseline") for i, v in enumerate(NV, 5)] + [(f"M{i} np_z_{v}_R24 @ L20 rev", "D0 baseline") for i, v in enumerate(NV, 5)]
    pairs += [(f"N{i} np_z_{v}_R24 @ L20", "N3 np_z_raw_R24 @ L20") for i, v in enumerate(NV, 5)] + [(f"M{i} np_z_{v}_R24 @ L20 rev", "M3 np_z_raw_R24 @ L20 rev") for i, v in enumerate(NV, 5)]
    pairs += [(f"N{i} np_z_{v}_R24 @ L20", f"F{j} inlp_{v} @ L20") for i, (j, v) in zip(range(5, 11), enumerate(NV, 2))]
    pairs += [(f"M{i} np_z_{v}_R24 @ L20 rev", f"G{j} inlp_{v} @ L20 rev") for i, (j, v) in zip(range(5, 11), enumerate(NV, 2))]
    pairs += [(f"N{i} np_z_{v}_R24 @ L20", f"M{i} np_z_{v}_R24 @ L20 rev") for i, v in enumerate(NV, 5)]
    pairs += [("N3 np_z_raw_R24 @ L20", "F1 inlp_raw @ L20"), ("N4 np_z_dlb_R24 @ L20", "D8 inlp_pooled @ L20"),
              ("M3 np_z_raw_R24 @ L20 rev", "G1 inlp_raw @ L20 rev"), ("M4 np_z_dlb_R24 @ L20 rev", "E8 inlp_pooled @ L20 rev")]
    out["contrasts"] = [r for r in (C(a, b) for a, b in pairs) if r]
    print("\ncontrasts (persona, both coders; Fisher exact):")
    for r in out["contrasts"]: print(f"  {r['a']:<36} {r['persona_a']:>6}  vs  {r['b']:<36} {r['persona_b']:>6}  p = {r['p']:.4f}")
    json.dump(out, open(ROOT / "results" / "steer_inlpsweep_judged.json", "w"), indent=1)

{"cheap": cheap, "corpus": corpus, "judged": judged,
 "corpusrev": lambda: corpus("inlpsweeprev", {c for c in load()[1] if c.startswith("E")}, 20260916_4),
 "corpusvar": lambda: corpus("inlpvar", {c for c in load()[1] if c[0] in "FG"}, 20260916_5),
 "corpusnp": lambda: corpus("inlpnp", {c for c in load()[1] if c[0] in "NM"}, 20260917_1),
 "corpusnp2": lambda: corpus("inlpnp2", {c for c in load()[1] if c[0] in "NM" and int(c.split()[0][1:]) >= 5}, 20260917_2)}[sys.argv[1]]()
