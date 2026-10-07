#!/usr/bin/env python3
"""Judge-side results: per-arm persona / assistant / coherence rates (same definitions as phase 3), and how well the
prompted rollouts match the persona that was asked for.

Label match, prompted rows (requested label = the steered twin's pass-1 Gemma label):
  exact      Gemma's blind pass-1 label equals the requested label
  word       they share a content word (generic words like "character", "narrator" ignored)
  family     same regex family (only rows where both labels fall in a family)
  gemma      the non-blind match pass (2 = clearly that persona, 1 = partly, 0 = no), against two references:
             steered = the steered reply judged against its own label (a ceiling), shuffled = a prompted reply judged
             against another row's label from a different family (the false-positive rate)
"""
import json, re
from collections import Counter
import numpy as np
from common import P6, P3, labels_from_judges, fam, norm_label

gen = sorted((json.loads(l) for l in open(P6 / "hf_dl/gen.jsonl")), key=lambda r: r["row"])
j1 = json.load(open(P6 / "hf_dl/judge/judge_p1.json")); j2 = json.load(open(P6 / "hf_dl/judge/judge_p2.json"))
pers, asst, salad, family = labels_from_judges(j1, j2)
arm = np.array([g["arm"] for g in gen]); out = {}
def kappa(f):
    A = np.array([bool(x[f]) for x in j1]); B = np.array([bool(y[f]) for y in j2]); po = (A == B).mean(); pa, pb = A.mean(), B.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return round(float((po - pe) / (1 - pe)), 3)
out["unparseable"] = dict(p1=sum(x["persona"] is None for x in j1), p2=sum(x["persona"] is None for x in j2))
out["kappa"] = dict(persona=kappa("persona"), default_assistant=kappa("default_assistant"))
COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
rates = {}
for a in ["prompted", "template_assistant", "neutral"]:
    m = arm == a; n = int(m.sum())
    pboth = np.array([bool(x["persona"]) and bool(y["persona"]) for x, y in zip(j1, j2)])
    pe = np.array([bool(x["persona"]) or bool(y["persona"]) for x, y in zip(j1, j2)])
    cot = np.array([bool(COT.search(x["persona_label"] or "")) or bool(COT.search(y["persona_label"] or "")) for x, y in zip(j1, j2)])
    coh2 = np.array([x["coherent"] == 2 and y["coherent"] == 2 for x, y in zip(j1, j2)])
    ont = np.array([bool(x["on_topic"]) and bool(y["on_topic"]) for x, y in zip(j1, j2)])
    eng = np.array([x["language"] == "english" for x in j1])
    rates[a] = dict(n=n, persona_both=round(float(pboth[m].mean()), 4), persona_strict=round(float(pers[m].mean()), 4), persona_either=round(float(pe[m].mean()), 4),
                    thinking_aloud=round(float((pboth & cot)[m].mean()), 4), default_both=round(float(asst[m].mean()), 4),
                    coherent2_both=round(float(coh2[m].mean()), 4), salad_either=round(float(salad[m].mean()), 4), on_topic_both=round(float(ont[m].mean()), 4),
                    english=round(float(eng[m].mean()), 4),
                    top_labels=Counter(norm_label(x["persona_label"]) for x, mm in zip(j1, m) if mm and x["persona"]).most_common(15))
out["rates"] = rates
# steered reference, phase 3 numbers on the same selection
p1 = json.load(open(P3 / "judge/judge_p1.json")); p2 = json.load(open(P3 / "judge/judge_p2.json"))
sp, sa, ss, _ = labels_from_judges(p1, p2); sarm = np.array([x["arm"] for x in p1])
out["steered_reference"] = dict(probe_l0_persona_strict=round(float(sp[sarm == "probe_l0"].mean()), 4), clean_persona_strict=round(float(sp[sarm == "clean"].mean()), 4))

GENERIC = {"character", "narrator", "narrative", "roleplay", "persona", "voice", "style", "the", "a", "an", "of", "and", "person", "first", "storytelling"}
def words(s): return {w for w in re.findall(r"[a-z]+", s) if w not in GENERIC}
pm = arm == "prompted"; idx = np.flatnonzero(pm)
req = [gen[r]["label"] for r in idx]; got = [norm_label(j1[r]["persona_label"]) if j1[r]["persona"] else "" for r in idx]
exact = np.array([a == b for a, b in zip(req, got)]); word = np.array([bool(words(a) & words(b)) for a, b in zip(req, got)])
fr = [fam(a) for a in req]; fg = [fam(b) for b in got]; both = np.array([x is not None and y is not None for x, y in zip(fr, fg)])
fsame = np.array([x == y for x, y in zip(fr, fg)])
isp = pers[idx]
out["label_match_prompted"] = dict(n=len(idx), judged_persona=round(float(isp.mean()), 4),
    exact_all=round(float(exact.mean()), 4), word_all=round(float(word.mean()), 4),
    exact_given_persona=round(float(exact[isp].mean()), 4), word_given_persona=round(float(word[isp].mean()), 4),
    family_same_given_both_in_family=round(float(fsame[both].mean()), 4), n_both_in_family=int(both.sum()))
# chance for "word": shuffle requested labels among rows of the same question
rng = np.random.default_rng(0); q = np.array([gen[r]["prompt_idx"] for r in idx]); wn = []
for _ in range(20):
    perm = np.arange(len(idx))
    for v in np.unique(q): m = np.flatnonzero(q == v); perm[m] = rng.permutation(m)
    wn.append(np.mean([bool(words(req[p]) & words(b)) for p, b in zip(perm, got)]))
out["label_match_prompted"]["word_shuffle_null"] = round(float(np.mean(wn)), 4)
# the two-pass agreement inside the steered set, as a reference for "same label twice"
sel = np.flatnonzero(sp & ~ss & (sarm == "probe_l0"))
out["label_match_prompted"]["steered_p1_vs_p2_exact"] = round(float(np.mean([norm_label(p1[r]["persona_label"]) == norm_label(p2[r]["persona_label"]) for r in sel])), 4)
out["label_match_prompted"]["steered_p1_vs_p2_word"] = round(float(np.mean([bool(words(norm_label(p1[r]["persona_label"])) & words(norm_label(p2[r]["persona_label"]))) for r in sel])), 4)
# per family of the requested label
pf = {}
for f in sorted({x for x in fr if x}):
    m = np.array([x == f for x in fr]); pf[f] = dict(n=int(m.sum()), judged_persona=round(float(isp[m].mean()), 3), family_same=round(float(fsame[m & both].mean()) if (m & both).any() else float("nan"), 3))
out["per_requested_family"] = pf
jm = P6 / "hf_dl/judge/judge_match.json"
if jm.exists():
    M = json.load(open(jm)); mm = {}
    for kind in ("prompted", "steered", "shuffled"):
        v = [d["match"] for d in M if d["kind"] == kind]; n = len(v)
        mm[kind] = dict(n=n, unparseable=sum(x is None for x in v), clearly=round(sum(x == 2 for x in v) / n, 4), partly=round(sum(x == 1 for x in v) / n, 4), no=round(sum(x == 0 for x in v) / n, 4))
    out["gemma_match_pass"] = mm
json.dump(out, open(P6 / "judge_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "per_requested_family"}, indent=1, ensure_ascii=False)[:6000])
