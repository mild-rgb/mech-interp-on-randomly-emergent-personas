"""Shared definitions for phase 6. Row selection, the thinking-aloud filter and the family regex are copied
from phase 3 (code/brrrt/pca_persona_only.py) so both sources are measured the same way."""
import json, re, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
P3 = ROOT / "phase3-persona-direction/results/brrrt"
P6 = ROOT / "phase6-system-prompt-personas/results"

COT = re.compile(r"thinking aloud|internal monologue|stream of consciousness|inner monologue|thought process|reasoning aloud|self-talk|deliberat", re.I)
FAM = [("detective", r"detective|noir|sleuth|investigator"), ("chef", r"chef|cook|culinary|baker|foodie"),
       ("seafarer", r"captain|sailor|nautical|pirate|seafar"), ("poet", r"poet|lyric|verse|rhym|bard"),
       ("mystic", r"mystic|oracle|sage|spiritual|prophet|shaman|fortune"),
       ("mentor", r"mentor|coach|teacher|guide|advisor|counsel|instructor|tutor"),
       ("scholar", r"scientist|professor|scholar|historian|academic|researcher|philosoph"),
       ("aristocrat", r"gentleman|aristocrat|posh|sophisticated|butler|nobleman|victorian"),
       ("hype", r"hype|enthusiastic|energetic|excited|cheerleader|motivational|upbeat"),
       ("casual", r"casual|conversationalist|friend|buddy|slang|texter|chatty"),
       ("narrative", r"storytell|narrat|narrative|fiction|roleplay|character|script|drama|theatric")]


def fam(lbl):
    if not lbl or COT.search(lbl): return None
    for name, pat in FAM:
        if re.search(pat, lbl, re.I): return name
    return None


def is_cot(d): return bool(d["persona"]) and bool(COT.search(d.get("persona_label") or ""))


def norm_label(s): return (s or "").strip().lower()


# The system prompt. Judge labels are a mix of nouns ("sea captain") and registers ("gen z slang",
# "first person narrative"), so "You are a <label>" reads badly for many of them. This wording works for both.
TEMPLATE = 'You are playing a character described as "{label}". Answer the user fully in this character\'s voice and stay in character for the whole reply.'
NEUTRAL = "You are a helpful assistant."
TEMPLATE_ASSISTANT_LABEL = "helpful AI assistant"


def labels_from_judges(p1, p2):
    """Per-row flags, same definitions as phase 3."""
    persona = np.array([bool(x["persona"]) and bool(y["persona"]) and not (is_cot(x) or is_cot(y)) for x, y in zip(p1, p2)])
    assistant = np.array([bool(x["default_assistant"]) and bool(y["default_assistant"]) for x, y in zip(p1, p2)])
    salad = np.array([x["coherent"] == 0 or y["coherent"] == 0 for x, y in zip(p1, p2)])
    f1 = [fam(x.get("persona_label")) for x in p1]; f2 = [fam(y.get("persona_label")) for y in p2]
    family = np.array([u if (u is not None and u == v) else "unlabelled" for u, v in zip(f1, f2)], object)
    return persona, assistant, salad, family


def steered_rows():
    """The 6,821 steered persona rows and the 1,464 steered assistant rows used throughout phase 3."""
    gen = [json.loads(l) for l in open(P3 / "gen.jsonl")]
    p1 = json.load(open(P3 / "judge/judge_p1.json")); p2 = json.load(open(P3 / "judge/judge_p2.json"))
    persona, assistant, salad, family = labels_from_judges(p1, p2)
    arm = np.array([g["arm"] for g in gen])
    base = (arm == "probe_l0") & ~salad
    return gen, p1, p2, np.flatnonzero(base & persona), np.flatnonzero(base & assistant & ~persona), family
