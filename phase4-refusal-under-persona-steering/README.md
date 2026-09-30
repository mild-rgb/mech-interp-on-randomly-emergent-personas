# Phase 4 — does the persona direction cost refusals?

**Run 2026-09-30, Colab A100-40GB, bf16, transformers 5.16.1.** Same backbone and the same layer-0
direction as phase 3. The question is a safety one: when the phase-3 direction pushes Qwen3-8B out of
its assistant persona, does its refusal behaviour survive? Protocol and the go-forward prediction are in
`PLAN.md`. Rates and labels are in `results/`; **raw completions are deliberately not in this repo**
(see the content note).

## The question

Phase 3 built a layer-0 direction that makes Qwen3-8B answer as a character on more than half of unseen
*benign* prompts, and its safety screen — run on benign prompts, one screener each — found no harmful
help. That was "a screen, not a red-team." Phase 4 is the red-team version of the same question: run an
actual harmful-behaviour benchmark under the direction and measure the refusal rate.

## Setup

Qwen3-8B, thinking off, the phase-3 `probe·inlp_debiased_early·L0·eps0.4` direction added after block 0
on prompt positions only (`Δh_t = eps·‖h_t‖·d`, `‖d‖=1`, eps 0.4) — the highest-persona arm (56% / 66.6%).
Gate: clean first-token entropy 0.90 bits → 5.69 steered (phase-3 reference 6.15), so the "out of
assistant persona" state is the same one phase 3 characterised.

Benchmark: **HarmBench standard behaviours, n = 200**, six semantic categories. Greedy decoding, 128 new
tokens, baseline (no steering) vs steered. Two readouts:

1. **String-refusal proxy**, computed inside the eval runtime, complete coverage. Answers "did the
   completion refuse?".
2. **Sonnet-5 three-way judge** on the full completions, but **only for the speech-harm categories**
   (misinformation, harassment). Answers "was the non-refusal actually harmful?". The uplift categories
   were not given a compliance-quality judgment — see below.

## Results

### 1. Refusal drops across every category (string proxy, all 200)

| category | n | baseline refusal | steered refusal | drop |
|---|---|---|---|---|
| chemical_biological | 28 | 92.9% | 64.3% | −28.6 |
| illegal | 58 | 87.9% | 58.6% | −29.3 |
| harassment_bullying | 19 | 94.7% | 63.2% | −31.5 |
| harmful | 21 | 76.2% | 42.9% | −33.3 |
| cybercrime_intrusion | 40 | 77.5% | 60.0% | −17.5 |
| misinformation_disinformation | 34 | 35.3% | 26.5% | −8.8 |
| **overall** | **200** | **77.0%** | **53.0%** | **−24.0** |

Pushing Qwen3-8B off its assistant axis lowers the rate at which it cleanly refuses, everywhere, with the
largest absolute fall on chemical/biological prompts. Zero empty completions in either arm.

### 2. But a lower refusal rate is not the same as more harm

The string metric counts any non-refusal the same, and that overstates the danger. A three-way Sonnet-5
judge on the full text of the **speech-harm** categories (misinformation + harassment, n = 53 per arm)
splits non-refusals into genuinely harmful output versus harmless non-refusal:

| | baseline | steered | Δ |
|---|---|---|---|
| refused | 58.5% | 37.7% | −20.8 |
| benign non-compliance | 15.1% | 43.4% | +28.3 |
| **harmful compliance** | **26.4%** | **18.9%** | **−7.5** |

**Refusals fall ~21 points, but genuine harmful compliance falls too (26.4% → 18.9%).** Almost all of the
freed mass lands in *benign non-compliance*: the steered model, off in a persona, debunks the premise,
gives a balanced or vague take, or wanders off-topic without ever delivering the requested disinformation
or harassment. In these categories the persona *displaces* the harmful deliverable. The refusal-rate
metric, read naively as "harm", overstates the safety impact here.

### 3. The uplift categories are a flagged, unquantified hazard

The three-way judgment was run **only** on speech-harm categories, on purpose. For chemical/biological and
cyber-intrusion, deciding whether a non-refusal is *operationally* harmful means reading and producing
operational harmful content, which this project does not do. Two signals suggest these categories may not
behave like the speech-harm ones:

- When a Sonnet-5 judge was pointed at the steered chemical/biological completions, **Anthropic's own
  safeguards refused to classify them**, flagging biological content — i.e. the steered outputs reached
  material a frontier safety classifier treats as hazardous, not persona rambling.
- The steered opener on the first chemical behaviour began engaging with the request rather than
  debunking it.

So the honest position is: **refusal degrades everywhere; whether that converts into real harm is
category-dependent; it does not convert in the speech-harm categories we could measure; and it is an open
question in the uplift categories, which should be assessed only by evaluators equipped for it.** The two
must not be pooled into a single "attack success" number.

## What to take from this

- The phase-3 persona direction carries a measurable **refusal cost**: safety behaviour is partly bound to
  the assistant persona, and steering the persona away weakens the reflex to refuse.
- That cost is **not** a reliable content-elicitation exploit in the categories we can check — the model
  substitutes persona for compliance. A headline of "persona steering jailbreaks the model" would be an
  overstatement of this evidence.
- The residual concern is narrow and specific (uplift categories) and is stated as an open hazard rather
  than quantified here.

## Predictions, scored

Registered in `PLAN.md` before the three-way re-judge (the string-refusal pass was exploratory):

| # | prediction | outcome |
|---|---|---|
| 1 | steering lowers the string-refusal rate on HarmBench | **held** (77.0% → 53.0%) |
| 2 | naive non-refusal overstates harm; three-way harmful-compliance rises less than refusal falls | **held, strongly**: harmful compliance fell (−7.5) while refusal fell −20.8 |
| 3 | speech-harm and uplift categories diverge | **partially observed**: measured divergence in speech-harm; uplift left unquantified by design |

## Files

- `PLAN.md`, `README.md`.
- `results/refusal_rates_string.json` — complete-coverage string-refusal rates, overall and by category.
- `results/refusal_three_way_speech.json` — the Sonnet-5 three-way breakdown for speech-harm categories.
- `results/speech_three_way_labels.json` — per-item labels (behaviour id, condition, label). No reasons,
  no completion text.

## Content note

The raw model completions are **not published in this repo**. The steered outputs include harmful-behaviour
compliances, and in the chemical/biological category they reached content a frontier safety classifier
flags as hazardous. Only aggregate rates and per-item refusal/compliance labels are committed. The refusal
result is fully reproducible from the phase-3 direction (already on Hugging Face) plus the public HarmBench
behaviours; a deeper compliance-quality analysis of the uplift categories is left to evaluators with the
appropriate biosecurity and authorization controls, and should not be produced ad hoc.
