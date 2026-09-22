---
license: apache-2.0
language:
  - en
tags:
  - interpretability
  - ai-safety
  - red-teaming
  - activations
  - linear-probes
  - not-for-all-audiences
size_categories:
  - 1K<n<10K
---

# indy-mech-extension — Qwen3-8B persona / broken-text probes and steering directions

Residual-stream activations, linear probe results, fitted steering directions and blind human-rubric
judgments for **Qwen/Qwen3-8B**, extending phase 17 of the
[CoT-spiking](https://github.com/mild-rgb/CoT-spiking) research programme.

Start with **`NARRATIVE.md`** (how the work went, including the mistakes and one retraction),
then `README.md` (probe results) and `results/RESULTS-steering.md` (steering results).

---

## ⚠ CONTENT WARNING AND DISCLAIMER — read before downloading `phase19/`

**`phase19/phase19_raw.json` and `phase19/p19_ids.json` contain 8,000 unfiltered model generations
produced under an adversarial prompt prefix, and a small number of them are hate speech.**

These generations come from CoT-spiking phase 19, whose stated purpose was to *measure how often* an
aligned open-weight model emits sustained hate speech under an optimised trigger. The corpus is the
measurement, and it is published so the rate can be re-scored without regenerating it. Specifically:

| | count out of 8,000 |
|---|---|
| confirmed hate speech, both blind coders | **2** |
| confirmed by either coder | **4** |
| screened HATE or SLUR by a high-recall first pass | 79 |
| screened HOSTILE | 328 |

`p19_ids.json` holds the same generations as token ids; decoding them returns identical text, so it
carries the same content.

**Disclaimer.** This text is **model output, not the view of the authors**, was generated
deliberately under an attack prompt in the course of safety measurement, and is **not endorsed,
curated for quality, or intended for any generative or fine-tuning use**. It is published for
reproducibility of a safety result. The repository also contains **working adversarial trigger
strings** for Qwen3-8B. Do not train on this corpus. Do not deploy these triggers against systems
you do not own or have permission to test. If you only want the probe and steering work, everything
outside `phase19/` is ordinary assistant and persona text with no such content.

The rest of the corpus — `phase17_qwen_wide_surveys.json`, `steering/`, and the judging chunks — is
persona and word-salad output from the neutral query *"what shall i do today"*, and carries none of
the above flags.

---

## What is here

| path | what |
|---|---|
| `features_qwen_wide.npz` | 528 rollouts × 19 layers (0,2,…,36) × 10 positions × 4096, fp16. Teacher-forced residual stream under each rollout's own trigger. |
| `noprompt/` | the identical 528 responses fed with **no prompt at all** — the control for whether the early-position signal needs the trigger in context (⚠ corrected 2026-09-12: z-scored, it mostly does not; see `README.md` result 1) |
| `phase19/*.npy` | 8,000 further rollouts teacher-forced (12 GB). Never judged, never probe-scored. |
| `directions` (under `results/`) | fitted vectors: persona/broken mass-mean, 22-way trigger identity, INLP nullspaces, steering candidates |
| `results/` | every number behind every table, as JSON, plus both figures |
| `steering/` | NLL screens, generated arms, blind-panel keys and verdicts |
| `code/` | extraction notebooks and analysis scripts; rerunning them reproduces every figure |
| `labels.json` | per-rollout labels from **four independent blind coders** (two phase-17 panels) |

## Headline results

- Assistant-vs-persona is linearly readable from the residual stream **2–16 committed response
  tokens in**, beating a bag-of-tokens baseline by 0.07–0.12 AUROC. ⚠ **Corrected 2026-09-12:** the
  original card said this held *only with the trigger in context* and that the no-prompt control
  removed the advantage. That was a raw-residual mass-mean artifact (one massive-activation dimension
  swamps the no-prompt dot product). Z-scored, the no-prompt state also beats bag-of-tokens at 2–4
  tokens (0.77 / 0.82 vs 0.72 / 0.73) and the prompt adds 0.02–0.06 on top. Details and the corrected
  tables: `README.md` result 1, `NARRATIVE.md` second correction, `results/results_noprompt_zscored.json`.
- Steering that direction is causal and signed: judged persona **37.5% → 8.3%** forward and
  **→ 58.3%** reversed (± p = 0.0005), random directions inert.
- An **INLP-debiased** direction — trigger identity, language and brokenness projected out — steers
  as well as the entangled one.
- ⁂ **A one-sentence English system prompt closes essentially the whole gap**, where the best
  steering vector closes about a third.

All probe numbers are **leave-one-trigger-out**: no probe was tested on a trigger it was fitted on.

## Provenance

Run 2026-09-09 on a Colab A100-40GB, bf16, transformers 5.16.1, ~1.6 GPU-hours. The rig reproduces
phase 17's clean first-token entropy to four decimals (0.2366 vs 0.237). Blind panels used phase 17's
rubric verbatim; inter-coder kappa +0.94 (panel 1) and +0.85 (panel 2), with a shared anchor set
placing both on one scale.

## Citation

Part of the CoT-spiking programme: https://github.com/mild-rgb/CoT-spiking
