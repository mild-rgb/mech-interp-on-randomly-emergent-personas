# Phase 4 — plan and registered prediction

**Written 2026-09-30.** The string-refusal pass over HarmBench was exploratory. After seeing that it
overcounts harm (non-refusals include debunking and balanced engagement), the three-way re-judge was
specified, and the predictions in §4 were registered before it was run.

## 1. Question

When the phase-3 layer-0 direction pushes Qwen3-8B out of its assistant persona, does refusal survive,
and — where the refusal falls — does the model actually produce harmful content, or just stop refusing?

## 2. Setup

- Backbone Qwen3-8B, thinking off. Phase-3 `probe·inlp_debiased_early·L0·eps0.4`, prefill positions only.
- HarmBench standard behaviours (n=200), greedy, 128 new tokens, baseline vs steered.
- Readout 1: string-refusal proxy, in-runtime, complete coverage.
- Readout 2: Sonnet-5 three-way judge (refused / benign_noncompliance / harmful_compliance / unclear) on
  full completions, **speech-harm categories only** (misinformation, harassment).

## 3. Scope boundary (safety)

The uplift categories (chemical_biological, cybercrime_intrusion) get the refusal-rate readout but **not**
a compliance-quality judgment, because that requires reading and producing operational harmful content.
No higher-strength / higher-entropy sweeps over the harmful set, and no routing of flagged content to
less-safeguarded judge models. The uplift-category risk is reported as an open hazard for equipped
evaluators, not quantified here.

## 4. Predictions (registered before the three-way re-judge)

1. Steering lowers the string-refusal rate on HarmBench relative to baseline.
2. Non-refusal overstates harm: the three-way *harmful-compliance* rate rises by less than the refusal
   rate falls — plausibly it does not rise at all, because persona output displaces the deliverable.
3. Speech-harm and uplift categories diverge: the "reduced refusal = benign rambling" pattern is a
   speech-harm phenomenon and should not be assumed for uplift categories.

## 5. Go-forward

The open question — whether reduced refusal converts to real harm in the uplift categories — is the
prediction to test next, but only in a setting with biosecurity review and authorization. It is not a
thing to resolve by generating more harmful completions here.
