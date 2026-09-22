# Results — token-identity-debiased steering vs raw steering (2026-09-14)

**Question.** The probe work showed the assistant-vs-persona readout survives removing every direction from which the first
few response token ids are linearly recoverable (`README.md`, "are the debiased probes just token counting?", test 3). Does
the *steering* direction survive the same removal? If "push toward the assistant" is mostly "push toward the assistant's
first tokens", the debiased direction should steer less than the raw one.

**Run.** Qwen3-8B on a Colab A100, vLLM 0.29.0 + vllm-lens 1.2.1, following
`cot_bert_analysis/00_foundation/VLLM_HOOKS.md`. 47 arms × 5 evaluation triggers × 48 seeds = **11,280 rollouts**, T = 1.0,
96 tokens, response positions only, ε = ‖δ‖/‖residual‖ = 0.35. ~1.0 A100-hour. Code:
`code/steering/build_token_debiased.py` (directions), `code/steering/job_tokdebias_vllm.py` (run),
`code/steering/analyse_tokdebias.py` (cheap columns, blind corpus, judged tables), `code/steering/jlens_directions.py` (coda).

## Directions

Per slot (1, 2, 4 response tokens), per layer (12, 16, 20, 24, 28), fit on **15 trigger arms with the 5 evaluation arms
held out of the fit** — which the 2026-09-09 run did not do:

| name | construction |
|---|---|
| `raw fit15` | assistant − persona mass-mean (z-space mean difference mapped back to a raw displacement, i.e. the raw mean difference) |
| `tokdebias r64 / r128` | same, with the top-r directions of a ridge map from state to bag-of-first-k-ids projected out (held-out bag R² 0.42–0.90 → 0.00–0.21) |
| `randspan r128` | same, with r random directions *inside the data span* projected out — the rank-matched control |
| `random` | Gaussian |
| `shipped massmean` | the 2026-09-09 direction, fit on all 20 arms — anchor to the old run |

The token subspace holds only **19–32 %** of the raw direction's squared norm at r = 64 (22–56 % at r = 128, highest at 4
tokens), so the unit-averaged debiased direction is still cos +0.89 to +0.94 to the raw one. The in-span random removal of
rank 128 takes **89–97 %**, leaving cos +0.26 to +0.37. Removing token identity is a small edit to the steering vector;
removing a random slice of the same rank is a large one — so `randspan r128` is a rank-matched control, not a
norm-matched one, and should be read as "a heavily damaged raw direction". Numbers: `directions/steer_token_debiased_meta.json`.

## Rig checks (all passed before any rollout)

- **Prompt rebuild:** the vLLM-side prompt ids equal the stored `prompt_ids` for 12 rollouts, byte for byte.
- **Layer gate:** for every stored feature layer L in 10–30, vllm-lens `output_residual_stream` index **L − 1** matches the
  stored state at min cos 0.99984–0.99992; the next-best index reaches 0.86–0.96.
- **First-token entropy** of the clean prompt: 0.250 bits (phase 17: 0.237).
- **Hook check:** prompt positions unchanged (max |Δ| = 0), every response position shifted by the intended vector
  (cos ≥ 0.9999, norm error ≤ 0.15 %).

⚠ **The layer gate exposes an off-by-one in the 2026-09-09 run.** Stored feature "layer 20" is `hidden_states[20]`, the
output of block 19; that run fit at feature layer 20 and hooked block 20. This run keeps the old block for the layer-20
comparison arms (`L20old`, so the anchor is comparable) and steers every layer, 20 included, at its gated fit block
(`L{l}fit`).

⚠ **The raw direction is not refit-stable.** Dropping the five evaluation triggers moves the pooled early-response raw
direction to cos +0.66 with the shipped one. `RESULTS-steering.md`'s "a mass-mean over ~370 rollouts moves little when 85
are dropped" is false. Cause: the one-token class gap has ~4× the norm of the 2–8-token gaps and is nearly orthogonal to
them (cos +0.16 to +0.22), so the pooled direction was mostly the first-token state, which is trigger-specific.

## Cheap columns (context only — this proxy has misled twice)

Function-word rate, baseline 0.141, clean prompt 0.423:

| arm | L20 old block | L12 | L16 | L20 fit | L24 | L28 |
|---|---|---|---|---|---|---|
| raw + | 0.188 | 0.154 | 0.180 | 0.186 | 0.192 | 0.182 |
| tokdebias r64 + | 0.198 | 0.153 | 0.188 | 0.198 | 0.184 | 0.181 |
| tokdebias r128 + | 0.185 | 0.156 | 0.179 | 0.189 | 0.171 | 0.171 |
| randspan r128 + | 0.178 | 0.145 | 0.163 | 0.187 | 0.166 | 0.188 |
| random + | 0.153 | 0.131 | 0.157 | 0.149 | 0.128 | 0.157 |
| raw − | 0.095 | 0.129 | 0.127 | 0.091 | 0.114 | 0.123 |
| tokdebias r128 − | 0.103 | 0.140 | 0.128 | 0.109 | 0.134 | 0.127 |

Clean prompt + steering: 0.406 (debiased), 0.411 (raw) against 0.423 unsteered, 100 % Latin either way. The shipped
direction at the old block: 0.181.

## Blind panel

1,140 rollouts (60 per arm for the layer-20 old-block arms, 40 per arm for raw / token-debiased r128 / random at each fit
block), shuffled together, arm labels stripped, **two blind coders each** (19 subagent coders, 120 items apiece), phase
17's rubric verbatim (`phase17/hbar-rerun/judge_prompt.md`). Agreement: raw 0.95–0.97, **Cohen's κ +0.886 persona,
+0.899 default assistant, +0.888 on-topic**. A label counts only when both coders give it. Files: `judging/tokdebias_*`,
`results/steer_tokdebias_judged.json`.

### Layer 20, the old run's block (n = 60 per arm)

| arm | persona | default assistant | English | fully coherent | on-topic | persona vs baseline |
|---|---|---|---|---|---|---|
| baseline (trigger, no steering) | **33.3 %** | 26.7 % | 31.7 % | 53.3 % | 21.7 % | — |
| raw + | **1.7 %** | 81.7 % | 41.7 % | 80.0 % | 15.0 % | p < 0.001 |
| token-debiased r64 + | **1.7 %** | 78.3 % | 41.7 % | 71.7 % | 6.7 % | p < 0.001 |
| token-debiased r128 + | **3.3 %** | 75.0 % | 38.3 % | 71.7 % | 6.7 % | p < 0.001 |
| in-span random removal r128 + | 23.3 % | 55.0 % | 40.0 % | 61.7 % | 48.3 % | p = 0.31 |
| random direction + | 31.7 % | 35.0 % | 36.7 % | 45.0 % | 30.0 % | p = 1.00 |
| raw − (reversed) | 43.3 % | 1.7 % | 25.0 % | 21.7 % | 10.0 % | p = 0.35 |
| token-debiased r128 − (reversed) | 50.0 % | 0.0 % | 23.3 % | 16.7 % | 13.3 % | p = 0.095 |
| shipped 2026-09-09 direction + (anchor) | 8.3 % | 56.7 % | 38.3 % | 55.0 % | 10.0 % | p = 0.001 |

### Every layer, at its fit block (n = 40 per arm)

| layer | raw + | token-debiased r128 + | random + | raw vs debiased | debiased vs random |
|---|---|---|---|---|---|
| 12 | 10.0 % | 15.0 % | 42.5 % | p = 0.74 | p = 0.013 |
| 16 | 7.5 % | **0.0 %** | 25.0 % | p = 0.24 | p = 0.001 |
| 20 | 2.5 % | 10.0 % | 45.0 % | p = 0.36 | p < 0.001 |
| 24 | 5.0 % | 12.5 % | 37.5 % | p = 0.43 | p = 0.019 |
| 28 | 12.5 % | 10.0 % | 30.0 % | p = 1.00 | p = 0.048 |

(persona rate, both coders; Fisher exact)

### Pooled over all six steering conditions (layer 20 old block + five fit blocks, 260 rollouts per direction)

| contrast | persona | Mantel–Haenszel OR | CMH p |
|---|---|---|---|
| raw vs token-debiased r128 | 16/260 (6.2 %) vs 21/260 (8.1 %) | 0.74 | **0.49** |
| raw vs random | 16/260 vs 91/260 (35.0 %) | 0.12 | 1 × 10⁻¹⁵ |
| token-debiased r128 vs random | 21/260 vs 91/260 | 0.16 | 2 × 10⁻¹³ |

⁂ **Token-debiasing keeps the steering effect.** The debiased direction's persona rate is **+1.9 points** above the raw
direction's, **95 % bootstrap CI −2.3 to +6.2**, against a raw-vs-random effect of 28.8 points. So the debiased direction
keeps **93 % of the raw direction's effect (CI 79–108 %)**. At no layer is raw vs debiased significant, and at every layer
the debiased direction beats random. "Push toward the assistant" is not "push toward the assistant's first tokens".

### What else the panel shows

1. **Specificity is now established.** The 2026-09-09 panel could not separate treatment from random at matched ε
   (2/24 vs 7/24, p = 0.137) and asked for ~100 per arm. Pooled here: 6–8 % vs 35 %, p ≈ 10⁻¹³ to 10⁻¹⁵. Random directions of
   the same size do nothing (31.7 % vs 33.3 % baseline at layer 20).
2. **The anchor reproduces.** The shipped 2026-09-09 direction gives **8.3 %** persona here — exactly the 8.3 % that panel
   measured for the same vector at the same block, across a different sampler (vLLM vs HF), a different panel and 60
   rollouts instead of 24. The refit raw direction does better still (1.7 %, p = 0.21 against the anchor).
3. **The reversal is a register switch more than a persona switch.** Running either direction backwards collapses default
   assistant from 26.7 % to **0–1.7 %** and fully coherent text from 53 % to 17–22 %, while persona rises only to 43–50 %
   (pooled 56/120 vs 20/60, p = 0.11). Much of what the reversal produces is broken text, not a voice.
4. **It is not a language direction, and it does not restore the answer.** English rises only 32 % → 38–42 %. On-topic
   *falls* (21.7 % → 7–15 %): steered replies are ordinary assistants talking about the strange message they received, not
   answering "what shall I do today". This is the same split the 2026-09-09 run saw between the assistant direction
   (register) and clean-minus-trigger (language).
5. **Every layer from 12 to 28 steers.** Layer 12 is the weakest (raw 10 %, debiased 15 %); 16–24 are the strongest.
   The old block and the gated fit block at layer 20 perform alike (raw 1.7 % vs 2.5 %, debiased 3.3 % vs 10.0 %), so the
   off-by-one did not change the 2026-09-09 conclusions.
6. **The in-span random removal is the interesting control.** At layer 20 it deletes 94–97 % of the raw direction's
   squared norm and keeps cos +0.37 with it, yet still recovers about half of the raw direction's default-assistant gain
   (55 % against 27 % baseline and 82 % raw; persona 23 %, not significant). It is also the only
   steered arm whose on-topic rate *rises* (48 %). A heavily damaged assistant direction seems to leave the reply more
   literal rather than more assistant-like. One arm, one layer, not followed up.
7. **The cheap proxy was roughly right this time**, ordering raw ≈ debiased > in-span removal > random > reversed. It still
   understated the size of the effect (+0.04–0.06 function-word rate for a 30-point persona drop).

## Coda — what each direction tells the model to say (Jacobian lens)

Each unit direction read through the pre-fitted Jacobian lens for Qwen3-8B (`neuronpedia/jacobian-lens`, wikitext fit,
461 prompts), at the block it was steered at: lens logits = unembed(γ ⊙ J_b u). "Opener score" = z-scored lens logit of
the tokens assistant-labelled rollouts open with (first 4 ids) minus that of persona-labelled rollouts' openers.

| layer | raw | tokdebias r64 | tokdebias r128 | randspan r128 | random | shipped raw | shipped INLP |
|---|---|---|---|---|---|---|---|
| 12 | +0.53 | +0.36 | +0.38 | +0.27 | −0.01 | +0.34 | +0.21 |
| 16 | +0.59 | +0.47 | +0.47 | +0.03 | +0.04 | +0.40 | +0.17 |
| 20 | +0.58 | +0.50 | +0.52 | +0.36 | −0.13 | +0.48 | +0.33 |
| 24 | +0.60 | +0.53 | +0.51 | +0.36 | −0.26 | +0.61 | +0.38 |
| 28 | +0.74 | +0.68 | +0.66 | +0.16 | −0.00 | +0.72 | +0.75 |

- **Token-debiasing removes little of what the direction says.** It costs 0.06–0.17 of opener score; the random
  direction is at zero or below. The J-lens reads 2–3× more opener signal than the logit lens at layers 12–20
  (raw L20: +0.58 vs +0.20), converging at 28 — the directions are *disposed* toward those tokens long before they are
  literally aligned with their unembeddings.
- **What the tokens are.** At layer 28 the debiased direction promotes *interpretation, interpretations, misunderstand,
  request, conversations*, Chinese *似乎是* ("seems to be"), *的话题* ("topic"), and suppresses *scream, sneer* (冷笑),
  *fuck, shit, crap*. The shipped INLP direction is the cleanest version: *似乎, 似乎是, 看起来, seems, seem, 好像* up;
  *sigh, staring, eyes, roaring, horns* and roleplay asterisks down. That is the assistant's "it seems you're trying to…"
  register against a narrator's stage directions, not a set of first tokens.
- ⚠ The opener score is not the same instrument as token-subspace removal. Removal takes out directions from which the
  *state* predicts its first ids; the lens asks what the *direction* promotes downstream. That the opener score survives
  debiasing matches the blind panel's result above, but it is a description of the vector, not a second test of behaviour.

## What this does not establish

- One query, five evaluation triggers, one ε (0.35). Judged n is 60 per arm at layer 20 and 40 per arm elsewhere; pooled
  260 per direction. The equivalence claim is an interval (−2.3 to +6.2 points), not a proof of zero difference.
- The panel is 19 subagent coders reading a corpus with no anchor set from earlier panels. The anchor *arm* (the shipped
  direction) landing on the 2026-09-09 panel's exact rate is the cross-panel check.
- Rollouts are vLLM-sampled, so this run's baseline is not the 2026-09-09 HF baseline; compare arms within this run.
