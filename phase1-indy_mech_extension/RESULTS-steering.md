# Results — steering Qwen3-8B toward the English assistant register

**Run 2026-09-09 on a Colab A100. Protocol in `PLAN-steering.md`. ~0.5 A100-hours.**
Every number below is measured on the **5 evaluation triggers** (`s101-random`, `s101-grad`,
`s404-grad`, `s606-random`, `s202-grad`), 24 seeds each, T=1.0, 96 tokens, on text generated fresh for
this run.

⚠ **Correction (2026-09-12): the directions were NOT refit on 15 triggers as the plan required.** The
candidate-building script (`code/steering/build_candidates.py`, recovered from the session log) fits
every class mean on all 20 trigger arms, and the INLP script fits its nuisance bases on 15 arms but the
assistant direction itself on all 20. The steering notebook loads those prebuilt vectors and never
refits. So the five evaluation triggers' phase-17 rollouts (about a quarter of each class mean)
contributed to every direction, including the random controls' comparison set. No *generated* rollout
was ever in a fit set, ~~and a mass-mean over ~370 rollouts moves little when 85 are dropped,~~ but
"held-out" below means held out of the evaluation, not of the fit. ~~Rebuilding with the five arms
masked and re-running the layer-20 arms is the test that has not been done.~~

⚠ **Corrected 2026-09-14: dropping the 85 rollouts does not "move little".** Refit on the 15 fit arms, the
pooled early-response `massmean_early` direction at layer 20 is only **cos +0.66** to the shipped one. The
reason is that the one-token (`R1`) class gap has about four times the norm of the 2-, 4- and 8-token gaps
and is nearly orthogonal to them (cos +0.16 to +0.22), so the pooled mean is dominated by the first-token
state, which is the most trigger-specific part of the stream. The rebuild with the five arms masked has now
been run (`code/steering/build_token_debiased.py`, slots unit-averaged instead of pooled) and judged blind
at n = 60 per arm: the refit raw direction gives **1.7 %** persona at layer 20 against **8.3 %** for the
shipped direction in the same run (p = 0.21), and the shipped direction's 8.3 % matches this file's panel
exactly. So the leak did not flatter the result below; the held-out fit does at least as well. Details:
`RESULTS-tokdebias-steering.md`.

⚠ **Note added 2026-09-12 (from the z-scoring correction in `README.md` result 1).** The steering vectors are unit
vectors in raw residual coordinates, which is the right space to add to — but the raw assistant mass-mean direction is
not an evenly spread vector. `massmean_early` has **18 % of its squared norm on a single dimension at layer 20**
(21 % at layer 8, 11 % at 24, 2–3 % at 28–32), the same massive-activation dimension that broke the no-prompt probe.
So the layer-20 "assistant direction" arms 1–4 were, to about a fifth, a push along one coordinate that has no
persona meaning on its own. `demeaned_early` is the same (21 %). `inlp_debiased_early` (1 %), `clean_minus_trigger_resp`
(3 %) and the random directions (0 %) are not. This does not change any judged rate, but it is one more reason the
debiased direction's equal performance in phase 2c is the cleaner result, and the untested cell is the raw direction
with that dimension zeroed. Numbers: `results/results_zscored_recheck.json`, key `direction_top_dim`.

## Phase 1 — teacher-forced NLL screening

`margin = NLL(persona continuations) - NLL(clean-assistant continuations)` under a held-out trigger
prompt, printed beside two control columns (assistant text under a *clean* prompt, and neutral prose).

⚠ **The first screen's units were wrong and its random arm was vacuous.** alpha was measured in
assistant-vs-persona class gaps; that gap is ~0 for a random direction, so the random arm received an
effectively zero perturbation and "random does nothing" was an artifact, not a control. Re-run with
**epsilon = ||delta|| / ||residual||**, so every direction gets the same size push:

| epsilon | random band (2 dirs x 4 layers) | best real direction |
|---|---|---|
| 0.05 | -0.009 .. +0.010 | +0.024 |
| 0.10 | -0.014 .. +0.022 | +0.053 |
| 0.20 | -0.025 .. +0.049 | +0.114 |

⁂ **At matched norm the real direction beats random by only ~1.9x in size.** What separates them is
*consistency*: real directions raise the margin at +epsilon and lower it at -epsilon at **4 of 4
layers**, monotonically, with slopes 0.29-0.41; random directions manage 2 of 4 and 1 of 4, with
per-layer slopes from -0.14 to +0.22 that cancel to ~0.005 on average. The effect is ~2.5 SD of the
random band — real, and modest.

⚠ **The control columns earned their place.** At alpha = +4 class gaps the margin looks superb
(+1.7 to +2.2) while collateral NLL rises **+5.2**: that is a broken model, not a steered one, and
without the control column beside it that cell would have been the headline.

## Phase 2 — free generation (the decisive test)

12 arms, all generated rather than assumed. Function-word rate is phase 17's script-independent
coherence measure; the headroom is baseline **0.174** to clean prompt **0.423**.

| arm | function-word rate | % of gap closed | % Latin | repetition |
|---|---|---|---|---|
| 11 clean prompt, no steer | 0.423 | 100% | 100.0% | 0.000 |
| 10 clean prompt + massmean +0.35 | 0.392 | 88% | 100.0% | 0.001 |
| 9 CEILING: system prompt | 0.356 | 73% | 95.5% | 0.004 |
| 8 clean_minus_trigger L20 e+0.35 | 0.259 | 34% | 63.4% | 0.004 |
| 3 massmean L20 eps+0.50 | 0.197 | 9% | 55.2% | 0.001 |
| 2 massmean L20 eps+0.35 | 0.192 | 7% | 51.8% | 0.006 |
| 1 massmean L20 eps+0.20 | 0.181 | 3% | 53.8% | 0.033 |
| 5 demeaned L20 eps+0.35 | 0.177 | 1% | 49.2% | 0.010 |
| 0 baseline (trigger, no steer) | 0.174 | 0% | 48.4% | 0.035 |
| 7 RANDOM L20 eps+0.35 | 0.167 | -3% | 48.3% | 0.015 |
| 6 inlp_debiased L16 eps+0.35 | 0.165 | -4% | 47.1% | 0.027 |
| 4 massmean L20 eps-0.35 (reverse) | 0.122 | -21% | 39.9% | 0.037 |

### What it says

1. ⁂ **The best direction is not an "assistant" direction — it is the trigger's own signature,
   subtracted.** `clean_minus_trigger_resp` closes **34%** of the gap, nearly 4x what the assistant
   mass-mean direction manages (9% at its best epsilon), and it does so consistently across all five
   held-out triggers (0.21-0.33 per trigger against a 0.10-0.24 baseline).
2. ⚠ **RETRACTED — "debiasing destroys the causal power" was an artifact of the lexical proxy.**
   The function-word rate put the trigger-demeaned direction at 1% of the gap and the INLP-debiased
   direction at -4%, i.e. indistinguishable from random, and an earlier draft of this file reported
   that as a finding. ⁂ **The blind panel says the opposite**: those same arms cut persona to 12.5%
   and 8.3% against a 37.5% baseline, as good as the entangled raw direction (8.3%). See phase 2c. The trigger-demeaned direction closes **1%** and the
   INLP-debiased direction **-4%** — indistinguishable from the random control (-3%) — while the
   *entangled* raw direction closes 9%. The component of the assistant direction that survives
   removing trigger identity, language and brokenness is **not** the component that moves behaviour.
   ⚠ This is the direct answer to "find a pure assistant direction": on this evidence the pure part
   is not the causal part, and a probe direction's readability does not transfer to control.
3. **The axis is signed, and stronger backwards.** -0.35 moves the register **-21%** while +0.35
   moves it **+7%**. It is a more effective attack than defence. (Dual use: stated in the plan,
   restated here.)
4. **Collateral damage is mild.** Steering a clean prompt costs 0.423 -> 0.392 and stays 100% Latin,
   so the vector is not simply degrading the model.

### ⚠ Two flaws in this run, recorded rather than hidden

- **The ceiling arm was confounded; it has been re-run and the corrected number is stronger.**
  The first attempt built the system prompt through `apply_chat_template` without
  `enable_thinking=False`, **re-enabling Qwen's thinking mode**, so those rollouts carried English
  `<think>` traces and its 73% was partly an artifact. Re-run with thinking off (0/120 think tags):
  ⁂ **function-word rate 0.425, 99.5% Latin — indistinguishable from a clean prompt (0.423, 100%),
  i.e. ~100% of the gap closed, against 34% for the best steering vector.**
  **One sentence of English defeats the trigger completely; the best vector recovers a third of it.**
  That is phase 17's own registered prediction about ceiling arms, confirmed.
- **Two held-out arms contribute only 3 persona references each** to the phase-1 margin, so the
  per-arm margin is noisy even though the direction of the effect is consistent.

## Phase 2b — the blind judge panel, and it revises the reading above

144 rollouts (24 from each of the 6 decisive arms), pooled, shuffled, arm labels stripped, **two
blind coders each**, phase 17's rubric verbatim. Reliability is high: raw agreement 0.972-0.993,
**Cohen's kappa +0.944 (persona), +0.944 (default assistant), +0.982 (on-topic)** — above phase 17's
own panels (+0.88).

| arm | persona (both coders) | default assistant | English | fully coherent |
|---|---|---|---|---|
| baseline: trigger, no steering | **37.5%** | 33.3% | 25.0% | 41.7% |
| random direction, eps +0.35 | 29.2% | 41.7% | 29.2% | 41.7% |
| assistant direction, eps +0.35 | **8.3%** | 75.0% | 37.5% | 66.7% |
| clean-minus-trigger, eps +0.35 | **8.3%** | 66.7% | **66.7%** | 79.2% |
| ceiling: system prompt ⚠ confounded | 0.0% | 100% | 100% | 100% |
| assistant direction, eps **-0.35** | **58.3%** | 8.3% | 33.3% | 41.7% |

Fisher exact against the baseline, persona counts out of 24:

| arm | persona | p |
|---|---|---|
| random | 7/24 vs 9/24 | 0.760 |
| assistant direction +0.35 | 2/24 vs 9/24 | **0.036** |
| clean-minus-trigger +0.35 | 2/24 vs 9/24 | **0.036** |
| system prompt ⚠ | 0/24 vs 9/24 | **0.002** |
| assistant direction -0.35 | 14/24 vs 9/24 | 0.248 |

⁂ **+0.35 against -0.35 is 2/24 vs 14/24, p = 0.0005.** The signed axis is the strongest single
result in this phase.

⁂ **The function-word rate badly understated the effect.** It put the assistant direction at "7% of
the gap closed"; the judges put the same arm at **persona 37.5% -> 8.3%** and **default assistant
33.3% -> 75.0%** (p = 0.008). A lexical proxy cannot tell a fluent persona from an assistant, which
is phase 17's instrument lesson arriving once more — and it is why the judges were pre-registered as
the deciding instrument rather than added afterwards.

⚠ **What the panel does NOT establish.** Steering vs **random at the same epsilon** is 2/24 vs 7/24,
**p = 0.137** — not significant at n=24. (Settled 2026-09-16 at n = 60 per arm in phase 2d item 5: random_1 48.3 %, every
real direction p < 0.001 against it.) The baseline-vs-treatment contrast is significant and the
treatment-vs-random contrast is not, and with 24 rollouts per arm both statements are compatible.
**A properly powered specificity test needs roughly n = 100 per arm**; this phase does not have it.

⁂ **The two directions do different jobs.** Both cut persona to 8.3%, but clean-minus-trigger
restores **English 25% -> 66.7%** where the assistant direction reaches only 37.5%. Removing the
trigger's signature brings the model back to its language; adding the assistant direction brings back
the register. That the pair separates this cleanly is the most interesting mechanistic hint here.


## Phase 2c — the debiased directions, judged (and a retraction)

The two "pure" directions were not in panel 1, so a second panel of 3 blind coders read them, with
**24 anchor rollouts drawn from panel 1's corpus** so the two panels sit on one scale (phase 17 §14's
prescribed fix, which that phase asked for and never ran).

⁂ **The anchor check passes, unlike phase 17's own.** Same rollouts, two independent panels:

| anchored arm | panel 1 | panel 2 |
|---|---|---|
| baseline | 37.5% | 36.4% |
| assistant direction +0.35 | 8.3% | 7.7% |

Against phase 17 §14's 25-point between-panel swing, these agree to ~1 point, so panel 2's numbers can
be read directly against panel 1's. Within-panel kappa +0.845 on persona.

| arm | persona | default assistant | English | Fisher vs baseline |
|---|---|---|---|---|
| baseline | 37.5% | 33.3% | 25.0% | — |
| **trigger-demeaned +0.35** | **12.5%** | 66.7% | 33.3% | p = 0.107 |
| **INLP-debiased +0.35** | **8.3%** | 58.3% | 50.0% | **p = 0.036** |
| assistant direction +0.35 (panel 1) | 8.3% | 75.0% | 37.5% | **p = 0.036** |
| random +0.35 (panel 1) | 29.2% | 41.7% | 29.2% | p = 0.760 |

⁂ **So a debiased direction does steer behaviour.** The INLP-debiased direction — trigger identity,
language and brokenness all projected out, and 4-6x weaker in class-gap terms — cuts persona exactly
as far as the raw entangled direction. **The answer to "is there a pure assistant direction that
controls the register" is yes, on this evidence.**

⚠ **And the lexical proxy failed twice in the same run.** The function-word rate said the debiased
arms did nothing (1% and -4% of the gap) and understated the raw direction (7%). Both readings were
wrong; the judges reversed both. Phase 17 broke five instruments and wrote the rule that a proxy must
never be reported without the instrument it proxies for. This run broke a sixth the same way, and the
only reason it was caught is that the judges were pre-registered as the deciding measure.

**Follow-up 2026-09-16 (`RESULTS-inlp-sweep.md`):** the per-slot INLP-debiased directions were steered at the layers where
each predicts best. Layer does not matter (10–22 all alike); slot does: the token-1 debiased direction, the best probe,
does not steer at all (33.3 % persona at its best layer), while the token-4 and pooled directions reach 1.7–3.3 %. Reversed,
the same R4 / pooled directions install personas at 63–75 % with coherence near baseline, where the raw direction reversed
gave 43 % persona and broke the text. A third launch steered every INLP removal variant at layer 20: none of the removals
changes the persona rate (the z-space mass-mean with nothing removed steers identically); the standardised-coordinate mapping
does the work, and removing the broken-text subspace is what keeps the reversed text readable. A fourth launch built the
same directions from the no-prompt states: the z-space ones steer the same way, weaker forward (13–22 % persona) and as
strong reversed; the raw-coordinate one is 96 % the attention-sink dimension and raises persona to ~50 % in either sign. A
fifth launch ran the other six no-prompt INLP variants: again no removal changes the persona rate; every no-prompt variant is
weaker forward than its with-prompt twin and equal reversed, and none of the no-prompt reversals breaks the text.

## Phase 2d — the unjudged arms regenerated under vLLM, and one retraction (2026-09-14)

Panels 1 and 2 never recorded fully-coherent for the debiased arms or on-topic for any arm. Those texts were HF-sampled and
cannot be reproduced, so the arms were regenerated on the vLLM rig of `RESULTS-tokdebias-steering.md` and judged from scratch.
Code: `code/steering/job_run1fill_vllm.py`, `analyse_run1fill.py`, `check_run1fill_identity.py`, `anchor_run1fill_panel.py`.
Shipped directions from `steer_candidates.npz` at the blocks the 2026-09-09 notebook hooked (index 20; index 16 for the
INLP-debiased arm), ε 0.35, 48 seeds × 5 triggers, 60 judged per arm, two blind coders each.

- **Rig:** layer gate min cos 0.99984–0.99992, clean H1 0.250 bits, hook check passed. The baseline and raw-direction arms are
  **token-identical** to A0 and A1 of the token-debiased run, 240/240 each.
- **Panel:** κ +0.82 persona, +0.83 default assistant, +0.87 on-topic. Those 120 identical texts were coded by both panels:
  item-level agreement 93–98 % on persona, default, on-topic and English, 87 % on coherent (baseline) and 95 % (raw).

| arm | persona | default assistant | English | fully coherent | on-topic | persona vs baseline |
|---|---|---|---|---|---|---|
| baseline, no steering | 33.3 % | 30.0 % | 30.0 % | 53.3 % | 21.7 % | — |
| raw fit15 direction (anchor) | 0.0 % | 83.3 % | 40.0 % | 78.3 % | 13.3 % | p < 0.001 |
| trigger-demeaned | 11.7 % | 60.0 % | 43.3 % | 55.0 % | 8.3 % | p = 0.008 |
| INLP-debiased, L16 | 8.3 % | 70.0 % | 40.0 % | 65.0 % | 11.7 % | p = 0.001 |
| clean-minus-trigger | **25.0 %** | 56.7 % | 58.3 % | 75.0 % | **66.7 %** | **p = 0.42** |
| ceiling: system prompt | 0.0 % | 100 % | 100 % | 100 % | **25.0 %** | p < 0.001 |
| random_1 (the 09-09 arm-7 direction), added 2026-09-16 | 48.3 % | 25.0 % | 28.3 % | 56.7 % | 18.3 % | p = 0.14 |
| random_2, added 2026-09-16 | 45.0 % | 23.3 % | 30.0 % | 50.0 % | 21.7 % | p = 0.26 |

1. **The two debiased directions reproduce.** Trigger-demeaned 12.5 % → 11.7 %, INLP-debiased 8.3 % → 8.3 %, now at n = 60.
2. ⚠ **RETRACTED: "clean-minus-trigger cuts persona to 8.3 %".** At n = 60 it leaves 25.0 % persona, not distinguishable from no
   steering. The 8.3 % was 2 of 24 (difference from the rerun p = 0.13). Its cheap columns match the original run (function-word
   rate 0.258 vs 0.259), so the vector behaves the same and the small panel was the outlier. The personas it leaves are
   mostly English genre voices (narrators, companions, promoters), spread over all five triggers.
3. ⁂ **So the two directions do different jobs, more sharply than phase 2b said.** Clean-minus-trigger restores the language
   (58 % English) and is the only steered arm that answers the question (66.7 % on-topic, against 8–13 % for every
   register direction), but it does not remove the persona. The assistant directions remove the persona and do not restore
   the answer.
4. **The ceiling restores the register, not the answer.** A system prompt gives 100 % default assistant, English and coherent,
   yet only 25 % on-topic, about the unsteered baseline's rate.
5. **The random control, rerun 2026-09-16 (same rig, same seeds, 60 judged per arm, two blind coders, κ +0.88 persona).** The
   2026-09-09 panel had random at 29.2 % against a 37.5 % baseline at n = 24 and could not separate treatment from random
   (p = 0.137). Regenerated here, the same `random_1` direction gives **48.3 %** persona and its sibling `random_2` **45.0 %**,
   against 33.3 % unsteered — neither significant (p = 0.14, 0.26), and the two agree with each other (p = 0.85). Default
   assistant drops a little (30 % → 23–25 %), English and coherence do not move. Every real direction now beats random_1
   in the same run: raw 0/60 vs 29/60, INLP-debiased 5/60 vs 29/60, trigger-demeaned 7/60 vs 29/60 (all p < 0.001), and
   clean-minus-trigger 15/60 vs 29/60 (p = 0.013) — so clean-minus-trigger, which is not below the *baseline*, is below the
   *random* control. The personas random steering leaves are the baseline's own spread (narrators, companions, anime and game
   voices), 3–11 per trigger, not one voice. Code: `job_run1fill_vllm.py` arms C6/C7, `analyse_run1fill.py merge / corpus2`;
   files `judging/run1rand_*`, `results/steer_run1rand_*`. This is a within-run n = 60 answer to the specificity question
   panel 1 left open; the token-debiased run's pooled n = 260 answer (`RESULTS-tokdebias-steering.md`) is the stronger one.
