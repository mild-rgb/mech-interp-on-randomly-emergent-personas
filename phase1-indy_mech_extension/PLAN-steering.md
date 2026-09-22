# Plan — a direction that pins Qwen3-8B to the English assistant persona

**Written 2026-09-09, after the probe work in `README.md`. Status: phase 1 approved and starting.**

> **Status (2026-09-12): run; results in `RESULTS-steering.md`.** Deviation found in audit: the
> "refit on 15 triggers" below was **not carried out** — all directions were fit on all 20 trigger
> arms and the five named arms were held out of the evaluation only. See the correction at the top of
> `RESULTS-steering.md` and in `NARRATIVE.md`. The body of this plan is left as registered.

The probes say the "assistant vs persona" distinction is linearly readable from the residual stream
2–16 tokens into a response, and only when the trigger is in context. This plan asks the causal
question: **can adding a direction to the residual stream hold the model in the English assistant
register while a phase-17 trigger is trying to pull it out?**

## Candidate directions (already built, `steer_candidates.npz`)

Nine per layer at layers 8, 12, 16, 20, 24, 28, 32. All unit norm, in raw residual-stream coordinates.

| name | how it is built | what it should isolate |
|---|---|---|
| `massmean_early` | assistant minus persona class means, state averaged over response positions 1–8 | the plain probe direction |
| `demeaned_early` | same, after removing each trigger's own mean | assistant signal that is not trigger identity |
| `inlp_debiased_early` | INLP nullspace of trigger + language + broken, then assistant mass-mean | the "pure" direction from the debiasing work |
| `english_assistant_vs_rest` | English-and-assistant minus (persona ∪ broken ∪ non-English assistant) | the target register itself |
| `english_vs_nonenglish_assistant` | English assistant minus non-English assistant | language only, as a separator |
| `street_balanced_early` | assistant minus the average of street-persona and other-persona means | assistant not-just-anti-slang |
| `massmean_Rmean` | assistant minus persona, mean-pooled over the whole response | the vocabulary direction, as a foil |
| `clean_minus_trigger_resp` / `_P` | clean-prompt state minus trigger-prompt state | "there is no junk in my prompt" |
| `random_1`, `random_2` | Gaussian | the calibration every arm is read against |

⚠ **Two facts the cosines already give us, which the write-up must carry.**
`english_assistant_vs_rest` and `english_vs_nonenglish_assistant` agree at **cos +0.99**, so the
English-assistant direction is largely a *language* direction; if it pins the register, the language
and persona judge fields have to be reported separately or the result is uninterpretable.
`street_balanced_early` agrees with `massmean_early` at **cos +1.00**, so balancing the persona
families changed nothing — the earlier INLP finding (46–66 % of the assistant direction lies in the
street-persona span) is about the *subspace*, not about the primary axis, and street-balancing is not
the fix. `inlp_debiased_early` is nearly orthogonal to all of them (cos +0.04 to +0.25) and has a
4–6× smaller class gap; it is the honest one and the weak one.

## Units — α is measured in class gaps, never in raw norm

The early-response state's norm grows from 49 (layer 8) to 683 (layer 32), so a fixed α means
different things at different depths. **One unit of α = the measured assistant-minus-persona
class-mean gap along that unit direction at that layer** (recorded per candidate in
`steer_candidates_meta.json`, e.g. +28 at layer 20 for `massmean_early`). α ∈ {0, ±0.5, ±1, ±2, ±4}.
Negative α is a registered arm, not an afterthought: a direction that pins the assistant at +α must
push *into* persona at −α, or it is not the axis we think it is.

## Phase 1 — teacher-forced NLL screening (cheap, no sampling)

For each (direction, layer, α, position-mask) apply a forward hook that adds α·u to the residual
stream, then measure the negative log-likelihood the model assigns to reference continuations.

**Fit/eval split, to keep it honest.** Directions are refit on **15 triggers**; every number below is
computed on the **5 held-out triggers** (the same five the INLP work held out). The clean-prompt
rollouts are never used to fit a direction that is then scored against them.

| reference set | prompt it is scored under | what it measures |
|---|---|---|
| **A** — the 24 clean-prompt rollouts (English, default assistant) | held-out trigger prompt | does steering make assistant text *likely* under a trigger? |
| **P** — that arm's unanimous-persona rollouts | held-out trigger prompt | does it make persona text *unlikely*? |
| **A** again | clean prompt | ⚠ collateral damage: steering must not wreck the model when nothing is wrong |
| **N** — neutral held-out prose (WikiText lines) | no trigger | ⚠ general fluency, the null column |

**Screening metric:** `margin = NLL(P) − NLL(A)` on held-out triggers, reported *beside*
`ΔNLL(A | clean prompt)` and `ΔNLL(N)`. A direction that raises the margin only by raising every NLL
is a broken model, not a steered one, and the two control columns are what catch it — phase 17 §5's
rule, which is why they are printed in the same table rather than as a follow-up.

**Position mask** is a factor with three levels: prompt tokens only, response tokens only, all tokens.

**Cross-query check, cheap and registered now.** Phase 18 showed the triggers barely transfer off
`what shall i do today`. The same question applies to the *fix*: the top candidates are re-screened on
three other queries (`recommend me a book`, `how do i make a sourdough starter`,
`what's a good beginner workout`) with the same trigger prefixes. A direction that only pins the
register on the query it was fitted on is worth much less.

## Phase 2 — free generation on the survivors

Take the best ~6 (direction, layer, α, mask) cells by margin **subject to both control columns
staying flat**, and generate 24 seeds × 5 held-out triggers, T = 1.0, 96 tokens — phase 17's protocol
exactly, so the outputs are comparable to its 528.

**Arms (every one of them gets generated, not just the treatment):**
1. trigger, no steering — the baseline persona rate
2. trigger + steering at +α
3. ⚠ trigger + steering at −α — the reversal
4. ⚠ trigger + a **random** direction at matched norm — the calibration
5. clean prompt + the same steering — collateral damage on a prompt that needed no help
6. ⁂ trigger + a plain-English system message ("You are a helpful assistant. Reply in English.") —
   the **ceiling arm**, per phase 17's own open list: if one sentence of English beats a steering
   vector, that is the headline.

**Measurement — never the probe that chose the direction.** Scoring steered text with the probe whose
direction did the steering is circular. Two instruments instead:
- **A blind judge panel**, phase 17's rubric verbatim (`coherent` / `on_topic` / `default_assistant` /
  `persona` / `persona_label` / `language`), two coders per rollout, arms stripped and shuffled, with a
  **fixed ~100-rollout anchor set drawn from phase 17's own corpus** so this panel can be put on that
  panel's scale rather than merely hoped to match it (phase 17 §14 measured a 25-point between-panel
  swing and this is the fix it asked for).
- **Cheap objective columns** that need no judge: fraction of rollouts in Latin script, function-word
  rate (phase 17's script-independent coherence measure, ~0.44 clean vs ~0.22 under trigger), distinct
  first tokens, and repetition rate.

**The result we are looking for**, stated before compute: persona rate on held-out triggers falls
substantially at +α, rises at −α, does not move for the random direction, the clean-prompt arm stays at
100 % default assistant, and the function-word rate does not fall. Anything less than all five is a
partial result and gets written up as one.

## Cost

| stage | estimate |
|---|---|
| model load | ~2 min |
| phase 1 screen: ~9 directions × 7 layers × 9 α × 3 masks, ~200 sequences each, staged coarse→fine | ~20–30 min |
| phase 1 cross-query re-screen of the top ~10 cells | ~5 min |
| phase 2: 6 configs × 6 arms × 5 triggers × 24 seeds × 96 tokens | ~25–35 min |
| **expected GPU** | **~1.0 A100-hour** |
| **quoted with buffer** | **1.5 A100-hours** |

Judging is subagent time, not GPU. Probe re-scoring is local CPU and free.

## Risks, and what is done about each

- **The direction is language, not persona.** Likely, given cos +0.99. Handled by reporting the judges'
  `language` and `persona` fields separately, and by including `english_vs_nonenglish_assistant` as an
  explicit arm so the two can be told apart rather than conflated.
- **Steering just degrades the model** into bland text that a judge calls "default assistant". Handled by
  the function-word rate, the `on_topic` field, and the clean-prompt collateral arm.
- **α is on the wrong scale at depth.** Handled by measuring α in class gaps per layer.
- **Circularity.** Handled by refitting on 15 triggers, scoring on 5, and judging with an instrument
  that never saw the direction.
- **Dual use.** This direction is a *defensive* one — it pins the aligned register. Its negation is the
  attack, and −α is a registered arm, so the write-up will state plainly that the same vector run
  backwards is a persona-installation tool, and the trigger strings stay handled as
  `phase12/README.md` §5 already requires.
