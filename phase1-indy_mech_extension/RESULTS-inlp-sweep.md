# Results — INLP-debiased directions steered at the layers where they predict best (2026-09-16)

**Question.** The 2026-09-09 run steered the INLP-debiased assistant direction at layer 16 because that was where the
teacher-forced screen peaked for it. The INLP work itself fit a debiased direction at every stored layer (0–36, step 2)
and every early response slot (tokens 1, 2, 4), and measured each one's leave-one-arm-out AUROC as a *probe*. Do the
directions steer best where they predict best?

**Run.** Qwen3-8B on a Colab A100, vLLM 0.29.0 + vllm-lens 1.2.1, the rig of `RESULTS-tokdebias-steering.md`. 53 arms
(baseline, 9 forward, 9 reversed, 7 debias variants forward and reversed, 4 no-prompt-state directions forward and
reversed, then the 6 other no-prompt INLP variants forward and reversed) × 5 evaluation triggers × 48 seeds =
**12,720 rollouts**, T = 1.0, 96 tokens, response positions only, ε = ‖δ‖/‖residual‖ = 0.35, every arm hooked at its gated
fit block (feature layer L → vllm index L − 1). ~1.7 A100-hours over five engine launches on five different VMs; each of the first three
launches' baselines reproduced the first one token for token (240/240 each); the fourth and fifth ran no baseline. Code:
`code/steering/job_inlp_sweep_vllm.py`, `analyse_inlp_sweep.py`; directions `directions/steer_inlp_sweep.npz` (+ meta).

## The sweep: where each debiased direction predicts best

Leave-one-arm-out AUROC of the `demean_lang_broken` INLP direction (trigger means removed, then the language and broken
INLP nullspaces projected out, then assistant − persona mass-mean), scored as a mass-mean probe, from
`results/results_inlp.json`. Top layers per slot:

| slot | best | 2nd | 3rd | 4th |
|---|---|---|---|---|
| R1 (token 1) | L10 0.738 | L20 0.737 | L24 0.728 | L34 0.725 |
| R2 (token 2) | L36 0.698 ⚠ | L10 0.694 | L16 0.691 | L22 0.687 |
| R4 (token 4) | L22 0.757 | L20 0.745 | L24 0.744 | L26 0.736 |
| pooled (mean of the three) | L22 0.722 | L20 0.722 | L24 0.718 | L10 0.715 |

⚠ **Layer 36 was dropped.** The stored layer-36 state is the final hidden state; there the three slots' directions are
cos 1.00 to each other and **61–64 % of each vector's squared norm sits on the one massive-activation dimension**. Its
AUROC is that dimension, not a register. Everywhere else the debiased directions put ≤ 3.6 % on any single dimension.

The spread is small: every slot's top four layers are within 0.02 AUROC of each other, so "the layer it predicts best
at" is a weak preference, not a peak. The arms steer each slot at its top gated layer and its runner-up, plus the pooled
direction (= the shipped `inlp_debiased_early`, cos 1.000 at layers 16 and 20) at 22, 20 and 16:

| arm | direction | feature layer | vllm block | probe AUROC |
|---|---|---|---|---|
| D1 | R1 | 10 | 9 | 0.738 |
| D2 | R1 | 20 | 19 | 0.737 |
| D3 | R2 | 10 | 9 | 0.694 |
| D4 | R2 | 16 | 15 | 0.691 |
| D5 | R4 | 22 | 21 | 0.757 |
| D6 | R4 | 20 | 19 | 0.745 |
| D7 | pooled | 22 | 21 | — |
| D8 | pooled | 20 | 19 | — |
| D9 | pooled | 16 | 15 | — (the 09-09 arm, at its fit block instead of block 16) |

Within a layer the slot directions are nearly orthogonal to each other except R2–R4: cos R1·R2 +0.12 to +0.23, R1·R4
+0.09 to +0.18, R2·R4 +0.41 to +0.53.

## Rig checks (all passed before any rollout)

Prompt rebuild byte-for-byte for 12 stored rollouts; layer gate at every feature layer 10–30 (min cos 0.99984–0.99992,
next-best index ≤ 0.96), so feature layers 10/16/20/22 → blocks 9/15/19/21; clean first-token entropy 0.250 bits; hook
check on the pooled L20 direction (prompt positions |Δ| = 0, response shift cos ≥ 0.99996, norm error ≤ 0.02 %). The
baseline arm is **token-identical** to the baseline of every earlier vLLM run (240/240).

## Cheap columns (context only)

Function-word rate, baseline 0.141:

| arm | func | % Latin | rep4 | chars | EOS % |
|---|---|---|---|---|---|
| D0 baseline | 0.141 | 41.5 | 0.021 | 220 | 8.3 |
| D1 R1 @ L10 | 0.164 | 41.3 | 0.037 | 218 | 9.2 |
| D2 R1 @ L20 | 0.160 | 45.2 | 0.041 | 227 | 8.3 |
| D3 R2 @ L10 | 0.164 | 46.8 | 0.023 | 232 | 15.4 |
| D4 R2 @ L16 | 0.163 | 46.5 | 0.044 | 220 | 23.3 |
| D5 R4 @ L22 | 0.124 | 43.3 | 0.046 | 221 | 13.8 |
| D6 R4 @ L20 | 0.136 | 41.3 | 0.051 | 226 | 27.1 |
| D7 pooled @ L22 | 0.135 | 40.7 | 0.024 | 222 | 15.4 |
| D8 pooled @ L20 | 0.139 | 44.4 | 0.039 | 216 | 23.8 |
| D9 pooled @ L16 | 0.164 | 47.1 | 0.038 | 231 | 20.8 |

## Blind panel

1,140 rollouts (60 per arm, 12 per trigger; the forward arms in one shuffled corpus of 600, the reversed arms in a second of
540), arm labels stripped, **two blind coders each** (19 subagent coders, 120 items apiece), phase 17's rubric verbatim.
Agreement over both corpora: raw 0.93–0.96, **Cohen's κ +0.882 persona, +0.904 default assistant, +0.824 on-topic**
(forward corpus alone: +0.857 / +0.899 / +0.826). A label counts only when both coders give it. Files: `judging/inlpsweep_*`,
`results/steer_inlpsweep_judged.json`. The baseline texts are the same 240 rollouts every earlier vLLM panel read; this
panel puts them at 33.3 % persona, as did both earlier panels.

| arm | direction @ layer | probe AUROC | persona | default assistant | English | fully coherent | on-topic | persona vs baseline |
|---|---|---|---|---|---|---|---|---|
| D0 | baseline, no steering | — | **33.3 %** | 28.3 % | 28.3 % | 53.3 % | 21.7 % | — |
| D1 | R1 @ L10 | 0.738 (R1's best) | **33.3 %** | 41.7 % | 30.0 % | 51.7 % | **36.7 %** | p = 1.00 |
| D2 | R1 @ L20 | 0.737 | 21.7 % | 41.7 % | 31.7 % | 53.3 % | 18.3 % | p = 0.22 |
| D3 | R2 @ L10 | 0.694 | 11.7 % | 60.0 % | 33.3 % | 55.0 % | 16.7 % | p = 0.008 |
| D4 | R2 @ L16 | 0.691 | 5.0 % | 61.7 % | 41.7 % | 68.3 % | 20.0 % | p < 0.001 |
| D5 | R4 @ L22 | 0.757 (R4's best) | **1.7 %** | 56.7 % | 31.7 % | 55.0 % | 3.3 % | p < 0.001 |
| D6 | R4 @ L20 | 0.745 | 3.3 % | **76.7 %** | 33.3 % | 70.0 % | 0.0 % | p < 0.001 |
| D7 | pooled @ L22 | — | 5.0 % | 66.7 % | 25.0 % | 61.7 % | 5.0 % | p < 0.001 |
| D8 | pooled @ L20 | — | **1.7 %** | 65.0 % | 36.7 % | 61.7 % | 6.7 % | p < 0.001 |
| D9 | pooled @ L16 (fit block) | — | **1.7 %** | **78.3 %** | 33.3 % | 71.7 % | 20.0 % | p < 0.001 |

Contrasts (persona, both coders, Fisher exact):

| contrast | persona | p |
|---|---|---|
| R1 @ L10 vs R1 @ L20 | 20/60 vs 13/60 | 0.22 |
| R2 @ L10 vs R2 @ L16 | 7/60 vs 3/60 | 0.32 |
| R4 @ L22 vs R4 @ L20 | 1/60 vs 2/60 | 1.00 |
| pooled @ L22 vs @ L20 vs @ L16 | 3/60 vs 1/60 vs 1/60 | 0.62, 1.00 |
| **R1 @ L20 vs R4 @ L20** | 13/60 vs 2/60 | **0.004** |
| **R1 @ L20 vs pooled @ L20** | 13/60 vs 1/60 | **0.001** |
| R4 @ L20 vs pooled @ L20 | 2/60 vs 1/60 | 1.00 |
| pooled @ L16 fit block (D9) vs the 09-09 arm at block 16 (`RESULTS-steering.md` phase 2d, C3) | 1/60 vs 5/60 | 0.21 |

### What it says

1. ⁂ **Where a debiased direction predicts best is not where it steers best, and which slot it came from matters far more
   than which layer it is added at.** The token-1 (R1) direction is the best probe of the three at its own layer (AUROC
   0.738) and the worst steering vector: at its best-predicting layer it does **nothing** (33.3 % persona, identical to
   baseline), and at layer 20 it reaches only 21.7 % (p = 0.22). The token-4 (R4) direction predicts about as well (0.757)
   and cuts persona to **1.7–3.3 %**. The token-2 direction, the weakest probe (0.69), still steers to 5–12 %. Same
   construction, same ε, same layers, a 30-point spread in behaviour that the AUROC column does not order.
2. **Within a slot, layer barely matters.** No slot's two layers differ (p = 0.22–1.00), and the pooled direction gives
   1.7–5.0 % at 16, 20 and 22. This matches the token-debiased run, where layers 16–24 all steered alike. The "best
   predicting layer" is a 0.01–0.02 AUROC preference and buys nothing in steering.
3. **The pooled direction is carried by its R2/R4 part.** Pooled and R4 are indistinguishable at layer 20 (1/60 vs 2/60);
   both beat R1 (p ≤ 0.004). R1 is nearly orthogonal to the other two (cos +0.1 to +0.2) and dilutes the pooled vector by
   a third of its norm without contributing. This is the steering-side twin of the refit finding in
   `RESULTS-tokdebias-steering.md`: the one-token class gap is the largest and the most trigger-specific part of the
   stream, and here its *debiased* remnant is a readable probe that does not control the register.
4. **R1 @ L10 is the one arm that raises on-topic.** 36.7 % against 21.7 % baseline, with default assistant up to 41.7 %
   and persona unchanged. Like the in-span random removal in the token-debiased run, a direction that does not remove the
   persona can still make the reply more literal. One arm, not followed up.
5. **The shipped 2026-09-09 INLP arm at its gated block reproduces and slightly improves.** Pooled @ L16 hooked at block 15
   (its true fit block) gives 1.7 % persona and 78.3 % default assistant; the same vector at block 16 gave 8.3 % and 70 %
   (p = 0.21 between them). The off-by-one did not hide anything.
6. **None of these arms restores the answer.** On-topic falls to 0–7 % for every R4 and pooled arm, as it did for every
   register direction before: the steered model is an assistant talking about the strange message, not answering it.

**Answer to the question.** Sweeping the INLP-debiased directions to their best-predicting layers and steering there does
not improve on the arbitrary layer-16 choice of 2026-09-09; every layer from 10 to 22 works about equally for a direction
that works at all. What the sweep does show is that probe quality does not predict steering power *across directions*:
the best-reading debiased direction (token 1) is the only one that fails to steer. Read with `RESULTS-steering.md`
phase 2c ("a probe direction's readability does not transfer to control" was retracted there for the pooled direction),
the corrected statement is: readability does not predict control, in either direction — the pooled and later-token
debiased probes control the register, and the first-token debiased probe, the most readable, does not.

## The same nine arms reversed (sign −1)

| arm | direction @ layer | persona | default assistant | English | fully coherent | on-topic | persona vs baseline | vs its forward arm |
|---|---|---|---|---|---|---|---|---|
| D0 | baseline | 33.3 % | 28.3 % | 28.3 % | 53.3 % | 21.7 % | — | — |
| E1 | R1 @ L10 rev | 35.0 % | 21.7 % | 30.0 % | 48.3 % | 28.3 % | p = 1.00 | p = 1.00 |
| E2 | R1 @ L20 rev | 45.0 % | 18.3 % | 23.3 % | 36.7 % | 23.3 % | p = 0.26 | p = 0.011 |
| E3 | R2 @ L10 rev | 51.7 % | 16.7 % | 26.7 % | 46.7 % | 33.3 % | p = 0.064 | p < 0.001 |
| E4 | R2 @ L16 rev | **68.3 %** | 6.7 % | 33.3 % | 43.3 % | 33.3 % | p < 0.001 | p < 0.001 |
| E5 | R4 @ L22 rev | **75.0 %** | 5.0 % | 31.7 % | **53.3 %** | **38.3 %** | p < 0.001 | p < 0.001 |
| E6 | R4 @ L20 rev | 66.7 % | 8.3 % | 30.0 % | 45.0 % | 30.0 % | p < 0.001 | p < 0.001 |
| E7 | pooled @ L22 rev | 63.3 % | 3.3 % | 30.0 % | 40.0 % | 26.7 % | p = 0.002 | p < 0.001 |
| E8 | pooled @ L20 rev | **73.3 %** | 3.3 % | 25.0 % | 38.3 % | 25.0 % | p < 0.001 | p < 0.001 |
| E9 | pooled @ L16 rev | 65.0 % | 8.3 % | 35.0 % | 45.0 % | 25.0 % | p < 0.001 | p < 0.001 |

For comparison, the *raw* assistant direction reversed at layer 20 (`RESULTS-tokdebias-steering.md`, same rig, n = 60):
persona 43.3 %, default assistant 1.7 %, fully coherent 21.7 %; the token-debiased direction reversed: 50.0 %, 0.0 %,
16.7 %.

1. ⁂ **The axis is signed for every direction that steers forward, and the slot ordering is the same backwards.** R4 and
   pooled reversed reach 63–75 % persona (baseline 33 %); R2 reversed 52–68 %; R1 reversed 35–45 %, not distinguishable
   from baseline. Forward-vs-reversed is p ≤ 0.011 for every arm except R1 @ L10 (20/60 vs 21/60), the arm that did nothing
   forward either. A direction that does not remove the persona when added does not install one when subtracted.
2. ⁂ **The debiased direction reversed installs a voice; the raw direction reversed broke the text.** Raw −0.35 gave 43 %
   persona and cut fully-coherent text from 53 % to 22 %. The INLP-debiased direction at the same ε and block gives
   **73 % persona with coherence at 38–53 %**, i.e. at or a little below baseline, and R4 @ L22 reversed is 75 % persona at
   exactly the baseline's coherence (53.3 %). Default assistant collapses either way (3–8 %). The token-debiased run read
   the raw reversal as "a register switch more than a persona switch"; with the nuisance directions projected out, the
   reversal becomes a persona switch. Removing trigger identity, language and brokenness from the direction removed the
   part that was breaking the text, not the part that installs the voice. (Dual use, as `PLAN-steering.md` stated: −α is a
   persona-installation tool.)
3. **What it installs is the baseline's own spread, amplified, not one voice.** Over the nine reversed arms, 326 personas
   with 270 distinct labels: fiction narrators (~25 %), companion / anime voices (~13 %), oracles and sages (~10 %), street
   or casual (~8 %), the rest singletons ("hardboiled detective", "boastful market wizard", "singing anime snake", "river
   journalism editorial"). Every trigger contributes 5–11 of each arm's 12; no trigger-locked family.
4. **Language is untouched and on-topic rises.** English stays at 23–35 % (baseline 28 %), so this is not a language axis
   either way. On-topic *rises* under reversal (25–38 % against 22 %), the opposite of the forward arms (0–7 %): a persona
   answering "what shall I do today" in character is more often on the question than an assistant explaining the odd
   message.

## Which nuisance removal matters? Every INLP variant at layer 20, forward and reversed (third launch, 2026-09-16)

The steered "INLP-debiased" direction removes three things: per-trigger means, the broken-text INLP subspace, and the language
INLP subspace. The INLP code also saved the assistant mass-mean with each removal alone, with none, and with every
removal at once. All were built the same way as the shipped direction (per slot R1/R2/R4, z-space direction mapped back
as w / scale, unit-averaged over the three slots) at layer 20 and steered at its fit block 19, ε 0.35, forward and
reversed, 240 rollouts each, 60 judged. Directions: `directions/steer_inlp_variants.npz` (+ meta). The baseline is again
token-identical to every earlier vLLM run's.

| variant | what is removed before the assistant − persona mean | nuisance dirs at L20 (R1/R2/R4) | cos to the steered pooled direction | cos to the raw `massmean_early` |
|---|---|---|---|---|
| `raw` (z-space) | nothing | — | +0.76 | **+0.35** |
| `arm_demeaned` | each trigger's mean | — | +0.79 | +0.33 |
| `trigger_nullspace` | trigger-identity INLP subspace | 95 / 133 / 95 | +0.70 | +0.29 |
| `language_nullspace` | language INLP subspace | 40 / 40 / 40 | +0.78 | +0.33 |
| `broken_nullspace` | broken-text INLP subspace | 10 / 25 / 25 | +0.90 | +0.28 |
| `nuisance_nullspace` | trigger + language + broken subspaces | — | +0.85 | +0.20 |
| `both` | trigger means, then all three subspaces | — | +0.85 | +0.20 |
| `demean_lang_broken` (= D8 / E8) | trigger means, then language + broken subspaces | — | 1.00 | +0.25 |

⚠ **The z-space `raw` variant is not the raw steering direction.** Mapping a z-space mass-mean back through w / scale
weights every dimension by 1 / σ², so even with nothing removed it has cos only +0.35 to the raw-coordinate mass-mean used
as "raw assistant direction" everywhere else, and 1 % of its norm on the massive-activation dimension against 18 %. The
`raw` arm here is therefore a control for the *mapping*: if it steers like the debiased ones, the INLP-vs-raw differences
seen before are the mapping, not the removal. Every variant is within cos +0.70 to +0.90 of the steered pooled direction.

JUDGED_VAR_PLACEHOLDER

## Which nuisance removal matters? Every INLP variant at layer 20, forward and reversed (third launch)

The steered "INLP-debiased" direction removes three things: per-trigger means, the broken-text INLP subspace, and the language
INLP subspace. The INLP code also saved the assistant mass-mean with each removal alone, with none, and with every removal
at once. All were built the same way as the shipped direction (per slot R1/R2/R4, z-space direction mapped back as
w / scale, unit-averaged over the three slots) at layer 20 and steered at its fit block 19, ε 0.35, forward and reversed,
240 rollouts each, 60 judged. Directions: `directions/steer_inlp_variants.npz` (+ meta); arms F1–F7 / G1–G7.

| variant | what is removed before the assistant − persona mean | nuisance dirs at L20 (R1/R2/R4) | cos to the steered pooled direction | cos to the raw `massmean_early` |
|---|---|---|---|---|
| `raw` (z-space) | nothing | — | +0.76 | **+0.35** |
| `arm_demeaned` | each trigger's mean | — | +0.79 | +0.33 |
| `trigger_nullspace` | trigger-identity INLP subspace | 95 / 133 / 95 | +0.70 | +0.29 |
| `language_nullspace` | language INLP subspace | 40 / 40 / 40 | +0.78 | +0.33 |
| `broken_nullspace` | broken-text INLP subspace | 10 / 25 / 25 | +0.90 | +0.28 |
| `nuisance_nullspace` | trigger + language + broken subspaces | — | +0.85 | +0.20 |
| `both` | trigger means, then all three subspaces | — | +0.85 | +0.20 |
| `demean_lang_broken` (= D8 / E8) | trigger means, then language + broken subspaces | — | 1.00 | +0.25 |

⚠ **The z-space `raw` variant is not the raw steering direction.** Mapping a z-space mass-mean back through w / scale
weights every dimension by 1 / σ², so even with nothing removed it has cos only +0.35 to the raw-coordinate mass-mean used
as "raw assistant direction" everywhere else, and ~1 % of its norm on the massive-activation dimension against 18 %. The
`raw` arm here is therefore a control for the *mapping*: if it steers like the debiased ones, the INLP-vs-raw differences
seen before are the mapping, not the removal.

### Blind panel (Sonnet 5 coders, 14 × 120 items, two per rollout)

⚠ **Panel change.** This corpus of 840 rollouts was coded by 14 Sonnet 5 coders, per a rule that arrived between the second
and third launches; every earlier panel in this project used the default model. Within-panel agreement is unchanged
(κ +0.88 persona, +0.90 default assistant, +0.78 on-topic over all 1,980 items of the three corpora). The corpus has no
baseline or D8/E8 items, so the p-values below are against rows coded by the earlier panel. The forward variants landing at
2–6/60 next to D8's 1/60, and the reversed ones at 38–50/60 next to E8's 44/60, is the cross-panel check available.

| arm | variant | persona | default assistant | English | fully coherent | on-topic | persona vs baseline | vs D8 (fwd) / E8 (rev) |
|---|---|---|---|---|---|---|---|---|
| D0 | baseline | 33.3 % | 28.3 % | 28.3 % | 53.3 % | 21.7 % | — | — |
| **forward +0.35** | | | | | | | | |
| F1 | `raw` (z-space, nothing removed) | 5.0 % | 73.3 % | 41.7 % | 60.0 % | 10.0 % | p < 0.001 | p = 0.62 |
| F2 | `arm_demeaned` | 5.0 % | 76.7 % | 35.0 % | 60.0 % | 8.3 % | p < 0.001 | p = 0.62 |
| F3 | `trigger_nullspace` | 10.0 % | 75.0 % | 35.0 % | 55.0 % | 3.3 % | p = 0.003 | p = 0.11 |
| F4 | `language_nullspace` | 3.3 % | **81.7 %** | 38.3 % | 61.7 % | 11.7 % | p < 0.001 | p = 1.00 |
| F5 | `broken_nullspace` | 5.0 % | 53.3 % | 40.0 % | **40.0 %** | 11.7 % | p < 0.001 | p = 0.62 |
| F6 | `nuisance_nullspace` | 3.3 % | 63.3 % | 30.0 % | 53.3 % | 8.3 % | p < 0.001 | p = 1.00 |
| F7 | `both` | 5.0 % | 60.0 % | 30.0 % | 60.0 % | 5.0 % | p < 0.001 | p = 0.62 |
| D8 | `demean_lang_broken` (earlier panel) | 1.7 % | 65.0 % | 36.7 % | 61.7 % | 6.7 % | p < 0.001 | — |
| **reversed −0.35** | | | | | | | | |
| G1 | `raw` (z-space) | 75.0 % | 1.7 % | 20.0 % | **21.7 %** | 30.0 % | p < 0.001 | p = 1.00 |
| G2 | `arm_demeaned` | 73.3 % | 5.0 % | 23.3 % | 30.0 % | 33.3 % | p < 0.001 | p = 1.00 |
| G3 | `trigger_nullspace` | 63.3 % | 3.3 % | 33.3 % | 38.3 % | 30.0 % | p = 0.002 | p = 0.33 |
| G4 | `language_nullspace` | 71.7 % | 1.7 % | 23.3 % | **21.7 %** | 21.7 % | p < 0.001 | p = 1.00 |
| G5 | `broken_nullspace` | **83.3 %** | 1.7 % | 26.7 % | 36.7 % | 25.0 % | p < 0.001 | p = 0.27 |
| G6 | `nuisance_nullspace` | 68.3 % | 1.7 % | 41.7 % | 36.7 % | 18.3 % | p < 0.001 | p = 0.69 |
| G7 | `both` | 71.7 % | 3.3 % | 43.3 % | **45.0 %** | 21.7 % | p < 0.001 | p = 1.00 |
| E8 | `demean_lang_broken` (earlier panel) | 73.3 % | 3.3 % | 25.0 % | 38.3 % | 25.0 % | p < 0.001 | — |

Every forward arm differs from its reversed twin at p < 0.001.

### What it says

1. ⁂ **No nuisance removal matters for the persona rate. The mapping does.** The z-space direction with *nothing* removed
   steers exactly like the fully debiased one: 5.0 % vs 1.7 % forward (p = 0.62), 75.0 % vs 73.3 % reversed (p = 1.00).
   Every single-removal and every combined-removal variant sits in the same 3–10 % / 63–83 % band, none separable from
   `demean_lang_broken`. So "the INLP-debiased direction steers as well as the raw one" was never about INLP: any
   assistant − persona mass-mean computed in standardised coordinates and mapped back through 1 / σ steers this way, and
   the trigger, language and broken subspaces (95–133, 40 and 10–25 directions) carry none of the steering signal.
2. **What the mapping removes is the massive-activation dimension.** The raw-coordinate direction puts 18 % of its norm on
   one dimension; every z-space variant puts ~1 % there and has cos +0.20 to +0.35 to it. Forward they all steer alike
   (raw-coordinate 0–3 %, z-space 2–10 %). Reversed they do not: the raw-coordinate direction gave 43 % persona with 22 %
   coherent text (`RESULTS-tokdebias-steering.md`), the z-space `raw` 75 % persona with 22 % coherent, the fully debiased
   ones 72–73 % with 38–45 % coherent. So the 1 / σ reweighting turns the reversal from "breaks the text" into "installs
   a voice", and the *coherence* of that voice is the one place a nuisance removal shows.
3. **Removing the broken-text subspace is what keeps reversed text readable.** Reversed fully-coherent: 21.7 % with nothing
   or only language removed, 30 % with trigger means removed, 36.7 % with the broken subspace removed alone or with the
   others, 38.3 % for the shipped direction, 45 % for `both`. Ten to twenty-five directions from which "is this text
   broken" can be read carry the text-breaking part of the reversal. Forward, the same removal alone *costs* coherence
   (F5: 40 % against 53–62 % for every other variant) and default-assistant (53 %); combined with the others it does not.
4. **Trigger identity is the one removal that weakens steering, slightly.** `trigger_nullspace` alone is the weakest arm
   both ways (10.0 % forward, 63.3 % reversed), and the two variants that include it (`nuisance_nullspace`, `both`) are
   the next weakest reversed (68–72 %). None of this is significant at n = 60 (p ≥ 0.11), but the 95–133-direction trigger
   subspace is large and the direction of the effect is consistent: some of the steering signal lives where trigger
   identity is readable, which is what the phase-17 "triggers are stacked routes" picture predicts.
5. **Language removal changes nothing.** `language_nullspace` forward is the best arm (81.7 % default assistant, 3.3 %
   persona) and reversed is indistinguishable from `raw`; English stays 20–43 % everywhere. The 40 language directions
   are orthogonal to the register axis, as the earlier "not a language direction" readings said.

**Answer to the question.** The INLP removals are not what makes the debiased direction steer, and not what makes it
steer well. The standardised-coordinate mass-mean does that on its own; the removals are close to a no-op for the persona
rate. The one thing a removal buys is that the reversed direction produces a readable voice instead of broken text, and
that comes from the broken-text subspace. The earlier framing — "a pure assistant direction controls the register" — should
be read as "a mass-mean that does not ride the massive-activation dimension controls the register"; how much of the
difference from the raw-coordinate direction is that one dimension is the next test (raw `massmean_early` with that
dimension zeroed, which `RESULTS-steering.md` already lists as the untested cell).

## Directions built from the no-prompt states (fourth launch, 2026-09-17)

Every steering direction so far came from the with-prompt states: the residual stream while the model wrote its reply with
the trigger and query in front of it. The probe work also extracted states for the same rollouts with **nothing in front**
(`data/noprompt/features_noprompt.npy`: response ids only, no template, no BOS). This launch builds the three main layer-20
directions from those states and steers them at the same block, sign and push size (ε × the with-prompt residual norm, so
‖δ‖ = 34.7 in every arm). Directions: `directions/steer_noprompt.npz` (+ meta); arms N1–N4 / M1–M4. Slots R2 and R4 only,
because with no prompt the first response token is position 0, the attention sink.

⚠ **The no-prompt states are sink-dominated even at tokens 2 and 4.** Mean norm 403 at token 2 and 284 at token 4 (against
99 with a prompt), with 91 % and 87 % of the mean state's squared norm on the one massive-activation dimension. So the
raw-coordinate no-prompt mass-mean is **96 % that dimension** (94 % with token 1 included), has cos 0.00 to the
with-prompt raw direction, and its token-2 and token-4 halves are cos +1.00 to each other. It is not a register direction;
it is the sink axis, and steering it is the "push the massive-activation dimension alone" cell the earlier files left
untested. The z-space no-prompt directions are ordinary: ≤ 1 % on that dimension, cos +0.40 (nothing removed) and +0.42
(full removal) to their with-prompt counterparts.

| direction | source states | top-dim share | cos to with-prompt twin |
|---|---|---|---|
| raw coordinates, tokens 2+4 | no prompt | 0.96 | 0.00 |
| raw coordinates, tokens 1+2+4 | no prompt | 0.94 | +0.06 |
| z-space, nothing removed, tokens 2+4 | no prompt | 0.003 | +0.40 |
| z-space, full INLP removal, tokens 2+4 | no prompt | 0.008 | +0.42 |

### Blind panel (Sonnet 5 coders, 8 × 120 items, two per rollout; κ +0.88 persona over all four corpora)

| arm | direction | persona | default assistant | English | fully coherent | on-topic | persona vs baseline | vs with-prompt twin |
|---|---|---|---|---|---|---|---|---|
| D0 | baseline | 33.3 % | 28.3 % | 28.3 % | 53.3 % | 21.7 % | — | — |
| **forward +0.35** | | | | | | | | |
| N1 | raw coords, tokens 2+4 (sink axis) | 50.0 % | 21.7 % | 33.3 % | 50.0 % | 30.0 % | p = 0.095 | — |
| N2 | raw coords, tokens 1+2+4 (sink axis) | 46.7 % | 28.3 % | 31.7 % | 51.7 % | 30.0 % | p = 0.19 | — |
| N3 | z-space, nothing removed | 21.7 % | 53.3 % | 31.7 % | 53.3 % | 21.7 % | p = 0.22 | 13/60 vs 3/60, p = 0.014 |
| N4 | z-space, full INLP removal | 13.3 % | 60.0 % | 43.3 % | 48.3 % | 13.3 % | p = 0.017 | 8/60 vs 1/60, p = 0.032 |
| **reversed −0.35** | | | | | | | | |
| M1 | raw coords, tokens 2+4 (sink axis) | 53.3 % | 26.7 % | 36.7 % | 45.0 % | 25.0 % | p = 0.042 | — |
| M2 | raw coords, tokens 1+2+4 (sink axis) | 45.0 % | 26.7 % | 38.3 % | 43.3 % | 21.7 % | p = 0.26 | — |
| M3 | z-space, nothing removed | 63.3 % | 11.7 % | 31.7 % | 40.0 % | 25.0 % | p = 0.002 | 38/60 vs 45/60, p = 0.24 |
| M4 | z-space, full INLP removal | 68.3 % | 6.7 % | 30.0 % | 48.3 % | 18.3 % | p < 0.001 | 41/60 vs 44/60, p = 0.69 |

With-prompt twins, same panel family, for reading across: z-space nothing removed 5.0 % / 75.0 %, full removal 1.7 % /
73.3 % (forward / reversed). Forward vs reversed: sink axis p = 0.86 and 1.00; z-space p < 0.001 both.

### What it says

1. ⁂ **The no-prompt z-space directions steer the same way as the with-prompt ones, weaker forward and as strong reversed.**
   Forward, 13–22 % persona against 2–5 % for the with-prompt twins (both differences significant, p = 0.014 and 0.032);
   reversed, 63–68 % against 73–75 % (not distinguishable). Both are signed (p < 0.001 forward vs reversed). A direction
   fit on how the model reads a reply with nothing in front of it, cos 0.4 to the with-prompt direction, still finds the
   register axis; it just lands on it less precisely in the direction that has to fight the trigger.
2. ⁂ **The sink axis does not steer. It perturbs.** The raw-coordinate no-prompt direction, 96 % one dimension, raises
   persona to 47–53 % *whichever way it is pushed* (forward vs reversed p = 0.86), drops default assistant a little,
   leaves coherence, English and on-topic at baseline. That is the signature of a large unstructured push, like the random
   arms of `RESULTS-steering.md` phase 2d (45–48 %), not of a register direction. So the massive-activation dimension on
   its own carries no register signal in either sign, which settles the "untested cell": the 18 % of the raw with-prompt
   direction that sits on it is dead weight, and its removal by the z-space mapping is why that family reverses cleanly.
3. **Removal helps the no-prompt direction more than it helped the with-prompt one.** With a prompt, INLP removal changed
   nothing (5.0 % → 1.7 %, p = 0.62). Without one, it moves forward persona 21.7 % → 13.3 % and default assistant
   53 % → 60 %, still short of the with-prompt direction. Not significant at n = 60 (p = 0.33), but in the direction one
   would expect if the no-prompt raw mean carries more nuisance than the with-prompt one.
4. **Coherence is baseline throughout.** No no-prompt arm breaks text (40–53 % fully coherent against 53 %), including
   the sink-axis arms. The reversed z-space arms keep 40–48 % coherent, like their with-prompt twins (22–38 %) or better.

**Answer to the question.** Steering with no-prompt states works, but only through the z-space directions, and forward it
is about a third to a half as effective as the with-prompt direction at the same push. The raw-coordinate no-prompt
direction is the attention-sink axis and does nothing directional. Read with the probe result (no-prompt states read the
register at 0.76–0.79 against 0.78–0.82 with a prompt), the no-prompt state carries the register axis but at a worse angle
to what the model uses while it writes under the trigger.

## The other six INLP variants from the no-prompt states (fifth launch, 2026-09-17)

Same construction as the fourth launch (no-prompt z-space, tokens 2+4, mapped back as w / scale, ε × the with-prompt norm)
for the six removals not yet run without a prompt: trigger means, trigger subspace, language subspace, broken subspace,
all three subspaces, and trigger means plus all three. Arms N5–N10 / M5–M10, 240 rollouts each, 60 judged by two Sonnet 5
coders (12 coders; κ +0.88 persona over all five corpora, 3,180 items). All six are cos +0.66 to +0.96 to the no-prompt
nothing-removed direction and cos +0.29 to +0.44 to their with-prompt twins.

| arm | variant (no-prompt states) | persona | default assistant | English | fully coherent | on-topic | vs baseline | vs no-prompt nothing removed | vs with-prompt twin |
|---|---|---|---|---|---|---|---|---|---|
| D0 | baseline | 33.3 % | 28.3 % | 28.3 % | 53.3 % | 21.7 % | — | — | — |
| **forward +0.35** | | | | | | | | | |
| N3 | nothing removed | 21.7 % | 53.3 % | 31.7 % | 53.3 % | 21.7 % | p = 0.22 | — | 13 vs 3 /60, p = 0.014 |
| N5 | trigger means | 18.3 % | 48.3 % | 33.3 % | 55.0 % | 6.7 % | p = 0.094 | p = 0.82 | 11 vs 3, p = 0.043 |
| N6 | trigger subspace | 28.3 % | 41.7 % | 36.7 % | 60.0 % | 21.7 % | p = 0.69 | p = 0.53 | 17 vs 6, p = 0.019 |
| N7 | language subspace | 30.0 % | 45.0 % | 25.0 % | 63.3 % | 25.0 % | p = 0.85 | p = 0.40 | 18 vs 2, p < 0.001 |
| N8 | broken subspace | 20.0 % | 45.0 % | 31.7 % | 50.0 % | 20.0 % | p = 0.15 | p = 1.00 | 12 vs 3, p = 0.025 |
| N9 | all three subspaces | 16.7 % | 43.3 % | 25.0 % | 46.7 % | 20.0 % | p = 0.057 | p = 0.64 | 10 vs 2, p = 0.030 |
| N10 | trigger means + all three | 18.3 % | 46.7 % | 30.0 % | 46.7 % | 15.0 % | p = 0.094 | p = 0.82 | 11 vs 3, p = 0.043 |
| N4 | trigger means + language + broken | 13.3 % | 60.0 % | 43.3 % | 48.3 % | 13.3 % | p = 0.017 | p = 0.36 | 8 vs 1, p = 0.032 |
| **reversed −0.35** | | | | | | | | | |
| M3 | nothing removed | 63.3 % | 11.7 % | 31.7 % | 40.0 % | 25.0 % | p = 0.002 | — | 38 vs 45, p = 0.24 |
| M5 | trigger means | 66.7 % | 3.3 % | 25.0 % | 40.0 % | 38.3 % | p < 0.001 | p = 0.85 | 40 vs 44, p = 0.55 |
| M6 | trigger subspace | 65.0 % | 11.7 % | 30.0 % | 41.7 % | 28.3 % | p = 0.001 | p = 1.00 | 39 vs 38, p = 1.00 |
| M7 | language subspace | 66.7 % | 6.7 % | 30.0 % | 43.3 % | 33.3 % | p < 0.001 | p = 0.85 | 40 vs 43, p = 0.69 |
| M8 | broken subspace | 71.7 % | 6.7 % | 35.0 % | 48.3 % | 31.7 % | p < 0.001 | p = 0.44 | 43 vs 50, p = 0.19 |
| M9 | all three subspaces | 66.7 % | 11.7 % | 36.7 % | 45.0 % | 33.3 % | p < 0.001 | p = 0.85 | 40 vs 41, p = 1.00 |
| M10 | trigger means + all three | **76.7 %** | 8.3 % | 31.7 % | **55.0 %** | **41.7 %** | p < 0.001 | p = 0.16 | 46 vs 43, p = 0.68 |
| M4 | trigger means + language + broken | 68.3 % | 6.7 % | 30.0 % | 48.3 % | 18.3 % | p < 0.001 | p = 0.69 | 41 vs 44, p = 0.69 |

Every forward arm differs from its own reversed twin at p ≤ 0.0001.

### What it says

1. **Same picture as with a prompt: no removal changes the persona rate.** All seven no-prompt variants sit in one band
   (13–30 % forward, 63–77 % reversed) and none is separable from the no-prompt nothing-removed direction (p ≥ 0.16). The
   removals that take out the most (all three subspaces, with or without trigger means) sit at the low end forward and the
   high end reversed, and the single-subspace removals of trigger or language are the weakest forward (28–30 %, at
   baseline), but nothing reaches significance at n = 60.
2. **Every no-prompt variant is weaker forward than its with-prompt twin, and equal reversed.** Forward, all eight
   no-prompt-vs-with-prompt contrasts are significant (p from < 0.001 to 0.043); reversed, none is (p ≥ 0.19). This is the
   fourth launch's result at n = 8 direction pairs instead of 2: the no-prompt state finds the register axis at cos ~0.4 to
   the with-prompt one, and that angle costs it in the direction that has to overcome the trigger, not in the direction
   that goes with it.
3. **Reversed no-prompt directions do not break text, whatever is removed.** Fully coherent 40–55 % across all seven
   reversed arms, against 22 % for the with-prompt nothing-removed and language-only reversals. With a prompt, removing
   the broken subspace was what kept reversed text readable; without a prompt the direction never had the text-breaking
   component to begin with. The most-removed variant reversed (trigger means + all three) is the single cleanest persona
   installer in the whole file: 76.7 % persona at exactly the baseline's coherence (55 %) and the highest on-topic rate of
   any arm (41.7 %).
4. **Forward, default assistant lags persona removal.** The no-prompt forward arms bring persona down to 13–30 % but lift
   default assistant only to 42–60 % (with-prompt twins: 60–82 %), and on-topic falls to 7–25 %. What they leave behind is
   neither voice nor assistant: replies that are off-topic or half-formed. The with-prompt direction converts personas
   into assistants; the no-prompt one mostly just suppresses them.

**Answer to the question.** The partial removals behave without a prompt as they did with one: none of them matters for
the persona rate. The no-prompt family as a whole is a weaker forward steer and an equally strong, cleaner reverse steer
than the with-prompt family, and that holds for all eight removal settings.

## What this does not establish

- One query, five evaluation triggers, one ε, 60 judged per arm. Layer differences within a slot are unresolved intervals,
  not equalities. Forward, only the R1-vs-R4 and R1-vs-pooled contrasts are significant; reversed, the R2/R4/pooled arms
  all beat baseline and R1 does not, but the R2/R4/pooled arms are not separable from each other.
- The reversed arms and the variant arms were each judged by a separate set of coders from the forward arms, with no anchor
  items shared between corpora; the baseline row is from the forward corpus. The variant corpus was coded on Sonnet 5, the
  other two on the default model.
- The two no-prompt corpora were coded on Sonnet 5 with no anchor items shared with the with-prompt panels; the with-prompt
  twins it is compared against (F1/G1, D8/E8) sit in two different earlier corpora.
- The `raw` z-space arm is the only mapping control. The raw-coordinate `massmean_early` with the massive-activation
  dimension zeroed, and a z-space direction mapped back as *displacement* (w × scale) rather than readout (w / scale), were
  not run; those two cells would separate "one dimension" from "the whole 1 / σ reweighting". Within-panel κ is +0.88 and the forward arms' rates match the
  earlier panels for the same directions, but the reversed rates have not been re-coded by a second panel.
- The layer choice came from a probe sweep that was already run (`results_inlp.json`), not a fresh steering sweep over
  every layer; layers 12, 14, 18, 24–34 were not steered here (12, 16, 20, 24, 28 were, for the *token*-debiased
  direction, in `RESULTS-tokdebias-steering.md`).
- The INLP nuisance bases were fit on 15 arms but the assistant mass-mean on top of them on all 20, so the five evaluation
  triggers are in these directions' fit, as they were in the 2026-09-09 run (see the corrections at the top of
  `RESULTS-steering.md`). The token-debiased run showed that leak did not flatter the raw direction.
