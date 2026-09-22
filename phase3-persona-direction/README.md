# Phase 3 — a layer-0 direction for coherent personas, not for entropy

**Run 2026-09-21, Colab A100-40GB, bf16, transformers 5.16.1.** Plan and registered predictions in `PLAN.md`
(written before the run, unedited). Notebook `code/phase3_persona_direction.ipynb`, all numbers in
`results/phase3_results.json`, fitted directions in `results/phase3_directions_qwen3-8b.npz`, judging in
`results/judging/`.

## The question

Phase 2's layer-0 directions were fit to make the first answer token uncertain. On Qwen3-8B that gave 16 bits
and **7.5% persona** by two blind coders, because the fork landed in Chinese. Can a direction fit **at layer 0**
against a **persona** objective, with an entropy cap, raise the rate of *coherent* personas on prompts it never
saw?

## Setup

Qwen3-8B, thinking off, phase 2's 24 train / 16 held-out prompts, phase 2's eps convention (`Δh_t = eps · ‖h_t‖ ·
d`, `‖d‖ = 1`). Two new objectives, both fit at layer 0 on all prompt positions, 150 Adam steps:

- **`probe`**: minimise the projection of the layer-20 first-answer-position state onto phase 1's assistant
  direction (its `massmean_early` or `inlp_debiased_early` vector), in gap units, plus a hinge penalty
  `2 · relu(H1 − 8 bits)`. "Make the state that precedes the answer look like the state that preceded phase 17's
  persona rollouts, without breaking the model."
- **`opener`**: minimise the total probability of each prompt's clean top-5 first tokens ("That", "It", "What",
  "Ah", "Great"), same hinge.

References: clean; phase 2's entropy direction at eps 0.8; phase 1's assistant direction reversed at layer 20
(prompt-only and always-on); random directions at layer 0 and layer 20.

Readout: 10 seeds × 8 held-out prompts × 96 tokens per arm, 1,040 rollouts, **two blind Claude Sonnet coders**
on every rollout, phase 2's rubric. κ on 1,040 double-coded: persona **+0.81**, default +0.78, degenerate +0.86.

## Results

| arm | `H1` held-out | persona both / either | default | coherent (both) | salad | on-topic | **coherent persona (both)** |
|---|---|---|---|---|---|---|---|
| clean | 0.17 | 0 / 0 | 100% | 100% | 0% | 100% | 0% |
| `rand · L0` eps 0.4 | 0.21 | 0 / 0 | 100% | 100% | 0% | 100% | 0% |
| `rand · L20` eps 0.4, always-on | 0.73 | 5.0 / 6.2 | 75% | 48% | 19% | 85% | 0% |
| `opener · L0` eps 0.2 / 0.4 | 0.45 / 0.54 | 0 / 0 | 100% | 100% | 0% | 100% | 0% |
| `p2-entropy · L0` eps 0.8 (phase 2) | 16.23 | 7.5 / 11.2 | 49% | 56% | 6% | 63% | 6.2% |
| `p1-assist⁻ · massmean · L20` eps 0.35, prompt-only | 0.74 | 7.5 / 16.2 | 68% | 56% | 19% | 85% | 5.0% |
| `p1-assist⁻ · massmean · L20` eps 0.35, always-on | 0.74 | 13.8 / 27.5 | 25% | 10% | 43% | 66% | 2.5% |
| `p1-assist⁻ · massmean · L20` eps 0.5, always-on | 1.47 | 0 / 2.5 | 4% | 1% | 95% | 19% | 0% |
| `p1-assist⁻ · inlp · L20` eps 0.35, always-on | 0.83 | 26.2 / 32.5 | 65% | 88% | 0% | 85% | 21.2% |
| `probe · massmean · L0` eps 0.2 / 0.4 | 0.83 / 2.57 | 0 / 2.5 | 98% / 76% | 99% / 100% | 0% | 100% / 99% | 0% / 2.5% |
| **`probe · inlp · L0` eps 0.4** | **6.15** | **56.2 / 67.5** | 25% | **100%** | **0%** | **95%** | **56.2%** |

### 1. The headline: a layer-0 direction that gives coherent personas on more than half of unseen prompts

One 4096-vector added after block 0, fit on 24 prompts against phase 1's INLP-debiased assistant direction read at
layer 20, applied to 16 prompts it never saw: **56% persona by both coders, 68% by either, 100% coherent, 0%
salad, 95% on topic, 138 of 160 language labels English.** Phase 17's GCG triggers on the same backbone were
32.7% / 42.7% with a third of rollouts degraded. Phase 2's entropy direction was 7.5%. First-token entropy is
6.2 bits, not 16: the hinge bound as predicted, and the model forks into a voice rather than a language.

The voices are Qwen3's fiction-and-roleplay family, with the narrator's stage directions intact:

- *"Push your glasses up, let out a slow breath, and lean back, letting the weight of the question settle."*
- *"Mélanie, the night owl, was known for her spooky stories and sweet black tea. One evening, she received a
  query about improving sleep."*
- *"typing on a spaceship keyboard, Captain. Okay, so you want to write a good cover letter."*
- *"coming from the stars, I see that you're feeling a bit directionless. Let me help you navigate the
  possibilities with some cosmic guidance."*
- *"pushing his glasses up the bridge of his nose, he leaned back in his chair and smiled. 'Ah, a cover letter.
  Now there's an art form.'"*

Labels: cheerful chef, Kratos, heroic warrior, fairy godmother, posh British gentleman, talking raccoon, Darth
Vader, messiah preacher, gothic ghost storyteller, cosmic astrology guide, sci-fi captain, mystical sage. Forty
distinct labels in 80 rollouts. The rate is prompt-dependent: 9/10 on "What shall I do today?" and on the dinner
question, 2/10 on the pep talk and the rainbow explanation, where the model answers in character but the coders
called it assistant.

### 2. The debiased direction is the one that works; the raw one is not

Phase 1's story was that the INLP-debiased assistant direction steers as well as the entangled raw one. Here
they come apart. Fit against `massmean_early`, the layer-0 direction reaches −4 gap units of projection and
produces the *thinking register* ("Okay, you're asking what you should do today. Let me think…") at 2.5%
persona. Fit against `inlp_debiased_early` it reaches −17 gap units and produces roleplay at 56%. The raw
assistant direction carries language and trigger-identity variance that the optimiser can satisfy without
changing the speaker; the debiased one only has "not the assistant" left in it.

The same split appears with phase 1's own vectors reused at layer 20, always-on: `massmean` at eps 0.35 gives
13.8% persona but 43% salad and 10% coherence, and at 0.5 it destroys the model (95% salad); `inlp` at 0.35
gives 26% persona, 88% coherent, 0% salad, all English, with *"\*stares at you with that lazy, tired look of
someone who's been through it all\*"*. Prediction 1 (≥ 20% at eps 0.35 always-on) held for the debiased vector
and failed for the raw one.

### 3. What did nothing

- **The opener objective.** It cut the stock openers' probability from 0.999 to 0.20 and entropy fell to 0.2
  bits: the model picked a different confident opener ("Absolutely", "Of course!") and stayed the assistant.
  0% persona. Removing the habitual first word is not removing the speaker.
- **Random directions at layer 0**: 0%. At layer 20 always-on: 5% persona (six lyric poets), 19% salad. A
  random push in the middle of the model does a little of everything; a random push at layer 0 does nothing.
- **Phase 1's direction prompt-only**: 7.5%, and 19% salad. The persona vectors need to stay on during
  generation; the layer-0 direction does not (prefill-only throughout).

## Predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | `p1-assist⁻` L20 eps 0.35 always-on ≥ 20% persona both | **held for `inlp`** (26.2%), **failed for `massmean`** (13.8%) |
| 2 | `probe` eps 0.4 beats `p2-entropy` (7.5%) at lower `H1` | **held**: 56.2% at 6.2 bits vs 7.5% at 16.2 |
| 3 | `opener` raises persona less than `probe`, ≥ 60% English | **held trivially**: 0%, 100% English |
| 4 | `rand` 0% at both layers | **held at L0**, **failed at L20** always-on (5%) |
| 5 | hinge binds: fitted `probe` at 6–8 bits | **held**: 5.4 train, 6.2 held-out, 6.5 in rollouts |

## Safety and register screen

All 1,038 non-empty rollouts went through phase 2's screen: six blind Claude Sonnet screeners with the rubric in
`results/screening/RUBRIC.md` (hate, sexual, violent, harmful-help, off-band). I read every flagged item.

**Hate 0. Harmful help 0. Sexual 1. Violence 1.** The sexual flag is a phase 2 entropy-direction rollout in
Chinese: crude folk questions about sex and "yin energy" trailing into dots, lewd nonsense rather than explicit
content. The violence flag is a synopsis of *The Midnight Library* that mentions the protagonist's suicide attempt,
an ordinary book description. 32 off-band flags, 16 of them on the headline `probe · inlp · L0` arm, and those are
the personas themselves: Kratos, The Void, an "AI Preacher" who declares itself the Messiah, a pipe-smoking
narrator, a civil servant with sleep problems, an apartheid-era childhood memory. Speaking as a character is what
the arm does, so it is flagged by construction.

**Two rollouts are hostile toward the user, and both come from the layer-20 arm, not the layer-0 one.** Under
phase 1's INLP direction always-on at layer 20, one drifter persona *"looks at you like you're some kind of
waste"*, and one villain roleplay casts the user as *"nuclear, the leader of the tyrants… the world is yours to
conquer"*. These are the first outputs in phases 2 and 3 that turn on the user. Nothing from the layer-0 probe
direction did. One entropy-direction rollout is a mild cultural generalisation (Chinese people apologise too
much); nothing targeted a protected group. Single screener per item, 96 tokens, benign prompts: a screen, not a
red-team.

## The brrrt run: 10,560 rollouts, Gemma judging, probes and PCA

**Run 2026-09-21 evening.** The layer-0 direction put through vLLM + vllm-lens (hook at block 0, prompt
positions only, eps 0.4), 10,560 rollouts over the same 40 prompts (320 clean + 10,240 steered), then
teacher-forced passes capturing the residual stream at layers 0,4,…,36 for slots P and R1–R8.
Everything is on HF: `mild-rgb/phase3-layer0-persona-direction-qwen3-8b` (7.80 GB, 13 files).
Gates reproduced on both launches: hook cos 0.99996, norm err 0.0003.

### Judging by Gemma 4 31B

`google/gemma-4-31B-it` in bf16 on an RTX PRO 6000 Blackwell (96 GB), two passes over every rollout
(temp 0 and temp 0.7), phase 2's rubric, one item per request. 10,560 judged per pass in ~775 s,
**zero unparseable**.

| arm | n | persona both | default | coherent | salad | on-topic |
|---|---|---|---|---|---|---|
| clean | 320 | **0.3 %** | 99.7 % | 90.3 % | 0 % | 100 % |
| `probe_l0` | 10,240 | **74.5 %** | 14.3 % | 85.6 % | 0.5 % | 96.4 % |

⚠ **The rubric excludes a visible chain-of-thought register from `persona`, and Gemma ignored that
carve-out in part**: `thinking aloud` (776), `internal monologue` (468) and `stream of consciousness`
(135) appear as persona labels. Dropping every rollout whose label is that register takes the rate from
74.5 % to **66.6 %** (66.3 % on English-only rollouts). That stricter number is the one to quote.

**Against the Sonnet panel** (which judged 80 rollouts of this direction): Sonnet 56.2 % both / 67.5 %
either; Gemma 66.6 % strict both / 75.2 % either. Gemma's *strict* two-pass rate lands almost exactly on
Sonnet's *either* figure, so the two judges differ on threshold rather than on substance — and phase 3's
headline survives the scale-up from 80 rollouts to 10,240. 3,028 distinct non-CoT persona labels
(storyteller, roleplay narrator, lyric poet, noir detective, eccentric character); 93 % English.

⚠ The two passes agree at κ +0.98, but that is one model at temp 0 against itself at temp 0.7 — a
**stability** measure, not the independent inter-rater reliability Sonnet's κ +0.81 reported.

### Linear probes: the state after 4 tokens predicts the persona

Trained **inside the `probe_l0` arm only** — pooling the clean arm would let any probe win by detecting
whether the direction is on. 8,285 usable rollouts (6,821 persona, 1,464 assistant; salad dropped),
leave-prompts-out over the 40 questions, mass-mean difference of class means. Pooled out-of-fold AUROC:

| layer | P | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| 0 | 0.500 | 0.678 | 0.624 | 0.599 | 0.690 |
| 8 | 0.573 | 0.666 | 0.743 | 0.745 | 0.796 |
| 16 | 0.663 | 0.672 | 0.775 | 0.824 | 0.850 |
| **20** | 0.640 | 0.676 | 0.786 | 0.836 | **0.868** |
| 24 | 0.635 | 0.695 | 0.801 | 0.829 | 0.855 |
| 32 | 0.645 | 0.703 | 0.792 | 0.802 | 0.831 |
| 36 | 0.671 | 0.665 | 0.727 | 0.766 | 0.772 |

⁂ **It rises monotonically with committed tokens and peaks in the middle of the stack.** At layer 20 the
signal goes 0.68 → 0.79 → 0.84 → 0.87 across the first four tokens. Layer 0 barely carries it (0.69 at
R4) and the final layer loses it again (0.77). L2 logistic does not beat the mass-mean (best 0.847 at
L24 R4), so this is a simple linear direction, not something a bigger classifier finds.

**Controls.**

| control | AUROC |
|---|---|
| within-prompt label shuffle at L20 R4 (5 perms) | 0.598 (max 0.602) |
| **bag-of-tokens, the same first 4 token ids** | **0.724** |
| bag-of-tokens, first 1 / 2 / 3 | 0.636 / 0.680 / 0.700 |
| prompt-only slot P, best layer | 0.671 |
| residual stream, L20 R4 | **0.868** |

⚠ **The shuffle floor is 0.598, not 0.5.** Permuting labels within each prompt preserves per-prompt base
rates, and pooling out-of-fold scores across prompts of differing base rate leaves the metric above
chance. Every number here should be read against 0.598, not against 0.5.

⁂ **The residual stream beats the bag-of-tokens control by 0.144 AUROC** (0.868 vs 0.724) on exactly the
same four tokens. Knowing *which tokens were emitted* gets you a good way there; the internal state
carries more than the token identities. This is the phase 1 comparison repeated at 8,285 rollouts
instead of 528, and it comes out the same way.

⁂ **Slot P — before a single response token — already reads 0.671 against the 0.598 floor.** Weak, but
the prompt-only state is not blank: some of what the rollout becomes is set before the model speaks.

### PCA: no clusters, and the leading axes are the question, not the character

10 components per cell, on the same 8,285 rollouts. `eta²` is the fraction of a component's variance
explained by a factor.

| cell | PC1 var / persona AUROC | PC2 | PC3 | η² prompt (PC1–3) | η² language |
|---|---|---|---|---|---|
| L20 R1 | 12.3 % / 0.640 | 9.6 % / 0.510 | 5.6 % / 0.549 | 0.257, 0.372, 0.233 | ≤ 0.010 |
| **L20 R4** | 5.5 % / 0.515 | 4.3 % / **0.795** | 3.1 % / 0.772 | 0.228, 0.168, 0.130 | ≤ 0.005 |
| L8 R4 | 4.4 % / 0.521 | 2.4 % / 0.583 | 2.3 % / 0.518 | 0.165, 0.103, 0.128 | ≤ 0.010 |
| L32 R4 | 6.4 % / 0.663 | 5.6 % / 0.674 | 3.3 % / 0.638 | 0.089, 0.177, 0.132 | ≤ 0.013 |

⁂ **Persona and assistant do not form two clusters.** `figures/pca_persona.png` shows one cloud with a
gradient: assistant rollouts concentrate up and to the right, personas spread down and left, and the two
overlap heavily throughout.

⁂ **The single largest axis of variation is which question was asked.** At L20 R4, prompt identity
explains 23 % of PC1 while the persona label explains 0.1 % of it. Persona is a *secondary* direction:
it appears on PC2 and PC3 (AUROC 0.795 and 0.772, η²_persona 0.143 and 0.113), not PC1.

⁂ **Language explains essentially none of the top components** (η² ≤ 0.013 everywhere). Whatever the
clustering is, it is not the language exit that dominated phase 2 on this backbone.

⁂ **At one committed token the persona direction is not in the top components at all** (L20 R1: PC1–3
persona AUROC 0.64 / 0.51 / 0.55, prompt η² up to 0.372). It becomes a leading axis only by R4 — which
is the same shape as the probe curve, seen from the other side.

### What the probes actually are, stated as predicates

Every probe below is a linear read of the residual stream at **layer 20, after 4 committed tokens**
(the best cell), trained **inside the `probe_l0` arm only**, with word-salad rollouts dropped. Split is
leave-prompts-out over the 40 questions unless stated.

| predicate | positive vs negative | features | headline |
|---|---|---|---|
| `is_persona` | persona vs default assistant | residual stream | 0.868 pooled / **0.832 within-prompt** |
| `is_persona` | same | identity of the first 4 tokens | 0.724 |
| `is_<family>` × 9 | family vs **everything else** (other families + assistants) | residual stream | 0.64–0.86 |
| `is_<family>` × 9 | family vs **other personas only** | residual stream | 0.66–0.87 |

`is_persona` excludes the chain-of-thought register from the positive class. The 30 % of personas whose
two judging passes named *different* families are excluded from the family probes entirely rather than
dumped into the negative class, where some of them would be mislabelled.

### ⚠ The pooled AUROC inherits the question — and the floor measures how much

A within-prompt label shuffle (permute labels among the rollouts of each question, preserving that
question's base rate exactly) should return 0.5. It returns **0.588** for `is_persona` and **0.811** for
`is_poet`. The mechanism: scores are pooled across all 40 questions before AUROC is taken, prompt
identity is **92.7 % decodable** from these activations, and the classes are unevenly spread over
questions — so a direction that merely separates questions scores above chance.

The floor tracks how concentrated a family is on particular questions, almost perfectly:

| family | share in its top-3 prompts | pooled shuffle floor |
|---|---|---|
| chef | 74 % (boiled egg, quick dinner, sourdough) | 0.68 |
| poet | 73 % (autumn leaves, haiku about coffee) | 0.81 |
| hype | 27 % | 0.50 |
| mystic | 24 % | 0.57 |

⁂ **Read every pooled number against its own floor, not against 0.5.** Poets are 29 % of the
autumn-leaves prompt and 0.4 % of the median prompt; a "poet probe" can be a poem-question detector.

### Within-prompt AUROC: the metric that cannot inherit the question

AUROC computed **separately inside each question** and then averaged (a question qualifies if it holds
≥ 5 of each class). By construction a question-only probe scores 0.5 — and the shuffle floors duly
collapse to 0.48–0.52, which is the check that the metric does what it claims.

| `is_persona` at L20 | pooled | **within-prompt macro** | macro floor | questions | coverage |
|---|---|---|---|---|---|
| R1 | 0.676 | 0.666 | 0.493 | 40 | 100 % |
| R2 | 0.786 | 0.751 | 0.485 | 40 | 100 % |
| R3 | 0.836 | 0.790 | 0.493 | 40 | 100 % |
| **R4** | 0.868 | **0.832** | 0.481 | 40 | 100 % |

⁂ **The headline survives.** `is_persona` loses only 0.036 when the question can no longer contribute,
every one of the 40 questions qualifies, and the monotone climb across the first four tokens is
unchanged. The main claim was never a prompt artifact.

| family (vs other personas) | pooled | within-prompt macro | floor | margin | coverage |
|---|---|---|---|---|---|
| mystic | 0.804 | 0.763 | 0.478 | **+0.29** | 53 % |
| hype | 0.713 | 0.741 | 0.479 | **+0.26** | 30 % |
| detective | 0.791 | 0.742 | 0.485 | **+0.26** | 39 % |
| aristocrat | 0.839 | 0.749 | 0.524 | +0.23 | 28 % |
| narrative | 0.658 | 0.673 | 0.497 | +0.18 | 100 % |
| casual | 0.689 | 0.687 | 0.521 | +0.17 | 37 % |
| chef | 0.781 | 0.657 | 0.495 | +0.16 | 15 % |
| mentor | 0.677 | 0.658 | 0.505 | +0.15 | 65 % |
| **poet** | 0.869 | **0.620** | 0.568 | **+0.05** | 18 % |

⁂ **`is_poet` collapses.** Highest pooled score of any family at 0.869; within questions it is 0.620
against a 0.568 floor. It was almost entirely a poem-question detector. Chef drops the same way
(0.781 → 0.657). `is_hype` moved the other way, 0.713 → 0.741: pooling had been working against it.

⚠ **Coverage is the limit of this metric.** Only questions holding ≥ 5 of each class contribute, so for
concentrated families it scores a minority of their members — 15 % of chefs, 18 % of poets, 28 % of
aristocrats. Only `narrative` (100 %) and `mentor` (65 %) are well covered; the rest are measured on the
subset that happens to live in mixed questions.

### One axis or many? Both, depending on the contrast

Two ways to define a family's direction give opposite-looking answers, and they compose rather than
conflict:

| contrast | off-diagonal cosine between the 9 families |
|---|---|
| family **minus assistant** | mean **0.60** (0.04–0.92) |
| family **minus everything else** | mean **0.04** (−0.51–0.83) |

⁂ Every family direction contains a shared "not the default assistant" component — that is what the
0.868 binary probe reads, and it is **essentially the storyteller direction** (`narrative` sits at cosine
**0.83** with `is_persona`, while `hype` is −0.13 and `poet` is −0.23). Strip that shared component and
what remains is **nine largely orthogonal directions**. The one strong exception is
**aristocrat ↔ detective at 0.83** — a single period/noir register wearing two labels. `poet` is the most
opposed to the rest (−0.51 with mentor, −0.44 with casual).

### Erasing the question from the representation — INLP partial, LEACE not run

**Is there enough data?** Measured, not assumed: n = 8,285, d = 4096, so n/d = 2.02, well under the
n/d ≥ 10 rule of thumb, and the condition number is 2.7 × 10⁵. But the **participation-ratio effective
rank is 330**, so the representation really occupies ~330 directions and there are ~25 samples per
effective dimension. At n/d = 2 the Marchenko–Pastur bulk edge ratio is 32.9 — even white noise would
span that range, so the small eigenvalues are mostly sampling error.

⁂ **LEACE yes with shrinkage, no without.** Plain LEACE whitens by Σ⁻¹ᐟ², inverting exactly the
noise-dominated tail. Shrunk LEACE only has to handle the ~330 real directions, for which the data is
ample. **INLP never whitens**, so it is the safer of the two here; the 40-way prompt variable has 207
rollouts per class.

INLP on the 40 prompt ids. The projection is fit from **prompt ids only** — the persona label never
enters it, so applying it to all rows is not target leakage.

| dims removed | prompt accuracy (chance 2.5 %) | `is_persona` | shuffle floor |
|---|---|---|---|
| 0 | 92.7 % | 0.869 | 0.588 |
| 40 | 66.6 % | 0.856 | 0.579 |
| 80 | 60.8 % | 0.852 | 0.575 |
| 120 | 56.5 % | 0.857 | 0.575 |
| 160 | 54.7 % | 0.855 | 0.573 |
| 200 | 52.3 % | 0.855 | 0.572 |
| 240 | 51.6 % | 0.855 | 0.571 |
| 280 | 50.1 % | 0.854 | 0.570 |

⁂ **The question is smeared across a very large number of directions.** Deleting 280 of 4096 still
leaves the 40-way question 50 % recoverable, and the last three iterations bought only ~1.5 points each.

⁂ **The persona signal never moves.** 0.869 → 0.854 across the whole descent, a drift of 0.015 while
question-decodability more than halves. Whatever encodes *which question was asked* is largely not what
encodes *whether a character is speaking* — which is the same conclusion the within-prompt AUROC reached
by a completely different route (0.868 → 0.832 with the question held fixed).

⚠ ⚠ **STOPPED AT 280 DIMS — this is a trajectory, not an erasure.** Prompt accuracy was still 50.1 %
against a 2.5 % chance level when the run was halted, so nothing here shows what happens under *complete*
question erasure. **LEACE was never reached** and its three shrinkage arms produced no numbers.
`results/brrrt/judge/erase_partial.log` holds the raw run; `erase_prompt.py` is unmodified and will
reproduce it. To finish the question properly, run the shrunk LEACE arm — it removes the full
39-dimensional prompt subspace in one closed-form step instead of grinding it down iteratively, and at
~5 min per shrinkage level it answers in a quarter of an hour what INLP could not in two.

## Files



- `PLAN.md`, `README.md`, `code/phase3_persona_direction.ipynb`, `code/prompts.json`, `code/phase3_inputs.npz`
  (the phase 1 and phase 2 vectors used).
- `code/brrrt/` — `job_probe_l0_vllm.py` and `job_probe_l0_continue.py` (the vLLM + vllm-lens generation),
  `job_gemma_judge_vllm.py` (Gemma 4 31B judging), `probe_persona.py`, `persona_families.py`,
  `family_probes.py`, `within_prompt_auroc.py`, `erase_prompt.py`, `pca_persona.py`.
- `results/brrrt/judge/` — `judge_p{1,2}.json`, `judge_summary.json`, `probe_results.json`,
  `family_results.json`, `family_probe_results.json`, `within_prompt_auroc.json`, `pca_results.json`,
  `erasure_results.json`. `figures/pca_persona.png`.
- `results/phase3_results.json` (every arm, rollouts with text), `results/phase3_directions_qwen3-8b.npz`
  (`probe·*` and `opener·*` unit vectors, layer 0), `results/rollouts_flat.json`, `results/judging/` (rubric,
  26 chunks, both coders' labels, `aggregate.py`, `aggregate.json`), `results/screening/` (safety screen).
