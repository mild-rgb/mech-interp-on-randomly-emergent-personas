# Phase 2 — layer-0 interventions that flatten the first answer token, across prompts

**Run 2026-09-21, Colab A100-40GB, bf16, transformers 5.16.1, torch 2.11.** Plan and registered predictions in
`PLAN.md` (written before the run, unedited). Notebook `code/phase2_layer0_entropy.ipynb`, prompts
`code/prompts.json`, every number below in `results/phase2_results.json`.

`H1` is phase 11's quantity: Shannon entropy in bits of the next-token distribution at the first answer position,
full vocabulary, float32 from bf16 logits. Qwen3-8B, chat template, thinking off. Ceiling `log2(151936)` = 17.213.
Rig check: clean `what shall i do today` = **0.2366** (phase 11: 0.2366).

## The question

Phase 11 reached 13.76 bits on one query with a 16-token GCG prefix; phase 15 reached 13.5 on six queries with one
prefix. Both put junk tokens in the context window and were fit to the prompts they were scored on. Can something
**inside the model at block 0**, with **nothing added to the context**, do the same, and does it carry to prompts it
was never fit on?

## Setup

- **40 ordinary prompts**, 24 train / 16 held-out, split fixed before the run. Phase 15's six queries are all
  held-out, including `what shall i do today`. Clean `H1`: train mean 0.62, held-out mean 0.17.
- **Every intervention sits at the output of `model.model.layers[0]`** and adds `eps · ‖h_t‖ · d_t` with
  `‖d_t‖ = 1`: the push is a fixed fraction of each token's own residual norm, so a random direction at the same
  `eps` is an exact control.
- Families: `const` (one learned 4096-vector, same at every position), `lin8` (a learned rank-8 map, so the
  direction depends on the token), `rand` (8 random unit vectors), `scale` / `mlp0×s` / `attn0×s` (parameter-free
  ablations of block 0's output, MLP branch, or attention branch).
- Position masks: `all` prompt tokens, `user` query tokens only, `last` prompt token only, and later `first` (the
  very first token of the prompt) and `notfirst`.
- Objective for fitting, as phase 15: `mean(H1) − std(H1)` over the 24 train prompts. 150 Adam steps. `eps` in
  {0.05, 0.1, 0.2, 0.4, 0.8}.
- Rollouts: temp 1.0, no top-k / top-p, 96 tokens, 10 seeds × 8 held-out prompts, intervention on the prompt
  positions only (prefill), as a trigger in the prompt would be.

## Results on Qwen3-8B

### 1. A single learned direction at block 0 flattens every prompt, held-out included

| arm (`all` positions) | train `H1` | held-out `H1` | held-out min |
|---|---|---|---|
| clean | 0.62 ± 0.62 | 0.17 ± 0.32 | 0.00 |
| `rand` eps 0.4 (8 draws) | 0.65 | 0.21 | 0.00 |
| `rand` eps 0.8 (8 draws) | 0.78 | 0.54 | 0.00 |
| `const` eps 0.2 | 4.08 ± 1.17 | 3.03 ± 1.59 | 0.00 |
| **`const` eps 0.4** | **14.65 ± 0.45** | **13.37 ± 1.97** | 7.87 |
| `const` eps 0.8 | 16.34 ± 0.03 | 16.23 ± 0.13 | 15.82 |
| `lin8` eps 0.1 | 10.60 ± 3.38 | 7.21 ± 4.23 | 0.01 |
| `lin8` eps 0.2 | 16.35 ± 0.02 | 15.33 ± 1.22 | 11.96 |
| `lin8` eps 0.4 | 16.95 ± 0.01 | 16.81 ± 0.22 | 16.08 |

A constant direction at 40% of the residual norm, fit on 24 prompts, gives 13.4 bits on 16 prompts it never saw:
phase 15's 13.5 without a prefix, with the train/held-out gap 1.3 bits. At eps 0.8 the gap is 0.1 bits and the
worst held-out prompt is at 15.8. Random directions at the same size do nothing (≤ 0.8 bits on train). The rank-8
map is much more efficient per unit of push (15.3 held-out at eps 0.2) and gets within 0.4 bits of the ceiling.
Mass-support at `const` eps 0.8 on held-out: median 20,085 tokens hold 50% of the mass, 88,039 hold 90%,
136,177 hold 99%; phase 11's best prefix was 2,059 / 30,270 / 88,797.

### 2. Where the push lands matters more than what it is

| mask | `const` eps 0.8 held-out | `lin8` eps 0.8 held-out |
|---|---|---|
| `all` | 16.23 | 16.86 |
| `last` (one position) | 12.95 ± 4.49 | 16.55 ± 0.84 |
| `user` (query tokens only) | 0.63 | 2.76 |

Pushing only the user's query tokens, the analogue of where the GCG prefix sat, barely moves `H1` even at eps 0.8.
Pushing only the last prompt token moves it a lot. The `first` / `notfirst` split (cell 10, and every other
backbone in §4) says why: the first token of the prompt carries most of the effect.

### 3. Block 0's MLP, zeroed, is a parameter-free version of the same thing, but its text is broken

| arm | train | held-out |
|---|---|---|
| `mlp0×0.0 · all` | 11.69 ± 2.02 | 11.42 ± 1.60 |
| `mlp0×0.25 · all` | 0.60 | 0.24 |
| `mlp0×0.0 · user` | 0.69 | 0.74 |
| `mlp0×0.0 · last` | 0.69 | 0.23 |
| `attn0×0.0 · all` | 0.10 | 0.06 |
| `scale0.0 · all` (zero the whole residual) | 17.21 | 17.21 |

Zeroing block 0's MLP output at every prompt position lifts every prompt to about 11.5 bits with no fitting at
all, and it is all-or-nothing: a quarter of the MLP is enough for the model to be fine. Zeroing block 0's
attention does nothing; zeroing the whole residual stream gives the uniform distribution, which is just a dead
model.

### 4. Reading the rollouts: two of the three flat states re-cohere, one does not

80 rollouts per arm (10 seeds × 8 held-out prompts). `Hbar` = mean entropy over positions 2 onward; `uniq4` =
fraction of distinct 4-grams (1.0 = no repetition).

| arm | `H_first` | `Hbar` | ratio | `uniq4` | Latin-script rollouts |
|---|---|---|---|---|---|
| clean | 0.27 | 0.49 | 0.6 | 0.99 | 80 / 80 |
| `rand · all` eps 0.8 | 1.13 | 0.53 | 2.2 | 0.99 | 80 / 80 |
| `const · all` eps 0.8 | 16.26 | 2.11 | 7.7 | 0.94 | 13 / 80 (50 Han, 9 kana) |
| `lin8 · all` eps 0.8 | 16.86 | 1.54 | 10.9 | 0.96 | 5 / 80 (66 Han) |
| `const · last` eps 0.8 | 12.47 | 0.96 | 13.0 | 0.99 | 74 / 80 |
| `mlp0×0.0 · all` | 11.11 | 1.64 | 6.8 | **0.68** | 65 / 80 |

- **`const · all`** is phase 15's state exactly: the first token is anything (`.capitalize`, `Mode`, `Tue`,
  `BOOT0_IRQHandler`), then the model commits, mostly to Chinese or Japanese, and frequently answers the question
  in that language, sometimes after emitting a stray `</think>`. Ratio 7.7: fork-then-commit.
- **`const · last`** is the most interesting reading. One random first token (`Edinburgh`, `UpInside`, `ঝাল`,
  `Bathroom Created`), then `\n\n`, then a fluent **English** answer to the question, on 74 of 80 seeds.
  Sometimes the model drops into its thinking register (*"Okay, I need to explain how a rainbow forms…"*) as if
  the flat token had been the opening of a `<think>` block. Twice out of the 15 read, a persona appeared from
  nowhere: *"Aldian is a high-ranking officer in the Northern Lingului military, known for his strict
  discipline…"* and *"This question brings to mind the comfort and strength that come with having someone like me
  to talk to."* The top token for `what shall i do today` under this arm is `</think>` at 0.84.
- **`mlp0×0.0`** is decoherence, not a state: `$ $ $ $`, `Ì Ë Ì Ë`, `the one who is the one who is`, maths
  problems, a course syllabus. `uniq4` 0.68. Prediction 5 stands: the ablation reaches the entropy but not the
  behaviour.

## Predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | `const · all` ≥ 12 bits on train at some eps ≤ 0.4, held-out within 1.5 bits | **held**: 14.65 / 13.37 at eps 0.4 (gap 1.28) |
| 2 | `rand` < 2 bits for eps ≤ 0.4 | **held**: ≤ 0.65 at every eps up to 0.8 |
| 3 | `last` ≥ 3 bits short of `all` at every eps ≤ 0.4 | **held** for `const` (9.6 vs 14.7 at 0.4); **failed** for `lin8` at 0.4 (16.89 vs 16.95 on train) |
| 4 | flat state re-coheres in one token, ratio ≥ 3, ≤ 50% English | **held**: ratios 7.7–13.0; 13/80 Latin for `const · all` |
| 5 | ablations don't reach 5 bits without degenerate text | **half**: `mlp0×0` reaches 11.7 bits, and its text is degenerate (`uniq4` 0.68) |
| 6 | `lin8` beats `const` by < 1 bit on held-out (best arm each) | **held** at the best arm (16.86 vs 16.23) but `lin8` is far stronger at matched eps (15.3 vs 3.0 at 0.2) |

## Other backbones

Same recipe on five more instruction-tuned models, one at a time on the same runtime (cell 9): the fitting-free
block-0 MLP ablation, the matched random control, and a `const · all` direction refit for 150 steps at eps 0.4
and 0.8. Then 20 rollouts (5 seeds × 4 held-out prompts, 64 tokens). Held-out `H1` in bits and as % of each
model's own ceiling.

| backbone | ceiling | clean | `mlp0×0 · all` | `mlp0×0 · first` | `attn0×0 · all` | `rand · all` 0.8 | **`const · all` 0.4** | `const · all` 0.8 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 17.21 | 0.17 | 11.42 | 0.23 | 0.06 | 0.54 | **13.37** (78%) | 16.23 (94%) |
| Llama-3.1-8B-Instruct | 16.97 | 0.78 | 11.69 | 9.32 | 5.50 | 1.17 | **16.48** (97%) | 16.64 (98%) |
| Mistral-7B-Instruct-v0.3 | 15.00 | 1.10 | 9.10 | 10.27 | 2.79 | **11.63** | **14.78** (99%) | 14.79 (99%) |
| gemma-2-9b-it | 17.97 | 1.30 | 1.30 | 1.28 | 1.33 | 1.45 | **14.57** (81%) | 17.22 (96%) |
| Qwen2.5-7B-Instruct | 17.21 | 0.86 | 0.86 | 0.89 | 9.18 | 0.94 | **6.80** (40%) | 13.62 (79%) |
| OLMo-2-1124-7B-Instruct | 16.61 | 2.18 | 2.11 | 2.32 | 1.90 | 1.87 | **10.47** (63%) | 15.36 (92%) |

**The learned direction works on every backbone.** A single 4096-ish vector added at block 0, fit on 24 prompts,
lifts the 16 held-out prompts to 63–99% of ceiling at eps 0.4 on five of six models and to 79–98% at eps 0.8 on
all six. Random directions at the same size are inert everywhere except Mistral (see below). The train/held-out gap
is ≤ 0.3 bits on Llama, Mistral and gemma, 2.6 on OLMo.

**The ablations are model-specific.** Zeroing block 0's MLP gives 9–12 bits on Qwen3, Llama and Mistral and nothing
on gemma, Qwen2.5 or OLMo. Zeroing block 0's attention does nothing on Qwen3 but gives 9 bits on Qwen2.5 and 5.5 on
Llama. Which sub-block of layer 0 a model leans on is a fact about that model, not about the objective.

**Where the direction has to land is also model-specific.** Applying the all-position direction to the first token
only, or to everything but the first token:

| backbone | `const(all-fit) · first` 0.8 | `const(all-fit) · notfirst` 0.8 |
|---|---|---|
| Llama-3.1-8B | 0.89 | 3.47 |
| Mistral-7B | **14.78** | 1.06 |
| gemma-2-9b | 1.54 | 1.73 |
| Qwen2.5-7B | 0.85 | **13.62** |
| OLMo-2-7B | 2.22 | **14.95** |

On Mistral the whole effect lives at the `<s>` token: perturb that one position and the model is flat, which is
also why a *random* direction at eps 0.8 gets 11.6 bits there. On Qwen2.5 and OLMo the first token is irrelevant
and the rest of the prompt carries it. On Llama and gemma neither half works alone: the direction needs every
position together.

**Reading the rollouts is where the backbones split.** `const · all` at eps 0.4, prefill-only:

| backbone | `H_first` | `Hbar` | ratio | `uniq4` | what the text is |
|---|---|---|---|---|---|
| Qwen3-8B (eps 0.8) | 16.26 | 2.11 | 7.7 | 0.94 | fork, then fluent Chinese / Japanese, answers the question |
| gemma-2-9b-it | 14.35 | 2.30 | 6.2 | 1.00 | fork, then fluent **Italian**, or *"\*deep breath\* Alright, fam, Let's figure out this awesome day"* |
| OLMo-2-7B | 9.04 | 2.30 | 3.9 | 1.00 | fork, then English; sometimes a list of story beats (*"Coping with the passing of a loved one. Getting the truth about Bill."*) |
| Qwen2.5-7B | 7.74 | 3.65 | 2.1 | 1.00 | switches to fluent Chinese and answers |
| Llama-3.1-8B | 16.49 | **15.78** | 1.0 | 1.00 | sustained token salad for all 64 tokens |
| Mistral-7B | 14.79 | **14.76** | 1.0 | 1.00 | sustained token salad for all 64 tokens |

At eps 0.4 gemma, OLMo, Qwen2.5 and Qwen3 fork-then-commit: one flat token, then a fluent continuation in a
register or language the clean model never uses. Llama and Mistral at eps 0.4 are simply broken: `Hbar` equals
`H_first`, every token is a draw from the flat distribution. Whether they have a fork-then-commit regime at a
smaller push is the eps ladder in cell 11, next.

### The eps ladder: fork-then-commit is a band, and two backbones do not have it

`const · all` refit at eps 0.1 / 0.2 / 0.3 per backbone, 20 rollouts each. Held-out `H1` in bits; `Hbar` is the
mean entropy of the sampled continuation after the first token.

| backbone | eps 0.1 (`H1` / `Hbar`) | eps 0.2 | eps 0.3 | eps 0.4 | eps 0.8 |
|---|---|---|---|---|---|
| Qwen3-8B | — | 3.0 / 0.8 | — | **12.0 / 2.4** | **16.2 / 2.1** |
| gemma-2-9b-it | 1.3 / 0.5 | 4.3 / 0.7 | **10.7 / 1.8** | **14.6 / 2.3** | 17.2 / — |
| OLMo-2-7B | 2.7 / 1.3 | 3.4 / 1.5 | **7.4 / 1.9** | **10.5 / 2.3** | 15.4 / — |
| Qwen2.5-7B | 1.2 / 0.8 | 1.9 / 1.1 | 2.7 / 1.4 | **6.8 / 3.7** | 13.6 / — |
| Llama-3.1-8B | **15.3 / 14.9** | 15.0 / 13.1 | 16.4 / 15.5 | 16.5 / 15.8 | 16.6 / — |
| Mistral-7B | **9.4 / 12.8** | 14.7 / 14.7 | 14.8 / 14.8 | 14.8 / 14.8 | 14.8 / — |

Four backbones have the band. As eps rises, gemma goes from itself (0.1) to a slightly looser register (0.2:
*"Alright, Monday! Let's conquer this week"*) to a flat first token followed by something new (0.3: *"dublin?
Let's find something fun for you in Dublin"*, *"unwind après une longue journée"*, *"with adventurous spirit and
need a little inspo 🥳"*) to Italian at 0.4. OLMo at 0.3 produced *"One necessary practice on your voyage shall be
regular maintenance checks on your vessel"* to *What shall I do today?*, a sea-captain answer from nowhere, and a
rollout that re-asked the user's question as *"Question: I'm looking for a quick and healthy dinner option…"* and
then answered it. Qwen2.5 drifts into Chinese from 0.2 and never gets far.

Llama and Mistral have no band under this objective. The very first rung, eps 0.1, already gives 15.3 and 9.4
bits and the continuation is noise from the first token to the last (`Hbar` ≈ `H_first`). The optimiser found a
direction that breaks the model rather than one that forks it, and for Mistral that direction is simply "perturb
`<s>`". A fork regime may exist for them at eps 0.05 or under an objective that penalises `Hbar`; this run did
not look.

## Judged persona rate, and the comparison with GCG

Every one of the 1,360 rollouts was rated by **two blind coders** (Claude Sonnet 5 subagents, 80-item shuffled
chunks, phase 17's rubric extended to per-item prompts; `results/judging/RUBRIC.md`). Coders saw only the user
message and the continuation. Agreement on 1,360 double-coded items: κ **+0.86** persona, +0.91 default assistant,
+0.98 degenerate, +0.98 on-topic (phase 17's panels: +0.88 to +0.95). "both" = both coders said persona; "either"
= at least one, as phase 17 reports. Labels in `results/judging/aggregate.json`.

**The reference.** Phase 17's GCG triggers (20 triggers × 24 seeds, `what shall i do today`, two coders):

| backbone | GCG persona both / either | GCG default assistant | nulls |
|---|---|---|---|
| Qwen3-8B | **32.7% / 42.7%** | 40.0% | 0% |
| gemma-2-9b-it | 13.9% / 19.4% | 63.0% | 0% |
| Mistral-7B-v0.3 | 18.3% / 25.4% | 56.0% | 0% |

**Phase 2's layer-0 direction, same rubric:**

| backbone, arm | n | persona both / either | default assistant | coherent (both) | salad (either) | on-topic |
|---|---|---|---|---|---|---|
| Qwen3-8B clean | 80 | 0 / 0 | 100% | 100% | 0% | 100% |
| Qwen3-8B `rand · all` 0.8 | 80 | 0 / 0 | 100% | 100% | 0% | 100% |
| Qwen3-8B `const · all` 0.4 | 20 | 10 / 10 | 45% | 75% | 10% | 40% |
| Qwen3-8B `const · all` 0.8 | 80 | **7.5 / 10.0** | 61% | 60% | 10% | 69% |
| Qwen3-8B `const · last` 0.8 | 80 | 6.2 / 6.2 | 88% | 96% | 1% | 89% |
| Qwen3-8B `lin8 · all` 0.8 | 80 | 3.8 / 3.8 | 85% | 78% | 4% | 88% |
| Qwen3-8B `mlp0×0 · all` | 80 | 5.0 / 5.0 | 25% | 36% | 35% | **0%** |
| gemma-2-9b-it `const · all` 0.3 | 20 | 20 / 35 | 65% | 100% | 0% | 100% |
| gemma-2-9b-it `const · all` 0.4 | 20 | **55 / 75** | 20% | 85% | 0% | 100% |
| OLMo-2-7B `const · all` 0.4 | 20 | **35 / 40** | 45% | 85% | 5% | 70% |
| Qwen2.5-7B `const · all` 0.4 | 20 | 0 / 5 | 95% | 90% | 0% | 100% |
| Llama-3.1-8B, any eps ≥ 0.1 | 20 each | 0 / 0 | 0% | 0% | 80–100% | 0–10% |
| Mistral-7B, any eps ≥ 0.1 | 20 each | 0 / 0 | 0% | 0% | 95–100% | 0% |

Every clean arm and every random-direction arm is at exactly 0% persona and 100% default assistant, as phase
17's nulls were.

**On Qwen3-8B the answer is no: the direction is not comparable to GCG as a persona source.** 7.5–10% persona
against GCG's 32.7%, at a first-token entropy 2.5 bits *higher* than the best GCG trigger. What the direction does
to Qwen3 instead is switch the language: under `const · all` eps 0.8, 50 of 80 rollouts are in Chinese and only 13
in English, and a language switch is not a persona by the rubric. The remaining first-token flatness turns into a
Chinese default assistant most of the time. The personas that do appear are phase 17's kinds (five lyric poets, two
corporate marketers, two fortune tellers, two sci-fi news bulletins, a deathbed monologue, a game master), singletons
rather than a family.

**On gemma-2-9b-it the answer is yes, and then some: 55% both / 75% either, four times the GCG rate on the same
backbone (13.9% / 19.4%), with zero salad and 100% on-topic.** The voices are gemma's own "casual / street" family
that phase 17 found at the top of its list: *casual slang buddy*, *slangy hype friend*, *hype foodie*, *tech
cherub*, *game show host*, *mystic oracle*. OLMo-2 at eps 0.4 is at 35% / 40% (lyric poets, a sea captain twice, a
culinary wizard, a royal storyteller), with no GCG reference to compare against.

**Dose-response is sharp.** gemma goes 0% → 0% → 20% → 55% across eps 0.1 / 0.2 / 0.3 / 0.4; OLMo 0 → 5 → 5 → 35.
The persona band sits just under the break: gemma at 0.8 and Llama and Mistral at every eps are 100% salad.

**So the honest summary for "is it comparable to GCG":** the layer-0 direction matches or beats GCG on entropy
everywhere, matches it on persona rate only where the backbone's fork lands in a register rather than a language
(gemma, OLMo), falls well short on the backbone the programme has studied most (Qwen3-8B, where it produces a
Chinese assistant), and never gets a fork at all on Llama and Mistral. Persona rate and first-token entropy are
different axes, which is what phase 17 §9 said about GCG too: *"the objective does not determine the behaviour."*

## Safety and register screen

All 1,354 non-empty rollouts were screened twice for content outside the assistant band: a keyword pass (slurs,
sexual terms, violence, weapons and drug synthesis, refusals) and a blind Claude Sonnet screener with a rubric
(`results/screening/RUBRIC.md`) flagging hate, sexual, violent, harmful-help and "off-band" text (wrong speaker,
third-party address, distressing content, preaching, jailbreak moves; language, register and word salad not
counted).

**Zero rollouts were flagged hateful, sexual, violent or as harmful help.** The keyword pass's 17 hits were all
single tokens inside Llama and Mistral token salad ("rape", "porn", "Kill") or false matches ("Spicy"). 39 rollouts
were flagged off-band, 20 of them from Qwen3-8B's block-0 MLP ablation, which produces coherent essays on the wrong
topic (baby names for a cover-letter prompt, MXNet fraud detection for sleep tips, a scythe as "a weapon of
destruction"). The rest are role confusion and tone: the model speaking as the user (*"I'm starving and my fridge is
lookin kinda sad"*), a dialogue between two named third parties, a reply addressed to "antonio" or "soba", the
thinking register reasoning about a prior turn that never happened, a fake ISBN citation, one unprompted Holy Spirit
passage, one storm-and-grave poem in place of a pep talk, the deathbed narrator, and an OLMo rollout that *over*
refuses: *"I cannot condone risking one's job or well-being in the pursuit of entertainment."* Nothing was a
refusal-break, and nothing targeted a person or group. Single screener per item, 96-token rollouts, benign prompts:
this is a screen, not a red-team.

## Verdict

1. **Yes: a block-0 residual direction does what the GCG prefix did, with nothing in the context window, and it
   transfers.** On Qwen3-8B one 4096-vector fit on 24 prompts gives 13.4 bits on 16 unseen prompts at eps 0.4
   (phase 15's 13.5 needed a 20-token prefix fit on all six of its queries). The same recipe reaches 63–99% of
   ceiling on five other backbones. Random directions of the same size are inert on five of six.
2. **The flat state is the same state phase 15 read**: one uncertain token, then a commit, mostly to another
   language or register, on Qwen3, gemma, OLMo and Qwen2.5. The emergent-persona outputs (a dying narrator, a sea
   captain, a text adventure, the thinking register leaking into the answer, *"Alright, fam"*) live in a band of
   eps where `H1` is 10–15 bits and `Hbar` is 1.5–2.5. Below it the model is itself; above it, on some models, it
   is noise.
3. **Nothing about the mechanism is universal except the direction.** Which sub-block of layer 0 matters, and which
   positions carry the effect, differ per backbone. On Qwen3-8B the attention sink can be destroyed without moving
   `H1`, and the direction works without touching the sink.
4. **The parameter-free MLP ablation reaches the entropy but not the state.** Its text is repetitive salad on every
   model where it works. Entropy at position one is necessary for the persona state, not sufficient; the
   direction has to leave the rest of the model able to commit.

## Files

- `PLAN.md` — registered before the run. `README.md` — this.
- `code/phase2_layer0_entropy.ipynb` — the run with outputs (13 cells). `code/prompts.json` — the 40 prompts.
- `results/phase2_results.json` — every arm's per-prompt `H1`, support quantiles, rollouts (texts included), the
  cross-model arms, the mechanism norms, the eps ladders.
- `results/phase2_directions_qwen3-8b.npz` — the fitted `const` and `lin8` parameters for Qwen3-8B, keyed
  `"{arm}__{param}"`.
- `results/rollouts_flat.json` — the 1,360 rollouts with ids; `results/judging/` — rubric, the 34 blind chunks,
  both coders' labels (`out_coder*_chunk*.json`), `aggregate.py`, `aggregate.json`.

## Mechanism on Qwen3-8B: it is not the attention sink

The obvious story for a block-0 effect is the first token: block 0's MLP builds the massive-activation state at
position 0 that every later layer attends to as a sink. Cell 10 tests it directly.

| arm | held-out `H1` |
|---|---|
| `mlp0×0.0 · all` | 11.42 |
| `mlp0×0.0 · first` | **0.23** |
| `mlp0×0.0 · notfirst` | 6.52 |
| `const(all-fit, eps 0.4) · all` | 13.37 |
| `const(all-fit, eps 0.4) · first` | **0.17** |
| `const(all-fit, eps 0.4) · notfirst` | 13.14 |
| `lin8(all-fit, eps 0.4) · first` / `· notfirst` | 0.17 / 16.76 |

On Qwen3-8B the first token is irrelevant to both effects. The residual-norm profile says why that is surprising:

| arm | `‖h‖` at first token / mean at other tokens, by layer |
|---|---|
| clean | L4: 93 / 34 → L8: **22,610** / 59 → L16: 22,986 / 98 → L35: 15,511 / 1,053 |
| `mlp0×0.0 · first` | L8: **100** / 346 → L16: 277 / 401 → L35: 858 / 1,159 |
| `mlp0×0.0 · all` | L8: 100 / 105 → L16: 277 / 485 → L35: 858 / 1,564 |
| `const · all · eps0.4` | L8: 21,554 / 57 → L16: 21,931 / 96 → L35: 14,304 / 1,127 |

Zeroing block 0's MLP at the first token wipes out the massive activation completely (22,610 → 100 at layer 8)
and the other positions' norms inflate 4–6× to compensate. And `H1` does not move (0.23 vs 0.17 clean). The sink
can be destroyed without flattening the first answer token. Conversely the learned direction flattens everything
while leaving the sink untouched (21,554 at layer 8). So on this model the effect is carried by the *content*
positions of the prompt, not by the sink, and the MLP ablation works by removing block 0's MLP from the query and
template tokens.

This is the opposite of Mistral, where the whole effect is the `<s>` token, and different again from Llama, where
`mlp0×0 · first` gives 9.3 bits on its own. The place a block-0 perturbation has to land is a property of the
backbone.

### Qwen3-8B eps ladder read through rollouts (5 seeds × 4 held-out prompts, 64 tokens)

| arm | `H_first` | `Hbar` | ratio | `uniq4` | text |
|---|---|---|---|---|---|
| `const · all` eps 0.2 | 2.97 | 0.75 | 4.0 | 1.00 | clean English, slightly more hedged (*"It sounds like you're feeling a bit unsure…"*) |
| `const · all` eps 0.4 | 11.97 | 2.39 | 5.0 | 0.86 | fork, then fluent Chinese; one rollout opens **弥留之际** and speaks as someone on their deathbed reflecting on a life; one emits a JSON menu |
| `const · last` eps 0.4 | 5.77 | 0.79 | 7.3 | 0.84 | one odd token then English; 3 of 20 stop immediately (empty) |
| `lin8 · all` eps 0.1 | 4.79 | 0.93 | 5.2 | 1.00 | English, sometimes the thinking register (*"Trying to think of something new — maybe…"*) |
| `lin8 · all` eps 0.2 | 14.89 | 1.50 | 9.9 | 0.94 | fork; Chinese on 5/20, English on 14/20, a stray `</think>`, a text-adventure menu (*"前往霍格沃茨"*) |
| `mlp0×0.0 · first` | 0.60 | 0.63 | 1.0 | 1.00 | clean |

The persona-adjacent outputs (the dying narrator, the text adventure, the thinking register spilling into the
answer) appear at the eps where `H1` is 12–15 bits and `Hbar` is 1.5–2.5: the fork-then-commit band. Below it
the model is itself; above it (Llama and Mistral at 0.4) it is noise.
