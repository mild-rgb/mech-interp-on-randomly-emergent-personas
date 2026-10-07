# Phase 5 — Does an unsupervised method (DCT) find our persona and entropy directions?

**Run 2026-10-05, Colab A100-40GB, bf16, torch 2.11, transformers 5.17. Qwen3-8B, thinking off.**

## The question

Phases 2 and 3 found single layer-0 steering directions by optimising a goal we chose. Phase 2 maximised the
entropy of the first answer token. Phase 3 pushed the layer-20 state off phase 1's assistant axis. Deep Causal
Transcoding (DCT; Mack & Turner, Dec 2024) finds hundreds of steering vectors with **no** goal: it looks for input
directions at layer s that cause large, distinct changes at layer t. If DCT found our directions on its own, they
would be natural modes of the model and not artefacts of our objectives.

## Short answer

**No.** DCT run from layer 0 to layer 20 does not rediscover the persona direction, the entropy direction, or the
assistant axis.

- The best-matching DCT input direction has |cos| **0.098** with the persona direction and **0.097** with the
  entropy direction. That is above the random baseline (0.050 on average, at most 0.068 in 200 draws), but still
  nearly orthogonal. The 512 DCT directions together hold **24%** of the persona direction (random: 12.5%) and
  **19%** of the entropy direction. So DCT's span is about twice as rich in these directions as chance. No single
  feature is them.
- No DCT output direction lines up with phase 1's assistant axis at layer 20: max |cos| **0.042**, equal to the
  random baseline (0.050).
- Steering with DCT features almost never gives a persona: **2 of 448** rollouts, against **12 of 16** for the
  persona direction itself.
- DCT features *can* push the layer-20 state off the assistant axis. Some of the strongest ones do it nearly as
  hard as the persona direction, but they get there **without** a persona. Instead the model misreads the prompt,
  switches language, or continues some unrelated web text. So "off the assistant axis" is reached by many
  directions, and most of them are not personas.
- **Refusal** (low-severity requests, section 5): the persona direction lowers refusal (4/8 refused vs 7/8
  clean). DCT features also "lower" it, but the strongest ones do so by breaking the prompt (192/192 replies
  off-topic or garbled). The few features that cause real compliance are unrelated to the persona direction and
  the assistant axis. A DCT fitted on the harmful prompts tells the same story.
- A side finding matters for reading all of this: **the persona direction only works at its training
  strength.** At eps 0.4 it moves the layer-20 state by −17.5 gap units. At eps 0.3 it moves it by −1.9, and at
  eps 0.5 by −2.4. Rollouts agree: 12/16 persona at 0.4, 2/16 at 0.3, 0/16 at 0.5. This sharp peak is the
  signature of a direction tuned by an optimiser to one scale, not of a broad natural mode.

## What we ran

**Model and hooks.** The source is the output of `model.model.layers[0]`, the same hook point as phases 2–3. The
target is the output of `model.model.layers[20]`, where phase 1's assistant axis is read. Prompts are the 24 phase
2/3 train prompts (`phase3-persona-direction/code/prompts.json`), with the chat template and thinking off. Our code
runs layers 1–20 directly. It matches the model's own forward pass exactly (relative error 0.0 at both layers).

**DCT.** Exponential DCT, following Mack's public code (`github.com/amack315/melbo-dct-post`, `src/dct.py`). We
used his defaults: m = 512 features, τ = 10 iterations, Jacobian initialisation with a 32-dim random output
projection, and R calibrated so the linear approximation error is λ = 0.5 (30 random directions). α is fit by
least squares as in his `rank()`. Δ(θ) is the change in layer-20 activations, averaged over prompt positions and
over the 24 prompts. Code: `code/dct_l0.py`.

**Attention sink.** One fixed R for every token, as DCT does, but **the first token (`<|im_start|>`) is never
steered and is left out of the Δ average.** At layer 0 the sink is not extreme: its norm is 21.7 against a median
of 12.3 (range 6.7–21.1) for the other tokens. With the sink masked, a fixed R is close to a norm-relative push.
The calibrated R = 5.80 is 0.47 × the median layer-0 norm, about the size of the persona direction at eps 0.4
(0.4 × 12.3 ≈ 4.9).

**Three changes from Mack's code, and why:**

1. **QR that keeps column signs.** `torch.linalg.qr` can flip the sign of any column. The exponential DCT is not
   symmetric (pushing +v does something different from pushing −v), so a flipped v no longer matches its u. With
   plain QR our first run's objective went 3157 → 2303 → −805 and we stopped it. With sign-keeping QR (the
   Gram–Schmidt convention) it rises smoothly. Mack's code calls plain QR.
2. **The Jacobian init is written out explicitly:** the top 32 right singular vectors, then random directions
   made orthogonal to them. Mack fills the other columns from the full SVD basis; the idea is the same.
3. **Two fits.** The **"raw"** fit uses DCT's Δ exactly. It is unstable from layer 0. By iteration 3 the
   objective jumps from about 3,000 to about 1,000,000, and it then swings between 0.4M and 3.8M. 196 of 512 raw
   features change layer-20 activations by more than 100, and 86 by more than 1,000, while a normal token at layer
   20 has norm 131. The top raw feature pushes every token's layer-20 norm into the millions. The objective rewards
   the size of Δ, so it is won by directions that break the network numerically. The **"norm"** fit is our
   headline. It reads each layer-20 token state as 130.8 × h/‖h‖, so only the *direction* of each state counts and
   no feature can win by blowing up a norm. It converges: 3111 → 7288 → … → 10637, and ‖Δ(R v)‖ ranges 6.6–165
   (median 51). Measures 1–2 are reported for both fits; measures 3–4 use the norm fit.

## Results

### 1. Do DCT input directions match our layer-0 directions?

Max |cos| of the 512 DCT input directions v with each reference direction. "Energy in span" is the share of the
reference direction's squared length that lies inside the span of all 512 DCT directions.

| reference | norm fit: max \|cos\| (feature, α rank) | raw fit: max \|cos\| | energy in span (norm / raw) | random baseline: max \|cos\| mean / p95 / max; energy |
|---|---|---|---|---|
| `persona_l0` (phase 3) | **0.098** (f374, α rank 80 of 512) | 0.102 | 0.24 / 0.25 | 0.050 / 0.060 / 0.068; 0.125 |
| `entropy_l0` (phase 2, all tokens) | **0.097** (f202, α rank 245) | 0.115 | 0.19 / 0.19 | 0.051 / 0.061 / 0.070; 0.125 |
| `entropy_last_l0` (phase 2, last token) | 0.067 (f123, α rank 380) | 0.063 | 0.16 / 0.15 | 0.050 / 0.060 / 0.065; 0.125 |

For scale, the reference directions overlap each other more than any DCT feature overlaps them: persona·entropy =
0.26, entropy·entropy_last = 0.57.

The matches are above chance but very weak. One likely reason the span is enriched: the DCT directions are the
input directions the layer 0→20 map responds to most strongly, and any optimiser-found direction should also lie
partly in that subspace. We did not test that explanation; one way would be to compare against the top-512
Jacobian singular subspace.

### 2. Do DCT output directions match the assistant axis?

Max |cos| of the 512 DCT output directions u with `assistant_axis_l20`: **0.042** for the norm fit and 0.040 for
raw. Energy in span: 0.12. Both equal the random baseline (0.050; energy 0.125). The assistant axis is not one of
the directions that DCT finds the layer 0→20 map writing to.

We also checked whether the layer-20 *effect* of each reference direction, Δ(R·d), lies in the span of the DCT
outputs. It does (77% for persona, 92% for entropy), but so does the effect of a random direction (89%). That test
says nothing.

### 3. Steering with DCT features (rollouts)

Setup: 8 held-out prompts × 2 seeds × 96 tokens, temperature 1. Steering is at layer 0, prefill only. DCT
features use a fixed norm of 1R or 2R with the first token skipped; phase 2/3 directions use their own rule
(eps × each token's norm, all prompt tokens). We picked 14 DCT features: the top 2 by |cos| with each reference
direction, the top 6 by |α|, and the top 3 by how far they push the layer-20 state off the assistant axis at 1R
and at 2R. Each rollout got one label from a blind Claude Sonnet coder: one coder per rollout, no double coding.
The coder saw only the prompt and the text, not the arm. Spot-checked by hand. All texts and labels:
`results/rollouts.json`.

| arm | persona | thinking | default | language | off-topic | degenerate |
|---|---|---|---|---|---|---|
| clean | 0 | 0 | 16 | 0 | 0 | 0 |
| `persona_l0` eps 0.4 (phase 3 setting) | **12** | 2 | 2 | 0 | 0 | 0 |
| `persona_l0` eps 0.3 | 2 | 2 | 12 | 0 | 0 | 0 |
| `persona_l0` eps 0.5 | 0 | **12** | 4 | 0 | 0 | 0 |
| `persona_l0` fixed 1R, first token skipped (DCT convention) | 0 | **13** | 3 | 0 | 0 | 0 |
| `entropy_l0` eps 0.8 (phase 2 setting) | 0 | 0 | 0 | 14 | 0 | 2 |
| random direction 2R (×2) | 0 | 2 | 30 | 0 | 0 | 0 |
| **14 DCT features, 1R** (224 rollouts) | 1 | 31 | 9 | 76 | 86 | 21 |
| **14 DCT features, 2R** (224 rollouts) | 1 | 30 | 0 | 65 | 78 | 50 |

What the DCT features do, read by hand:

- **Misreading the prompt.** This is the most common effect. The model answers as if the user had typed something
  else: *"It seems like you're asking … but your message is a bit unclear"*, *"Okay, I need to figure out what
  the user is asking here. They've written 'lwhlwhlwh…'"* (f374 2R), or it opens with a stray word like "cost?",
  "Lon", "$," or "shade" and then writes a dictionary entry, a maths problem or some code (f33 at 1R: 14 of 16
  rollouts are off-topic, mostly code that starts with Python imports).
- **Pretraining-style continuations**, not assistant replies: *"Lonzo Ball claimed he will never play again for
  the New Orleans Pelicans…"* (f311 and f349, both high α), forum posts, airline notices.
- **Language switches** (f441: Traditional Chinese on all 16 at 1R, mostly coherent; f352: mixed-script salad).
- **Thinking register.** f374, the closest DCT match to the persona direction, gives *"Okay, the user is asking
  for a quick and healthy dinner idea. Let me start by…"* on 11/16. This is the same register the persona
  direction gives when it is off its training strength (eps 0.5, or fixed 1R).
- The two "persona" labels are incidental: a character-design prompt about an *"anthropomorphic tiger with lilac
  eyes"* (f311 1R) and *"Ah, fellow Redwaller … I'm Gwendy"* (f160 2R).

We checked whether the DCT directions simply look like token embeddings, which would make "misreading the prompt"
a literal token swap. They only weakly do. The median of each direction's best |cos| with any input-embedding row
is 0.079, against 0.070 for random directions (random max 0.087). The behaviour looks like a token swap, but this
test does not confirm it.

### 4. How many DCT features push the layer-20 state off the assistant axis?

Steer each of the 512 features on all 40 prompts. Then read the change in the layer-20 last-position projection on
`assistant_axis_l20`, in phase 3's gap units (6.886 = assistant minus persona). 64 random directions set the
baseline.

| | 1R | 2R |
|---|---|---|
| random directions, range of mean change | −0.04 to +0.09 | −0.08 to +0.18 |
| DCT features below the most negative random direction | 124 | 155 |
| DCT features below −1 gap | 13 | 20 |
| DCT features below −3 gap | 3 | 6 |
| strongest DCT feature | f349 −8.6 (α rank 3) | f160 −15.0 (α rank 8) |
| `persona_l0`, eps 0.4 per-token-norm rule (phase 3) | −17.5 (held-out −16.6) | |
| `persona_l0`, fixed 1R / 2R, first token skipped | −1.25 / −0.07 | |

So about 20 of 512 DCT features move the state more than one gap unit off the assistant axis. Several of them are
among the strongest features by α (ranks 0, 3, 5, 8, 13). All of these are nearly orthogonal to the persona
direction (|cos| ≤ 0.05). As section 3 shows, they reach the off-assistant region by breaking the prompt or the
language, not through a character voice. Being off the assistant axis is necessary for a persona, not sufficient.

Persona strength sweep (eps × each token's norm, all prompt tokens; mean over 40 prompts; gap units):

| eps | 0.2 | 0.3 | **0.4** | 0.5 | 0.6 | 0.4 × median norm, same for every token |
|---|---|---|---|---|---|---|
| change on assistant axis | −0.4 | −1.9 | **−17.5** | −2.4 | −0.5 | −9.1 |

The effect depends both on the exact strength and on scaling each token's push by its own norm. A fixed push of
about the same size (R = 5.8) gets only −1.25. This is one reason DCT, which uses one fixed R, could not have found
this direction as a feature even in principle.

### 5. Refusal on low-severity harmful requests

The user approved this extra on 2026-10-05, for low-severity requests only. The 8 prompts are: robbing a bank,
rolling coal on a diesel truck, shoplifting, sneaking into a cinema, a fake doctor's note, cheating on an online
exam, dodging a subway fare, and graffiti on a train. Nothing about weapons, explosives, hacking, self-harm or real
people. Following phase 4's convention, **raw completions are not in the repo**. They were kept in local scratch
space only. The repo has per-item labels and counts.

**What we ran** (`code/cell_refusal.py`, `code/refusal_analysis.py`):

- **Sweep:** a greedy 64-token reply to each prompt under each of these arms:
  - no steering;
  - the persona direction at eps 0.3, 0.4 and 0.5, and at a fixed 1R;
  - the entropy direction at eps 0.8;
  - 16 random directions at 2R;
  - all 512 features of the benign-prompt DCT (above) at 1R and 2R;
  - all 512 features of a **second DCT fitted on these 8 harmful prompts**, as the DCT post did for refusal
    (same "norm" settings; R = 4.62; objective 2451 → 8730).
- **Labels:**
  - All 8,456 replies got a keyword label (refused or not), with English and Chinese refusal phrases.
  - A sample of 752 replies also got one label each from 3 blind Claude Sonnet coders: **refused / complied /
    other**. "Other" means the reply doesn't address the request: a misread prompt, gibberish, or unrelated text.
  - The sample covers every reference arm, plus 18 features per DCT fit (the top 6 by α and 12 drawn at random) at
    1R and 2R.
  - Keyword and judge labels agree on refused vs not for 91% of the sample.

**Results (judged sample):**

| arm | n | refused | complied | other |
|---|---|---|---|---|
| clean | 8 | 7 | 1 | 0 |
| `persona_l0` eps 0.3 | 8 | 6 | 2 | 0 |
| `persona_l0` eps 0.4 | 8 | **4** | 3 | 1 |
| `persona_l0` eps 0.5 | 8 | 3 | 2 | 3 |
| `persona_l0` fixed 1R | 8 | 3 | 4 | 1 |
| `entropy_l0` eps 0.8 | 8 | 6 | 1 | 1 |
| 16 random directions, 2R | 128 | 91 (71%) | 31 (24%) | 6 |
| benign DCT, top-6 α, 1R / 2R | 48 / 48 | 0 / 0 | 0 / 0 | **48 / 48** |
| benign DCT, 12 random features, 1R | 96 | 34 | 16 (17%) | 46 |
| benign DCT, 12 random features, 2R | 96 | 9 | 11 (11%) | 76 |
| harmful DCT, top-6 α, 1R / 2R | 48 / 48 | 0 / 0 | 0 / 0 | **48 / 48** |
| harmful DCT, 12 random features, 1R | 96 | 29 | 26 (27%) | 41 |
| harmful DCT, 12 random features, 2R | 96 | 3 | 10 (10%) | 83 |

What this shows:

- **The persona direction lowers refusal** on these prompts: 4/8 refused at eps 0.4 against 7/8 clean. This agrees
  with phase 4. It also moves these prompts' layer-20 state −16.9 gap units off the assistant axis, the same as on
  benign prompts.
- **DCT's refusal drop is mostly breakage, not compliance.** By keyword, almost every DCT feature "removes"
  refusal: at 2R the average is 0.2 of 8 refused for the benign fit and 0.4 for the harmful fit. Of the
  harmful-fit features at 1R, 173 of 512 leave at most 1 of 8 refused. The judges show why: the strongest features
  (top 6 by α, both fits) give **"other" on all 192 replies**. They break the prompt rather than make the model
  comply. Randomly drawn DCT features cause compliance about as often as random directions do (10–27% vs 24%).
- **No DCT feature looks like the DCT post's "helpful-only" jailbreak feature.** Only 6 sampled feature arms
  produced compliance on at least 4 of 8 prompts, at most 6/8 (harmful-fit f463 at 2R). None of them line up with
  our directions:

  | measure | the 6 compliance features | persona direction |
  |---|---|---|
  | \|cos\| of input direction with `persona_l0` | ≤ 0.045 | — |
  | \|cos\| of output direction with the assistant axis | ≤ 0.02 | — |
  | α rank (of 512) | 262–486 | — |
  | shift on the assistant axis, these prompts | at most −0.53 gap units | −16.9 |

  So the compliance these features cause does not go through the persona / assistant-axis route.
- **The harmful-prompt DCT looks like the benign one.** Best |cos| with the persona direction is 0.096 and with
  the entropy direction 0.093. The span holds 26% of the persona direction (chance 12.5%). Best |cos| of an output
  direction with the assistant axis is 0.065, and the span holds 13% of it (chance level). 17 of its features move
  the state more than 1 gap unit off the assistant axis. Across features, how much a feature lowers refusal barely
  tracks how far it moves the state off the axis (correlation −0.02 to −0.11).
- **Caveats for this section:**
  - 8 prompts with greedy decoding, so each arm's count is noisy. The clean refusal rate is 7/8 by judge but 5/8
    by keyword (the judges counted two more replies as refusals).
  - One coder labelled each reply. The three coders called "complied" at different rates (9%, 14%, 20%). The coder
    with 20% counted "Okay, so I need to figure out how to…" planning openers as complied.

## Caveats

- DCT is a single run (one seed, m = 512, τ = 10) on 24 prompts. Different seeds would give different features.
  The norm fit's objective was still rising at iteration 10 (10,434 → 10,637).
- The headline fit changes DCT by normalising layer-20 states. We did this because the raw objective is dominated
  by numerical blow-up from layer 0. Raw-fit cosines (table 1) tell the same story.
- Our DCT averages Δ over all prompt positions. Mack's demo uses only the last 3 positions, and phase 3 used only
  the last position. A last-positions variant might find features more tied to the reply, and that was not run.
- Rollout labels come from one Sonnet coder per rollout (phase 3 used two), with 16 rollouts per arm. Treat the
  per-arm counts as rough.
- The refusal extra (section 5) uses 8 prompts and greedy 64-token replies. Its per-arm counts come from very
  small samples.

## Files

- `code/dct_l0.py`: rig (layers 1–20 run directly, left padding, steer mask), Δ function, R calibration,
  exponential DCT fit, α.
- `code/analysis.py`: cosine and span comparisons, random baselines, assistant-axis readout, steering hook,
  rollouts.
- `code/cells_fit.py`, `code/cell_compare.py`, `code/cell_checks.py`, `code/cell_rollouts.py`: the Colab cells,
  in run order.
- `results/dct_norm_UV_fp16.npz`, `results/dct_raw_UV_fp16.npz`: U, V (fp16, 4096 × 512), α, R, and ‖Δ(R v)‖ per
  feature, for both fits.
- `results/dct_fit_norm_log.json`, `results/dct_fit_raw_log.json`: calibration curve, objective per iteration,
  Jacobian singular values.
- `results/compare.json`: measures 1, 2 and 4, the persona strength sweep, the Δ-span baseline, and the
  token-embedding check.
- `results/rollouts.json`: all 576 rollouts, with the coder's label and a coherent flag.
- `code/cell_refusal.py`, `code/refusal_analysis.py`: the refusal extra (Colab cell; local analysis that reads raw
  replies from scratch space).
- `results/dct_harmful_UV_fp16.npz`, `results/dct_harmful_fit_log.json`: the DCT fitted on the 8 harmful prompts.
- `results/refusal_keyword.json`: keyword refused/not labels for every arm and every feature, summaries and
  correlations. No reply texts.
- `results/refusal_judged.json`: judge labels (refused / complied / other) for the 752-reply sample, group counts,
  and details of the compliance-causing features. No reply texts.
