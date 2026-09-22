# Phase 2 — layer-0 interventions that flatten the first answer token, across prompts

**Written 2026-09-21, before the run.** Predictions in §4 are registered here and not edited after.

## 1. Question

Phase 11 (CoT-spiking) reached `H1` = 13.76 bits on one query with a 16-token GCG prefix; phase 15 reached
13.5 on six queries with one prefix. Both need junk tokens in the context window, and both were fit to
the prompts they were scored on. Can an intervention **inside the model at transformer layer 0**, with
**nothing added to the context**, do the same thing, and does it carry over to prompts it was not fit on?

`H1` is unchanged from phase 11: Shannon entropy in bits of the next-token distribution at the first
answer position, full 151,936-token vocabulary, float32 from bf16 logits, `Qwen/Qwen3-8B`, chat
template, thinking off. Ceiling `log2(V)` = 17.213. Clean `what shall i do today` = 0.237 (rig check).

## 2. Prompt set

40 ordinary user requests (`code/prompts.json`): advice, factual, how-to, creative, coding, planning.
**24 train / 16 held-out**, fixed before the run. Phase 15's six queries are all in the held-out set,
including `what shall i do today`, so the numbers line up with phases 11 and 15 directly.

## 3. Interventions

All at the output of `model.model.layers[0]` (the residual stream after block 0). Every arm adds a
perturbation of **matched relative size**: at position `t`, `Δh_t = eps · ‖h_t‖ · d_t` with `‖d_t‖ = 1`.
So `eps` is the fraction of the token's own residual norm, and a random direction at the same `eps`
is an exact control (the lesson from phase 1's steering-alpha mistake).

| family | direction `d_t` | params | fit |
|---|---|---|---|
| `const` | one unit vector `u`, same at every position | 4,096 | Adam on train prompts |
| `lin8` | `M ĥ_t / ‖M ĥ_t‖`, `M` rank-8, so the direction depends on the token | 65,536 | Adam on train prompts |
| `rand` | random unit vector, 8 draws | 0 | none (control) |
| `scale` | none: `h_t ← s · h_t`, and separately zero block 0's attention or MLP output | 0 | none (ablation) |

Position masks: `all` (every prompt position), `user` (only the user's query tokens, the analogue
of where the GCG prefix sat), `last` (only the final prompt position, the minimal contact).

Objective, as phase 15: `mean(H1) − 1.0 · std(H1)` over the 24 train prompts. `eps` sweep
{0.05, 0.1, 0.2, 0.4, 0.8}. 150 Adam steps each. Report train and held-out mean / std / min.

During sampled rollouts the intervention is applied to the **prompt positions only** (prefill), not to
generated tokens. That is the analogue of a trigger sitting in the prompt, and phase 15's reading of the
state; whether an always-on version behaves differently is a later question.

## 4. Predictions (registered)

1. `const · all` reaches **≥ 12 bits** mean `H1` on train at some `eps ≤ 0.4`, and the held-out mean is
   **within 1.5 bits** of train. (Phase 10's soft prompt hit 17.02, so continuous perturbations are
   known to get there; transfer is the open part.)
2. `rand` at the same `eps` stays **below 2 bits** for `eps ≤ 0.4`. Above that it may rise, but only
   by breaking the model (checked by reading the text and by `Hbar`).
3. `last` needs a larger `eps` than `all` to reach the same `H1`: one position perturbed at layer 0
   has 35 more layers to be corrected, with no other positions carrying the perturbation. I expect
   `last` to fall **≥ 3 bits short** of `all` at every `eps ≤ 0.4`.
4. The flat state re-coheres in one token, as phase 15: `H1 / Hbar ≥ 3`, and the sampled continuations
   are multilingual with ≤ 50% English.
5. `scale` and the two ablations do not reach 5 bits without producing degenerate text (repetition or
   empty output on ≥ half the seeds).
6. `lin8` beats `const` by **< 1 bit** on held-out: a constant push is enough for this objective.

## 5. What would change my mind

If `rand` matches `const` at matched `eps`, the learned direction is not doing anything a norm bump
would not, and the result is "layer 0 is fragile", not "there is a direction". If held-out is far
below train, the direction is prompt-specific, the same failure phase 15's spread penalty was built
against. If the rollouts are word salad rather than fluent-in-another-register, this is decoherence,
not a persona state, and it belongs to phase 10's story, not this programme's.

## 6. Files

- `code/phase2_layer0_entropy.ipynb` — the run (Colab, A100 or L4, bf16).
- `code/prompts.json` — the 40 prompts and the split.
- `results/phase2_results.json` — every arm's `H1` per prompt, mass-support quantiles, rollouts.
