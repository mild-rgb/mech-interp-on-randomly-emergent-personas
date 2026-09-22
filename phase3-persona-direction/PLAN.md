# Phase 3 — a layer-0 direction for coherent personas, not for entropy

**Written 2026-09-21, before the run. Predictions in §4 are registered and not edited after.**

## 1. Question

Phase 2's layer-0 directions were fit to flatten the first answer token. On Qwen3-8B that gave 16 bits and
7.5% persona (two blind coders), because the fork mostly landed in Chinese. Can a direction fit **at layer 0**
against a **persona** objective raise the rate of *coherent* personas (persona by both coders, coherence 2 by
both) on prompts it was not fit on, without the language exit?

## 2. Arms

Backbone Qwen3-8B, thinking off, same 24 train / 16 held-out prompts as phase 2, same eps convention
(`Δh_t = eps · ‖h_t‖ · d`, `‖d‖ = 1`, so random directions are exact controls).

| arm | where | what | fit |
|---|---|---|---|
| `clean` | — | — | — |
| `p2-entropy` | L0, all prompt positions, eps 0.8 | phase 2's `const · all` direction | none (reference) |
| `p1-assist⁻` | **L20**, eps {0.2, 0.35, 0.5}, prompt-only and always-on | phase 1's assistant direction (`massmean_early`, `inlp_debiased_early`) **reversed** | none (reuse) |
| **`probe`** | **L0**, all prompt positions, eps {0.2, 0.4} | new direction fit to minimise the projection of the layer-20 first-answer-position state onto phase 1's assistant direction, with a hinge penalty when `H1` > 8 bits | Adam, 150 steps |
| **`opener`** | **L0**, all prompt positions, eps {0.2, 0.4} | new direction fit to minimise the total probability of each prompt's clean top-5 first tokens, same `H1` hinge | Adam, 150 steps |
| `rand` | L0 and L20, eps 0.4 | random unit vectors, 8 draws | none (control) |

The `H1` hinge is what keeps "persona" from collapsing into "broken": phase 2 showed the entropy objective alone
walks into salad on Llama and Mistral and into Chinese on Qwen.

## 3. Readout

10 seeds × 8 held-out prompts × 96 tokens per arm (~800 rollouts). Two blind Claude Sonnet 5 coders, phase 2's
rubric. Headline number: **coherent-persona rate** (persona by both, coherence 2 by both). Also default-assistant
rate, language split, `H1`, `Hbar`, and the phase 2 safety screen as an outcome.

## 4. Predictions (registered)

1. `p1-assist⁻` at L20, eps 0.35, always-on, gives **≥ 20%** persona by both coders on clean held-out prompts
   (phase 1 got 58% on trigger prompts).
2. `probe` at eps 0.4 gives a **higher coherent-persona rate than `p2-entropy`** (7.5%) at a **lower** `H1`.
3. `opener` raises persona less than `probe` but keeps **≥ 60% English** rollouts.
4. `rand` stays at 0% persona at both layers.
5. The `H1` hinge binds: the fitted `probe` direction sits at 6–8 bits, not 16.

## 5. Files

`code/phase3_persona_direction.ipynb`, `code/prompts.json`, `results/phase3_results.json`, `results/judging/`.
Phase 1's directions come from `../phase1-indy_mech_extension/directions/steer_candidates.npz`; phase 2's from
`../phase2-layer0-entropy/results/phase2_directions_qwen3-8b.npz`.
