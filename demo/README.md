# Demo: chat with Qwen3-8B under the phase 2 and phase 3 directions

A standard chatbot UI with two sliders:

| slider | what it adds | headline setting |
|---|---|---|
| **Persona push** (default) | phase 3's layer-0 persona direction (`probe · inlp_debiased_early · L0 · eps0.4`), one-sided, 0 to 0.8. It was fit only to push the model *off* the assistant axis, so it has no "more assistant" side | **0.4**: 56% persona on unseen one-turn prompts, coherent, English |
| **Layer-0 entropy direction** | phase 2's entropy direction (`const · all · eps0.8`) | **+0.8**: ~16 bits on the first answer token; on Qwen3 it usually forks into Chinese or Japanese |

Both use the phases' rig exactly: added at the output of `model.model.layers[0]` as `h_t += eps · ‖h_t‖ · d`
(`‖d‖ = 1`) on every prompt token, prefill only, thinking off. With both on, the two pushes add. The chat history
is re-prefilled each turn, so earlier turns are steered too. Under "Advanced" you can switch the first slider
to phase 1's INLP-debiased assistant axis at layer 20, always on. That one is two-sided, −0.8 to +0.8, with + toward
the assistant (phase 3's reference arm: 26% persona at −0.35; the + side was only tested in phase 1 on trigger prompts,
with mixed results),
switch the entropy slider to phase 2's **last-token** direction (`const · last · eps0.8`, pushed at the last prompt
token only: one random token, then mostly English, 74/80 Latin-script in phase 2, often in the thinking register), and change
temperature (the phases used 1.0) and length.

The last-token direction is weak on two held-out prompts in phase 2 and here alike: "what shall i do today"
(~1–2 bits) and the SQL question. Its held-out mean reproduces at 12.8 bits (phase 2: 12.9).

`load(..., reuse=(model, tokenizer))` and `main(..., tunnel=False)` let you edit `app.py` and reload it in a live
Colab kernel without reloading the model or changing the tunnel URL.

After each reply the page shows the first-token entropy, steered vs clean, and the top 5 first tokens for each.

`directions.npz` holds the four unit vectors (65 KB), copied from
`phase2-layer0-entropy/results/phase2_directions_qwen3-8b.npz`, `phase3-persona-direction/results/phase3_directions_qwen3-8b.npz`
and `phase3-persona-direction/code/phase3_inputs.npz`.

## Running it on Colab

Needs a GPU with 24 GB or more (L4 or A100; bf16 weights are ~16.4 GB).

```python
!pip install -q -U "transformers>=4.51" gradio accelerate
# upload app.py and directions.npz to /content, then:
import os, sys; sys.path.insert(0, "/content")
os.environ["DEMO_ALLOWED_IPS"] = "203.0.113.7,2001:db8:1234:5678::/64"   # your IPv4, and your IPv6 /64
import app; server, tunnel, url = app.main()
```

`main()` loads the model, serves the UI on `127.0.0.1:7860` inside the VM, and opens a Cloudflare quick tunnel
(no account needed). It prints a `https://….trycloudflare.com` URL.

## Who can reach it

The URL itself is public, but every request is checked against `DEMO_ALLOWED_IPS` by an ASGI middleware before
Gradio sees it. The check reads `CF-Connecting-IP`, which Cloudflare sets to the real client address and does not
let the client override. Requests without that header (anything not coming through the tunnel) are refused too.
Everyone else gets `403 forbidden`. Allow your IPv6 /64, not a single IPv6 address: browsers rotate the low bits.
The app refuses to start with an empty allowlist.

## Content note

Phase 4 found that the persona push lowers refusal on HarmBench (77% → 53%). Keep that in mind
before pointing anyone else at a running demo.
