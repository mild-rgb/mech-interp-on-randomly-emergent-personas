# Steering demo: chat with Qwen3-8B while two layer-0 directions are dialled in.
#
#   assistant slider -> by default phase 3's persona direction (probe · inlp_debiased_early · L0), one-sided:
#                       0..0.8 pushes off the assistant axis (phase 3 headline: 0.4). It was only fit to push one
#                       way, so there is no "more assistant" side. With phase 1's L20 axis selected it is two-sided.
#   entropy slider   -> phase 2's entropy direction (const · all · eps0.8), phase 2 headline: +0.8
#                       or, under Advanced, phase 2's last-token variant (const · last · eps0.8), added at the last
#                       prompt token only: phase 2 got a fluent English answer after one random token on 74/80 seeds
#
# Both use the phases' rig exactly: at the output of model.model.layers[0], h_t += eps * ||h_t|| * d with ||d|| = 1,
# on every prompt position, prefill only (decode steps untouched). Thinking off. Chat history is re-prefilled
# every turn, so earlier turns are steered too, as in the "all" mask.
#
# Optional: swap the assistant slider for phase 1's INLP-debiased assistant axis read at layer 20, always-on
# (phase 3's "p1-assist⁻ · inlp · L20" reference arm, which got 26% persona at -0.35).
#
# Access: the server listens on 127.0.0.1 only and is reached through a Cloudflare quick tunnel. An ASGI
# middleware rejects every request whose CF-Connecting-IP (set by Cloudflare, not spoofable by the client)
# is outside DEMO_ALLOWED_IPS (comma-separated IPs or CIDRs).
import ipaddress, os, re, socket, subprocess, threading, time, math
from threading import Lock, Thread

import numpy as np
import torch, torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3-8B"
HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2)

tokenizer = model = DIRS = None
STEER = dict(l0=[], l20=[])          # lists of (unit vector, eps, last_only); l20 entries are always-on
LOCK = Lock()                        # hooks read global state, so one generation at a time


def _hook(layer):
    def f(mod, inp, out):
        specs = STEER["l0"] if layer == 0 else STEER["l20"]
        if not specs: return out
        hs = out[0] if isinstance(out, tuple) else out
        if layer == 0 and hs.shape[1] == 1: return out           # layer 0 is prefill-only
        hf = hs.float(); nrm = hf.norm(dim=-1, keepdim=True)
        last = torch.zeros_like(nrm); last[:, -1] = 1               # the "last prompt token" mask
        new = hf
        for d, eps, last_only in specs: new = new + eps * nrm * d * (last if last_only else 1)
        new = new.to(hs.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
    return f


def load(dirs_path=os.path.join(HERE, "directions.npz"), reuse=None):
    """reuse=(model, tokenizer) attaches to an already-loaded model (e.g. after editing this file in a live kernel)."""
    global tokenizer, model, DIRS
    if reuse:
        model, tokenizer = reuse
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        try: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
        except TypeError: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.eval(); model.requires_grad_(False)
    for l in (0, 20): model.model.layers[l]._forward_hooks.clear()     # drop hooks from any earlier load
    z = np.load(dirs_path)
    DIRS = {k: torch.tensor(z[k], device=model.device, dtype=torch.float32) for k in z.files}
    for k, v in DIRS.items(): DIRS[k] = v / v.norm()
    model.model.layers[0].register_forward_hook(_hook(0))
    model.model.layers[20].register_forward_hook(_hook(20))


def set_steer(assist, entropy, source, emask=None):
    STEER["l0"], STEER["l20"] = [], []
    if entropy:
        last = emask == EMASKS[1]
        STEER["l0"].append((DIRS["entropy_last_l0" if last else "entropy_l0"], entropy, last))
    if assist:
        if source.startswith("Phase 3"): STEER["l0"].append((DIRS["persona_l0"], assist, False))    # one-sided: + = persona
        else: STEER["l20"].append((DIRS["assistant_axis_l20"], assist, False))


def _text(content):
    if isinstance(content, str): return content
    if isinstance(content, (list, tuple)):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict): return content.get("text", "")
    return str(content)


def _ids(message, history):
    msgs = [{"role": m["role"], "content": _text(m["content"])} for m in history if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": _text(message)})
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)


def _first_token_stats(logits):
    lp = F.log_softmax(logits.float(), -1)
    H = float(-(lp.exp() * lp).sum() / LN2)
    top = lp.exp().topk(5)
    toks = [(tokenizer.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)]
    return H, toks


def _fmt_top(toks):
    return " · ".join(f"`{t!r}` {p:.2f}" for t, p in toks)


def chat(message, history, assist, entropy, source, emask, temperature, max_new_tokens):
    with LOCK:
        ids = _ids(message, history)
        with torch.no_grad():                                   # clean first-token distribution, for comparison
            set_steer(0, 0, source); H0, top0 = _first_token_stats(model(ids).logits[0, -1])
        set_steer(assist, entropy, source, emask)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        box = {}
        def run():
            try:
                with torch.no_grad():
                    box["out"] = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer,
                        do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0, top_k=0,
                        max_new_tokens=int(max_new_tokens), output_logits=True, return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id)
            except Exception as e:
                box["err"] = e; streamer.end()
        t = Thread(target=run); t.start()
        text = ""
        try:
            for piece in streamer:
                text += piece
                yield text, "*generating…*"
        finally:
            t.join(); set_steer(0, 0, source)
        if "err" in box: raise gr.Error(f"generation failed: {box['err']}")
        H1, top1 = _first_token_stats(box["out"].logits[0][0])
        n = box["out"].sequences.shape[1] - ids.shape[1]
        stats = (f"**First-token entropy:** {H1:.2f} bits steered vs {H0:.2f} clean "
                 f"(ceiling 17.2) · {n} tokens · prompt {ids.shape[1]} tokens\n\n"
                 f"**Top first tokens, steered:** {_fmt_top(top1)}\n\n"
                 f"**Top first tokens, clean:** {_fmt_top(top0)}")
        yield text, stats


EMASKS = ["All prompt tokens (const · all)", "Last prompt token only (const · last)"]
SLIDER = {   # (min, max, label, info) for the assistant slider, per source
    "p3": (0.0, 0.8, "Persona push (phase 3, layer 0)",
           "Pushes the model off its assistant axis. Fit to push one way only, so there is no 'more assistant' side. "
           "Phase 3: 0.4 → 56% persona, 100% coherent, on unseen one-turn prompts."),
    "p1": (-0.8, 0.8, "Assistant axis (phase 1, layer 20)",
           "+ toward the assistant, − toward persona. Phase 3: −0.35 → 26% persona, 88% coherent. "
           "The + side was only tested in phase 1 on trigger prompts, with mixed results."),
}
SOURCES = ["Phase 3 layer-0 direction (prefill only)", "Phase 1 INLP assistant axis at layer 20 (always on)"]


def build_ui():
    with gr.Blocks(title="Persona steering demo") as demo:
        gr.Markdown("## Persona steering demo · Qwen3-8B\n"
                    "Chat normally. Each slider adds a fitted direction to the residual stream (`h += eps·‖h‖·d`); "
                    "by default after block 0 on every prompt token, other options under Advanced. "
                    "Headline numbers are from one-turn prompts; long chats are untested. Settings apply from your next message.")
        with gr.Row():
            lo, hi, lab, inf = SLIDER["p3"]
            assist = gr.Slider(lo, hi, value=0.0, step=0.05, label=lab, info=inf)
            entropy = gr.Slider(0.0, 1.2, value=0.0, step=0.05, label="Layer-0 entropy direction",
                                info="Flattens the first answer token. Phase 2 headline: +0.8 → ~16 bits; "
                                     "on Qwen3 it usually forks into Chinese or Japanese.")
        stats = gr.Markdown("*Stats for the last reply appear here.*")
        with gr.Accordion("Advanced", open=False):
            source = gr.Radio(SOURCES, value=SOURCES[0], label="What the assistant slider uses")
            emask = gr.Radio(EMASKS, value=EMASKS[0], label="Where the entropy slider pushes",
                             info="Each is its own fitted direction. Last token only: one random first token, then usually "
                                  "a fluent answer, mostly English (phase 2: 74/80 Latin-script at +0.8, vs 13/80 for all tokens).")
            temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Temperature",
                                    info="The phases sampled at 1.0, no top-k / top-p. 0 = greedy.")
            max_new = gr.Slider(32, 1024, value=320, step=16, label="Max new tokens")
        def _swap(src):
            lo, hi, lab, inf = SLIDER["p3" if src.startswith("Phase 3") else "p1"]
            return gr.update(minimum=lo, maximum=hi, value=0.0, label=lab, info=inf)
        source.change(_swap, source, assist)
        gr.ChatInterface(
            chat, additional_inputs=[assist, entropy, source, emask, temperature, max_new], additional_outputs=[stats],
            examples=[["What shall I do today?"], ["What's a quick and healthy dinner I can make tonight?"],
                      ["How do I write a good cover letter?"], ["Tell me a fun fact about octopuses."]],
            cache_examples=False, fill_height=True)
    return demo


class IPAllowlist:
    """Pure-ASGI middleware, so it covers http and websocket alike."""
    def __init__(self, app, nets):
        self.app, self.nets = app, nets
    def ok(self, scope):
        h = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        ip = h.get("cf-connecting-ip")
        if not ip: return False                             # not via the tunnel -> deny
        try: a = ipaddress.ip_address(ip.strip())
        except ValueError: return False
        return any(a in n for n in self.nets)
    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and not self.ok(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008}); return
            await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"forbidden\n"}); return
        await self.app(scope, receive, send)


def start_tunnel(port):
    exe = "/usr/local/bin/cloudflared"
    if not os.path.exists(exe):
        subprocess.run(["wget", "-q", "-O", exe,
                        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod(exe, 0o755)
    log = open("/tmp/cloudflared.log", "w")
    p = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], stdout=log, stderr=log)
    url = None
    for _ in range(180):                                    # wait for the URL, a registered connection, and DNS
        time.sleep(1)
        s = open("/tmp/cloudflared.log").read()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", s)
        if not (m and "Registered tunnel connection" in s): continue
        url = m.group(0)
        try: socket.getaddrinfo(url[8:], 443); return p, url
        except socket.gaierror: pass
    raise RuntimeError(f"tunnel {url} did not come up; see /tmp/cloudflared.log")


def main(allowed=None, port=7860, tunnel=True):
    """tunnel=False restarts only the web server, so an already-running tunnel (and its URL) keeps working."""
    import uvicorn
    from fastapi import FastAPI
    allowed = allowed or os.environ.get("DEMO_ALLOWED_IPS", "")
    nets = [ipaddress.ip_network(x.strip(), strict=False) for x in allowed.split(",") if x.strip()]
    if not nets: raise SystemExit("set DEMO_ALLOWED_IPS (comma-separated IPs or CIDRs); refusing to start open")
    if model is None: load()
    demo = build_ui(); demo.queue(default_concurrency_limit=1)
    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    app.add_middleware(IPAllowlist, nets=nets)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    Thread(target=server.run, daemon=True).start()
    time.sleep(3)
    if not tunnel:
        print(f"server restarted on 127.0.0.1:{port}  (allowed: {', '.join(map(str, nets))})"); return server, None, None
    proc, url = start_tunnel(port)
    print(f"demo up at {url}  (allowed: {', '.join(map(str, nets))})")
    return server, proc, url


if __name__ == "__main__":
    main()
    threading.Event().wait()
