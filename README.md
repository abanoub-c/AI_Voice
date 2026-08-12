# Voice Agent — "John" stack

This is a working starter for the low-latency, interruptible voice agent from your proposal. It's built on **Pipecat** (the open-source framework behind most production voice agents) and every import in `bot.py` has been verified against the real, currently-installed `pipecat-ai` package — nothing here is guessed from memory.

## The stack, and why it's "John" and not "Alex"

| Layer | Tool | Why this one |
|---|---|---|
| Framework | **Pipecat** (open source) | Handles the audio pipeline, turn-taking, and barge-in for you — this is the plumbing that makes interruption possible at all. |
| STT | **Deepgram Nova-3** (streaming) | ~200-300ms to first transcript, best-in-class accuracy for the price. |
| LLM | **Groq** — Llama 3.3 70B | Runs on Groq's LPU hardware at 300-900+ tokens/sec — the single biggest lever on "dead air" between the caller finishing a sentence and the bot starting to respond. |
| TTS | **Cartesia Sonic-2** | ~90ms time-to-first-audio, built for real-time conversation rather than narration. |
| VAD / turn-taking | **Silero** (local, built into the pipeline) | Detects the caller starting to talk mid-response and lets them cut the bot off instantly — this *is* the interruption feature. |
| Transport | **Daily / WebRTC** for local testing → swap to Twilio/Vonage/Telnyx for a real phone number | Zero telephony setup needed to demo in a browser today. |

Rough all-in AI-services cost at these settings: **~$0.04–0.08/min** (Deepgram ≈$0.005–0.008/min + Cartesia ≈$0.03/min + Groq LLM ≈ fractions of a cent/min). Add ~$0.014/min if you route through Twilio for a real phone number, and Daily/Pipecat Cloud hosting on top if you deploy. Exact numbers move — check each provider's pricing page before you quote a client. Either way you're well inside "John" territory, nowhere near $0.40+/min.

## The 2-hour plan

**0:00–0:20 — Get three free API keys** (all have generous free tiers, no card needed to start):
- Deepgram: https://console.deepgram.com/signup
- Groq: https://console.groq.com
- Cartesia: https://play.cartesia.ai/sign-up

**0:20–0:30 — Install and configure**
```bash
# if you don't have uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

cd voice-agent-john
cp .env.example .env
# paste your 3 API keys into .env

uv sync
```

**0:30–0:40 — Run it and talk to it**
```bash
uv run bot.py
```
Open `http://localhost:7860/client`, click **Connect**, allow the mic, and talk. First run takes ~20s to download the VAD model; after that it's instant. Try talking over it mid-sentence — that's the barge-in working.

**0:40–1:30 — Make it actually your agent**
Open `bot.py` and rewrite the `SYSTEM_PROMPT` block near the top — that's the only part that's product-specific. Everything below it (STT/LLM/TTS wiring) doesn't need to change for a first version. Tell me what the product does and what the call should accomplish and I'll write this properly, including any tool calls it needs (booking a slot, looking up an order, escalating to a human).

**1:30–2:00 — Show it to your client**

Two ways to demo, depending on whether they just watch or try it themselves — both free:

- **Screen-share (simplest, $0):** run `uv run bot.py`, open `localhost:7860/client`, and talk to it live on your call with them. No deployment, no extra signup.
- **Send them a link they can click themselves (still $0, temporary):**
  ```bash
  # in a second terminal, while bot.py is running:
  ngrok http 7860
  ```
  Sign up free at https://ngrok.com, then send your client the `https://...ngrok-free.app/client` URL it gives you. It only works while your laptop and `bot.py` are running, but there's no cost and no cloud account needed.

Note: **Pipecat Cloud is usage-based, not free** — skip `pipecat cloud deploy` for a demo. It's the right move later if you want a persistent, always-on link, not for showing a client today.

A practical limit while demoing: Cartesia's free tier is only ~15–20 minutes of total audio before it needs a paid plan, so a couple of live run-throughs is about what you get before topping up.

## Files

- `bot.py` — the whole agent. Import-checked against pipecat-ai 1.7.0.
- `pyproject.toml` — dependencies for `uv`.
- `.env.example` — copy to `.env` and fill in your keys.

## Customer support / FAQ agent — what's left

`bot.py` is now set up as a support agent that answers strictly from a knowledge base you paste in (not free-roaming), and hands off to a human whenever the answer isn't in that knowledge base or the caller sounds frustrated. Two things to do before it's real:

1. **Paste your FAQ/help-doc content** into `KNOWLEDGE_BASE` near the top of `bot.py`. Plain Q&A pairs work well. If you send me the actual doc, I'll drop it in and tune the prompt around it directly.
2. **Fill in `[YOUR COMPANY]`** in `SYSTEM_PROMPT`.

For a real hand-off ("let me transfer you to a human"), the fastest v1 is having the bot say so and end the call to a monitored number — a live warm-transfer needs a bit more telephony wiring (SIP transfer via Twilio/Telnyx), worth doing once you're past the demo stage.
