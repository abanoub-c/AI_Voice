"""
Voice AI agent — the "John" stack.

STT:       Deepgram (nova-3, streaming)
LLM:       Groq (Llama 3.3 70B) — fast + cheap. Swap to OpenAI/Anthropic any time,
           see the comment on the `llm =` line below.
TTS:       Cartesia (Sonic-2) — sub-200ms time-to-first-audio.
Transport: Daily (WebRTC) for local/browser testing. Swap to Twilio/Vonage for
           real inbound/outbound phone calls (see README "Going to a real phone
           number" section).
VAD:       Silero, wired into the user context aggregator below. This is what
           gives you instant barge-in — the caller can cut the bot off mid-
           sentence — instead of "Alex"'s press-to-talk, wait-for-silence
           behavior.

Run it:
    uv sync
    uv run bot.py
Then open http://localhost:7860/client and click Connect.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)
# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE — paste your actual FAQ / help-doc / policy text here.
#    For a v1 built in an afternoon, dropping the raw text straight into the
#    prompt is faster and more reliable than standing up a RAG pipeline, and
#    Llama 3.3 70B's context window comfortably holds a full FAQ page or two.
#    Swap to real retrieval only once this grows past a few thousand words.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE — Your Marketing & Technical Portfolio
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = """
Q: Who are you and who do you represent?
A: I am the AI technical assistant for Abanoub, a Data Scientist with a B.Sc. and over three years of experience turning prototypes into production.

Q: What makes Abanoub different from other freelance data scientists or developers?
A: With Abanoub, you stop paying for code and start paying for ROI. He doesn't just deliver technical outputs; he translates vague business goals into solvable ML problems and builds trackable assets that drive actual revenue.

Q: What are his core technical skills and expertise?
A: He specializes in End-to-End MLOps, predictive modeling, and NLP. He handles everything from data preparation and real-world tuning using tools like XGBoost and SHAP, to full production deployments using Python, FastAPI, and React dashboards.

Q: Can you give me examples of his past work?
A: He recently built an AI Contract Review Assistant using a multi-LLM RAG system with Weaviate hybrid search and a fact-checking layer to prevent hallucinations. He also deployed a retail demand forecasting model that cut stockout-driven lost sales by thirty-four percent.

Q: Why should I hire him?
A: Many data scientists stop at the notebook. Abanoub cleans messy data, validates models against real-world scenarios, and provides clear dashboards so non-technical stakeholders can actually trust and use the solution.

Q: How much does he charge?
A: He bills at an hourly rate, though the total investment depends entirely on the scope and requirements of the project.

Q: How do we start working with him?
A: Just give me a brief description of your problem or KPI today. Abanoub will follow up with clear, practical next steps to turn your data into impact.
"""

# ---------------------------------------------------------------------------
# 2. THE AGENT'S BRAIN — The Sales Engineer Persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a sharp, respectful, and dryly witty AI sales engineer representing Abanoub (pronounced Ah-bah-noob), a freelance data scientist. 

Your job is to talk like a real human on a quick phone call: ultra-short sentences, entirely polite to the client, but with a sharp, sarcastic wink toward typical bad tech industry habits and broken Jupyter notebooks.

Knowledge base about Abanoub:
---
{KNOWLEDGE_BASE}
---

Rules for the conversation:
- HUMAN LENGTH: Keep every single response to one or two short sentences max. Never ramble or list things out.
- TONE BLEND: Be entirely respectful and professional to the client, but keep a dry, witty edge. Make them smile, don't lecture them.
- Never use emojis, markdown, bullet points, or asterisks -- none of that can be spoken aloud.
- Spell out numbers for natural speech (e.g., say "thirty-four percent").
- PRICING: If they ask about cost, state smoothly that he bills hourly, but the final price tag depends on what the project actually needs.
- If they mention a business problem, drop a quick technical mention (like XGBoost or FastAPI) and how Abanoub actually ships it to production.
- If they are ready to move forward, politely offer to grab their contact details so Abanoub can send a proposal.
"""

async def run_bot(transport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
        model="nova-3-general",  # Deepgram's most advanced model with the lowest Word Error Rate
        punctuate=True,
        smart_format=True,
    )
)

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en" # A highly natural, professional female voice
    )
    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
    #     model=os.getenv("CARTESIA_MODEL", "sonic-2"),
    # )

    # Groq gives you the lowest LLM-side latency and the lowest per-token cost,
    # which is most of what separates "John" from "Alex" on the bill.
    # To use OpenAI instead (better reasoning, a bit slower/pricier):
    #   from pipecat.services.openai.llm import OpenAILLMService
    #   llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # vad_analyzer here is the whole barge-in story: it detects the caller
        # starting to speak and interrupts the bot instantly instead of
        # waiting for it to finish talking.
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # audio in from the caller
            stt,  # speech -> text
            user_aggregator,  # add the caller's turn to conversation history
            llm,  # text -> reply
            tts,  # reply -> speech
            transport.output(),  # audio out to the caller
            assistant_aggregator,  # add the bot's turn to conversation history
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message(
            {"role": "developer", "content": "Greet the caller briefly and ask how you can help."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point Pipecat's runner calls — native WebRTC without Daily."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
    main()
