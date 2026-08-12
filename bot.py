import os
import asyncio
from dotenv import load_dotenv
from loguru import logger

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams

try:
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport, SmallWebRTCConnection
except ModuleNotFoundError:
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport, SmallWebRTCConnection
    except ModuleNotFoundError:
        from pipecat.transports.services.small_webrtc import SmallWebRTCTransport, SmallWebRTCConnection
        
load_dotenv(override=True)

# Initialize FastAPI
app = FastAPI()

# ---------------------------------------------------------------------------
# 1. CORS CONFIGURATION
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://f379fecc-08b1-4f6a-8f94-525339570e46.lovableproject.com",
        "https://id-preview--f379fecc-08b1-4f6a-8f94-525339570e46.lovable.app",
        "*"  # Allows all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE
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
# 3. SYSTEM PROMPT
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

# ---------------------------------------------------------------------------
# 4. PIPECAT PIPELINE RUNNER
# ---------------------------------------------------------------------------
async def run_bot(transport):
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            punctuate=True,
            smart_format=True,
        )
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en"
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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

# ---------------------------------------------------------------------------
# 5. WEBRTC OFFER ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/api/offer")
async def offer(request: Request):
    offer_data = await request.json()

    # 1. Instantiate the WebRTC connection object
    webrtc_connection = SmallWebRTCConnection()

    # 2. Pass webrtc_connection to SmallWebRTCTransport
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # 3. Launch the bot pipeline asynchronously
    asyncio.create_task(run_bot(transport))

    # 4. Handle the SDP offer exchange
    try:
        answer = await webrtc_connection.handle_offer(
            offer_data.get("sdp"), 
            offer_data.get("type", "offer")
        )
    except TypeError:
        answer = await webrtc_connection.handle_offer(offer_data)

    # Format the response back to Lovable
    if hasattr(answer, "sdp") and hasattr(answer, "type"):
        return JSONResponse(content={"sdp": answer.sdp, "type": answer.type})
    return JSONResponse(content=answer)

# ---------------------------------------------------------------------------
# 6. APP ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
