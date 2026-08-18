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

# --- PIPECAT WEBRTC IMPORTS ---
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
    IceCandidate
)

try:
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
except ModuleNotFoundError:
    try:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
    except ModuleNotFoundError:
        from pipecat.transports.services.small_webrtc import SmallWebRTCTransport

import asyncio
from webrtc_stability_patches import apply_aioice_patches, install_stun_retry_exception_filter
apply_aioice_patches()

load_dotenv(override=True)

# Initialize FastAPI
app = FastAPI()

# Initialize the Pipecat handler that seamlessly manages peer connections
webrtc_request_handler = SmallWebRTCRequestHandler()

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

    # Fixed: Updated to use the Settings object
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(
            voice="aura-asteria-en"
        )
    )

    # Fixed: Updated to use the Settings object
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )
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
# 5. WEBRTC ENDPOINTS
# ---------------------------------------------------------------------------

# POST route for the initial WebRTC Offer
@app.post("/api/offer")
async def post_offer(request: Request):
    offer_data = await request.json()
    
    # 1. Parse incoming data into Pipecat's required format
    webrtc_request = SmallWebRTCRequest.from_dict(offer_data)
    
    # 2. Define the callback that launches your bot when connection is established
    async def webrtc_callback(webrtc_connection):
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True)
        )
        # Launch the bot pipeline asynchronously
        asyncio.create_task(run_bot(transport))

    # 3. Pipecat's handler dynamically creates the connection, triggers the bot callback, and returns the SDP answer
    response_data = await webrtc_request_handler.handle_web_request(
        request=webrtc_request,
        webrtc_connection_callback=webrtc_callback
    )
    
    # Send the generated SDP answer back to the Lovable UI
    return JSONResponse(content=response_data)

# PATCH route for Trickle ICE Candidates (Fixes your 405 Error)
@app.patch("/api/offer")
async def patch_offer(request: Request):
    patch_data = await request.json()
    
    # Parse the incoming ICE candidates
    pc_id = patch_data.get("pc_id")
    candidates_data = patch_data.get("candidates", [])
    
    # Transform incoming JSON fields to Pipecat's IceCandidate objects
    candidates = [
        IceCandidate(
            candidate=c.get("candidate", ""),
            sdp_mid=c.get("sdpMid", c.get("sdp_mid", "")),
            sdp_mline_index=int(c.get("sdpMLineIndex", c.get("sdp_mline_index", 0)))
        ) for c in candidates_data
    ]
    
    # Reconstruct the patch request and hand it to Pipecat
    patch_request = SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
    await webrtc_request_handler.handle_patch_request(patch_request)
    
    return JSONResponse(content={"status": "patched successfully"})

# ---------------------------------------------------------------------------
# 6. APP ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
