import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse

from .dashboard import STATS_DASHBOARD_HTML
from .engine import MiniInferenceEngine
from .schemas import GenerateRequest, GenerateResponse
from .settings import load_app_settings
from .state import OverloadedError

settings = load_app_settings()
engine = MiniInferenceEngine(
    settings.max_requests,
    batch_size=settings.batch_size,
    model_name=settings.model_name,
    server_profile=settings.server_profile,
    prefill_mode=settings.prefill_mode,
    decode_mode=settings.decode_mode,
    tick_log_every=settings.tick_log_every,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(lifespan=lifespan)


def raise_service_unavailable(error: OverloadedError):
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(error),
    )


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        res = await engine.submit_request(req.prompt, req.max_new_tokens)
    except OverloadedError as error:
        raise_service_unavailable(error)

    return GenerateResponse(
        request_id=res.request_id,
        text=res.text,
        finish_reason=res.finish_reason,
        arrival_time=res.arrival_time,
        service_start_time=res.service_start_time,
        completion_time=res.completion_time,
        batching_delay=res.batching_delay,
        processing_delay=res.processing_delay,
        generated_tokens=res.generated_tokens,
    )


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest, request: Request):
    try:
        ctx = await engine.submit_streaming_request(req.prompt, req.max_new_tokens)
    except OverloadedError as error:
        raise_service_unavailable(error)

    async def event_generator():
        disconnected = False
        try:
            while True:
                if await request.is_disconnected():
                    engine.cancel_request(ctx.request_id)
                    disconnected = True
                    break

                token = await ctx.token_queue.get()
                if token is None:
                    break

                yield f"data: {json.dumps({'token': token})}\n\n"

            if not disconnected:
                result = await ctx.future
                yield f"data: {json.dumps({
                    'done': True,
                    'request_id': result.request_id,
                    'finish_reason': result.finish_reason,
                    'generated_tokens': result.generated_tokens,
                })}\n\n"

        except Exception:
            engine.cancel_request(ctx.request_id)
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    return HTMLResponse(STATS_DASHBOARD_HTML)


@app.get("/stats.json")
async def stats_json():
    return engine.get_stats()
