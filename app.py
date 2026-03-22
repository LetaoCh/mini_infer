import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import MiniInferenceEngine
from .types import OverloadedError


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int


class GenerateResponse(BaseModel):
    request_id: int
    text: str
    finish_reason: str
    arrival_time: float
    service_start_time: float
    completion_time: float
    batching_delay: float
    processing_delay: float
    generated_tokens: int


max_request = 10
engine = MiniInferenceEngine(
    max_request,
    batch_size=4,
    prefill_mode=os.getenv("PREFILL_MODE", "batched"),
    decode_mode=os.getenv("DECODE_MODE", "batched"),
)


@asynccontextmanager
async def lifespan(app):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        res = await engine.submit_request(req.prompt, req.max_new_tokens)
    except OverloadedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

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
    except OverloadedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

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
    return engine.get_stats()
