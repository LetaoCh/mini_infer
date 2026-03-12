from contextlib import asynccontextmanager
from fastapi import FastAPI
from .engine import MiniInferenceEngine

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int


class GenerateResponse(BaseModel):
    request_id: int
    text: str
    finish_reason: str
    arrival_time: float
    batch_start_time: float
    completion_time: float
    batching_delay: float
    processing_delay: float


max_request = 10
engine = MiniInferenceEngine(max_request, batch_size=4)


@asynccontextmanager
async def lifespan(app):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest):
    res = await engine.submit_request(req.prompt, req.max_new_tokens)
    return GenerateResponse(
        request_id=res.request_id,
        text=res.text,
        finish_reason=res.finish_reason,
        arrival_time=res.arrival_time,
        batch_start_time=res.batch_start_time,
        completion_time=res.completion_time,
        batching_delay=res.batching_delay,
        processing_delay=res.processing_delay,
    )
