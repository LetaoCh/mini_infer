from pydantic import BaseModel


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
