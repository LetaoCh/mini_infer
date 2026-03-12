from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceRequest:
    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_time: float
    batch_start_time: Optional[float] = None


@dataclass
class InferenceResult:
    request_id: int
    text: str
    finish_reason: str
    arrival_time: float
    batch_start_time: float
    completion_time: float
    batching_delay: float
    processing_delay: float


@dataclass
class ActiveRequest:
    request: InferenceRequest
    generated_tokens: int = 0
    output_text: str = ""