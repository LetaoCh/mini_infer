from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


@dataclass
class InferenceRequest:
    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_time: float
    admit_time: Optional[float] = None
    service_start_time: Optional[float] = None


@dataclass
class InferenceResult:
    request_id: int
    text: str
    finish_reason: str
    arrival_time: float = 0
    service_start_time: float = 0
    completion_time: float = 0
    batching_delay: float = 0
    processing_delay: float = 0


class OverloadedError(Exception):
    pass


@dataclass
class ActiveRequest:
    request: InferenceRequest
    generated_tokens: int = 0
    output_text: str = ""
