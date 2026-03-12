from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    FINISHED = auto()
    FAILED = auto()


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
