from dataclasses import dataclass
from enum import Enum, auto


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


@dataclass
class InferenceResult:
    request_id: int
    text: str
    finish_reason: str
