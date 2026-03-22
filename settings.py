import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    max_requests: int = 10
    batch_size: int = 4
    model_name: str | None = None
    prefill_mode: str = "batched"
    decode_mode: str = "batched"


def load_app_settings() -> AppSettings:
    return AppSettings(
        max_requests=int(os.getenv("MAX_REQUESTS", "10")),
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
        model_name=os.getenv("MODEL_NAME") or None,
        prefill_mode=os.getenv("PREFILL_MODE", "batched"),
        decode_mode=os.getenv("DECODE_MODE", "batched"),
    )
