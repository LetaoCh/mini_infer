import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    server_profile: str = "custom"
    max_requests: int = 64
    batch_size: int = 4
    model_name: str | None = None
    prefill_mode: str = "batched"
    decode_mode: str = "batched"
    tick_log_every: int = 50


PROFILE_PRESETS: dict[str, AppSettings] = {
    "balanced": AppSettings(
        server_profile="balanced",
        max_requests=64,
        batch_size=16,
        prefill_mode="batched",
        decode_mode="batched",
        tick_log_every=200,
    ),
    "throughput": AppSettings(
        server_profile="throughput",
        max_requests=64,
        batch_size=32,
        prefill_mode="batched",
        decode_mode="batched",
        tick_log_every=200,
    ),
}


def _load_profile_defaults() -> AppSettings:
    profile_name = os.getenv("SERVER_PROFILE", "balanced").strip().lower()
    if not profile_name:
        return AppSettings()

    try:
        return PROFILE_PRESETS[profile_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILE_PRESETS))
        raise ValueError(f"invalid SERVER_PROFILE: {profile_name} (expected one of: {valid})") from exc


def load_app_settings() -> AppSettings:
    base = _load_profile_defaults()
    return AppSettings(
        server_profile=base.server_profile,
        max_requests=int(os.getenv("MAX_REQUESTS", str(base.max_requests))),
        batch_size=int(os.getenv("BATCH_SIZE", str(base.batch_size))),
        model_name=os.getenv("MODEL_NAME") or base.model_name,
        prefill_mode=os.getenv("PREFILL_MODE", base.prefill_mode),
        decode_mode=os.getenv("DECODE_MODE", base.decode_mode),
        tick_log_every=max(1, int(os.getenv("TICK_LOG_EVERY", str(base.tick_log_every)))),
    )
