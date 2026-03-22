from dataclasses import dataclass
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .models.gpt_full import ToyGPTConfig, ToyGPTForCausalLM, ToyTokenizer, build_toy_model_bundle
except ImportError:
    from models.gpt_full import ToyGPTConfig, ToyGPTForCausalLM, ToyTokenizer, build_toy_model_bundle

DEFAULT_MODEL_BACKEND = "hf"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TOY_MODEL_NAME = "toy-gpt"


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    model_name: str


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_hf_model_bundle(model_name: str, device: torch.device) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type in {"cuda", "mps"} else torch.float32,
        trust_remote_code=True,
    ).to(device)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
    )


def _load_toy_model_bundle(model_name: str | None, device: torch.device) -> ModelBundle:
    if model_name and os.path.exists(model_name):
        checkpoint = torch.load(model_name, map_location=device)
        config = ToyGPTConfig(**checkpoint["config"])
        tokenizer = ToyTokenizer(max_seq_length=config.max_seq_length)
        model = ToyGPTForCausalLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return ModelBundle(
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_name=model_name,
        )

    toy_bundle = build_toy_model_bundle(device=device)
    return ModelBundle(
        tokenizer=toy_bundle.tokenizer,
        model=toy_bundle.model,
        device=toy_bundle.device,
        model_name=model_name or DEFAULT_TOY_MODEL_NAME,
    )


def load_model_bundle(
    model_name: str | None = None,
    model_backend: str = DEFAULT_MODEL_BACKEND,
) -> ModelBundle:
    device = pick_device()

    if model_backend == "hf":
        return _load_hf_model_bundle(model_name or DEFAULT_MODEL_NAME, device)
    if model_backend == "toy":
        return _load_toy_model_bundle(model_name, device)

    raise ValueError(f"invalid model_backend: {model_backend}")
