from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    model_name: str


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_bundle(model_name: str = DEFAULT_MODEL_NAME) -> ModelBundle:
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in {"cuda", "mps"} else torch.float32,
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
