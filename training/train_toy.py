import argparse
from dataclasses import asdict
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from models.gpt_full import ToyGPTConfig, ToyGPTForCausalLM, ToyTokenizer


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the toy GPT model on a plain text file.")
    parser.add_argument("--text-file", required=True, help="Path to a plain text corpus.")
    parser.add_argument("--out", default="checkpoints/toy-gpt.pt", help="Checkpoint output path.")
    parser.add_argument("--resume", help="Checkpoint path to resume training from.")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-embed", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="User: Hello\nAssistant:")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return pick_device()
    return torch.device(name)


def load_tokens(text_path: Path, tokenizer: ToyTokenizer) -> torch.Tensor:
    text = text_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        raise ValueError("training corpus is too short; need at least 2 tokens")
    # tokens: [num_tokens]
    return torch.tensor(token_ids, dtype=torch.long)


def split_data(tokens: torch.Tensor, train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = max(1, int(len(tokens) * train_ratio))
    split_idx = min(split_idx, len(tokens) - 1)
    return tokens[:split_idx], tokens[split_idx:]


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    if len(data) < 2:
        raise ValueError("dataset split must contain at least 2 tokens")

    seq_len = min(seq_len, len(data) - 1)
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        starts = torch.zeros(batch_size, dtype=torch.long)
    else:
        starts = torch.randint(0, max_start + 1, (batch_size,))

    # x: current tokens, y: next-token targets
    # x / y: [batch_size, seq_len]
    x = torch.stack([data[i : i + seq_len] for i in starts.tolist()])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


def compute_loss(model: ToyGPTForCausalLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=x, use_cache=False)
    logits = outputs.logits  # [B, T, vocab_size]
    vocab_size = logits.size(-1)
    # Flatten batch and time so cross-entropy sees one row per token position.
    return F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: ToyGPTForCausalLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
):
    model.eval()
    metrics = {}
    for split_name, split_data in {"train": train_data, "val": val_data}.items():
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(split_data, batch_size, seq_len, device)
            losses.append(compute_loss(model, x, y).item())
        metrics[split_name] = sum(losses) / len(losses)
    model.train()
    return metrics


@torch.no_grad()
def generate_sample(
    model: ToyGPTForCausalLM,
    tokenizer: ToyTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    max_prompt_len = max(1, model.max_seq_length - max_new_tokens)
    prompt_ids = tokenizer.encode(prompt)[-max_prompt_len:]
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # [1, prompt_len]
    outputs = model(input_ids=input_ids, use_cache=True)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens - 1):
        # After the first full prompt pass, decode one token at a time from cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        past_key_values = outputs.past_key_values
        if token_id == tokenizer.eos_token_id:
            break

    return prompt + tokenizer.decode(generated)


def save_checkpoint(
    out_path: Path,
    model: ToyGPTForCausalLM,
    optimizer: torch.optim.Optimizer,
    config: ToyGPTConfig,
    step: int,
    train_loss: float,
    val_loss: float,
    text_file: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "text_file": text_file,
        },
        out_path,
    )


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = ToyGPTConfig(**checkpoint["config"])
    return checkpoint, config


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    checkpoint = None
    start_step = 0
    last_metrics = {"train": 0.0, "val": 0.0}

    if args.resume:
        checkpoint, config = load_checkpoint(Path(args.resume), device)
        tokenizer = ToyTokenizer(max_seq_length=config.max_seq_length)
        model = ToyGPTForCausalLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        last_metrics = {
            "train": float(checkpoint.get("train_loss", 0.0)),
            "val": float(checkpoint.get("val_loss", 0.0)),
        }
    else:
        tokenizer = ToyTokenizer(max_seq_length=args.seq_len)
        config = ToyGPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_embed=args.d_embed,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout,
            max_seq_length=args.seq_len,
        )
        model = ToyGPTForCausalLM(config).to(device)

    tokens = load_tokens(Path(args.text_file), tokenizer)
    train_data, val_data = split_data(tokens)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"device={device.type}")
    print(f"tokens={len(tokens)} train_tokens={len(train_data)} val_tokens={len(val_data)}")
    print(
        f"config=d_embed:{config.d_embed} n_layers:{config.n_layers} "
        f"n_head:{config.n_head} seq_len:{config.max_seq_length} batch_size:{args.batch_size}"
    )
    if args.resume:
        print(f"resume={args.resume} start_step={start_step}")
        checkpoint_text_file = checkpoint.get("text_file")
        if checkpoint_text_file and checkpoint_text_file != args.text_file:
            print(f"warning=checkpoint_text_file_mismatch checkpoint={checkpoint_text_file} current={args.text_file}")

    start_time = time.time()

    if args.steps <= start_step:
        print(f"nothing to do: steps={args.steps} already reached by checkpoint step={start_step}")
        return

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        input_ids, target_ids = get_batch(train_data, args.batch_size, config.max_seq_length, device)
        loss = compute_loss(model, input_ids, target_ids)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            last_metrics = estimate_loss(
                model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                seq_len=config.max_seq_length,
                eval_batches=args.eval_batches,
                device=device,
            )
            elapsed = time.time() - start_time
            print(
                f"step={step:5d} train_loss={last_metrics['train']:.4f} "
                f"val_loss={last_metrics['val']:.4f} elapsed={elapsed:.1f}s"
            )
            save_checkpoint(
                Path(args.out),
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                train_loss=last_metrics["train"],
                val_loss=last_metrics["val"],
                text_file=args.text_file,
            )

        if step % args.sample_every == 0 or step == args.steps:
            sample = generate_sample(
                model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.sample_tokens,
                device=device,
            )
            print("----- sample -----")
            print(sample)
            print("------------------")

    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
