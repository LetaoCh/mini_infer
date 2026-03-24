import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from models.gpt_full import ToyGPTConfig, ToyGPTForCausalLM, ToyTokenizer

IGNORE_INDEX = -100


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tune the toy GPT model on chat JSONL.")
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to JSONL chat data prepared by training.prepare_chat_data.",
    )
    parser.add_argument("--out", default="checkpoints/toy-gpt-sft.pt", help="Checkpoint output path.")
    parser.add_argument("--resume", help="Checkpoint path to resume training from.")
    parser.add_argument("--init-from", help="Optional checkpoint path to initialize weights from before SFT.")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-embed", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="User: What is continuous batching?\nAssistant:")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return pick_device()
    return torch.device(name)


def load_chat_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages:
                continue
            records.append(messages)
    if not records:
        raise ValueError("no chat records found")
    return records


def split_records(records, train_ratio: float = 0.98):
    split_idx = max(1, int(len(records) * train_ratio))
    split_idx = min(split_idx, len(records) - 1)
    return records[:split_idx], records[split_idx:]


def build_sft_example(messages, tokenizer: ToyTokenizer, max_seq_length: int):
    input_ids = []
    labels = []

    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        prefix_ids = tokenizer.encode(f"{role}: ")
        content_ids = tokenizer.encode(content)
        newline_ids = tokenizer.encode("\n")

        input_ids.extend(prefix_ids)
        labels.extend([IGNORE_INDEX] * len(prefix_ids))

        input_ids.extend(content_ids)
        if message["role"] == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(content_ids))

        input_ids.extend(newline_ids)
        if message["role"] == "assistant":
            labels.extend(newline_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(newline_ids))

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def tokenize_dataset(records, tokenizer: ToyTokenizer, max_seq_length: int):
    tokenized = []
    for messages in records:
        example = build_sft_example(messages, tokenizer, max_seq_length)
        if example is not None:
            tokenized.append(example)
    if not tokenized:
        raise ValueError("all tokenized examples were empty after filtering")
    return tokenized


def get_batch(data, batch_size: int, seq_len: int, tokenizer: ToyTokenizer, device: torch.device):
    examples = random.choices(data, k=batch_size)
    max_len = min(seq_len, max(len(item["input_ids"]) for item in examples))

    batch_input_ids = []
    batch_labels = []
    attention_masks = []

    for item in examples:
        input_ids = item["input_ids"][-max_len:]
        labels = item["labels"][-max_len:]
        pad_len = max_len - len(input_ids)

        batch_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_len)
        batch_labels.append(labels + [IGNORE_INDEX] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)

    x = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    y = torch.tensor(batch_labels, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    return x, y, attention_mask


def compute_loss(
    model: ToyGPTForCausalLM, x: torch.Tensor, y: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    outputs = model(input_ids=x, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = y[:, 1:].contiguous()
    vocab_size = shifted_logits.size(-1)
    return F.cross_entropy(
        shifted_logits.reshape(-1, vocab_size),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, seq_len, eval_batches, tokenizer, device):
    model.eval()
    metrics = {}
    for split_name, split_data in {"train": train_data, "val": val_data}.items():
        losses = []
        for _ in range(eval_batches):
            x, y, attention_mask = get_batch(split_data, batch_size, seq_len, tokenizer, device)
            losses.append(compute_loss(model, x, y, attention_mask).item())
        metrics[split_name] = sum(losses) / len(losses)
    model.train()
    return metrics


@torch.no_grad()
def generate_sample(model, tokenizer: ToyTokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    max_prompt_len = max(1, model.max_seq_length - max_new_tokens)
    prompt_ids = tokenizer.encode(prompt)[-max_prompt_len:]
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, use_cache=True)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens - 1):
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        past_key_values = outputs.past_key_values
        if token_id == tokenizer.eos_token_id:
            break

    return prompt + tokenizer.decode(generated)


def save_checkpoint(out_path: Path, model, optimizer, config, step, train_loss, val_loss, data_file: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "data_file": data_file,
        },
        out_path,
    )


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = ToyGPTConfig(**checkpoint["config"])
    return checkpoint, config


def assert_same_config(current: ToyGPTConfig, loaded: ToyGPTConfig, path: str):
    if current != loaded:
        raise ValueError(f"checkpoint config mismatch for {path}: " f"loaded={loaded} current={current}")


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
        if args.init_from:
            init_checkpoint, init_config = load_checkpoint(Path(args.init_from), device)
            assert_same_config(config, init_config, args.init_from)
            model.load_state_dict(init_checkpoint["model_state_dict"])

    records = load_chat_records(Path(args.data_file))
    train_records, val_records = split_records(records)
    train_data = tokenize_dataset(train_records, tokenizer, config.max_seq_length)
    val_data = tokenize_dataset(val_records, tokenizer, config.max_seq_length)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"device={device.type}")
    print(f"records={len(records)} train_records={len(train_data)} val_records={len(val_data)}")
    print(
        f"config=d_embed:{config.d_embed} n_layers:{config.n_layers} "
        f"n_head:{config.n_head} seq_len:{config.max_seq_length} batch_size:{args.batch_size}"
    )
    if args.init_from:
        print(f"init_from={args.init_from}")
    if args.resume:
        print(f"resume={args.resume} start_step={start_step}")
        checkpoint_data_file = checkpoint.get("data_file")
        if checkpoint_data_file and checkpoint_data_file != args.data_file:
            print(f"warning=checkpoint_data_file_mismatch checkpoint={checkpoint_data_file} current={args.data_file}")

    start_time = time.time()

    if args.steps <= start_step:
        print(f"nothing to do: steps={args.steps} already reached by checkpoint step={start_step}")
        return

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        x, y, attention_mask = get_batch(train_data, args.batch_size, config.max_seq_length, tokenizer, device)
        loss = compute_loss(model, x, y, attention_mask)

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
                tokenizer=tokenizer,
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
                data_file=args.data_file,
            )

        if step % args.sample_every == 0 or step == args.steps:
            sample = generate_sample(model, tokenizer, args.prompt, args.sample_tokens, device)
            print("----- sample -----")
            print(sample)
            print("------------------")

    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
