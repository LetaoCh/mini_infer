import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare chat SFT datasets for training.train_toy_sft.")
    parser.add_argument(
        "--dataset",
        choices=["smol_smoltalk"],
        default="smol_smoltalk",
        help="Chat dataset source to export.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to export.")
    parser.add_argument("--out", default="data/generated/smol_smoltalk_train.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-examples", type=int, default=50000, help="Maximum conversations to export.")
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_messages(messages):
    normalized = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def export_smol_smoltalk(out_path: Path, split: str, max_examples: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with '.venv/bin/pip install datasets'."
        ) from exc

    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, streaming=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            messages = normalize_messages(example.get("messages", []))
            if len(messages) < 2:
                continue
            if not any(message["role"] == "assistant" for message in messages):
                continue

            row = {
                "messages": messages,
                "source": example.get("source"),
            }
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")
            count += 1
            if count >= max_examples:
                break

    return count


def main():
    args = parse_args()
    out_path = Path(args.out)
    ensure_parent(out_path)

    if args.dataset == "smol_smoltalk":
        count = export_smol_smoltalk(out_path, args.split, args.max_examples)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    print(f"wrote {count} chat records to {out_path}")


if __name__ == "__main__":
    main()
