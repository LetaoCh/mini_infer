import argparse
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare plain-text corpora for train_toy.py.")
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "tinystories_local", "local_text"],
        default="tinystories",
        help="Dataset source to export.",
    )
    parser.add_argument("--split", default="train", help="Dataset split for Hugging Face datasets.")
    parser.add_argument("--out", default="data/generated/tinystories_train.txt", help="Output text file.")
    parser.add_argument("--max-examples", type=int, default=50000, help="Maximum records to export.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--local-glob",
        default="data/raw/*.txt",
        help="Glob used when --dataset=local_text.",
    )
    parser.add_argument(
        "--source-file",
        default="data/generated/TinyStories/TinyStories-train.txt",
        help="Source text file used when --dataset=tinystories_local.",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def export_tinystories(out_path: Path, split: str, max_examples: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with '.venv/bin/pip install datasets'."
        ) from exc

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=min(max_examples * 2, 10000))

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            text = (example.get("text") or "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            count += 1
            if count >= max_examples:
                break

    return count


def export_local_text(out_path: Path, local_glob: str):
    files = sorted(Path().glob(local_glob))
    if not files:
        raise ValueError(f"no files matched glob: {local_glob}")

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            count += 1

    return count


def export_local_tinystories(out_path: Path, source_file: str, max_examples: int):
    source_path = Path(source_file)
    if not source_path.exists():
        raise ValueError(f"source file does not exist: {source_file}")

    count = 0
    marker = "<|endoftext|>"
    buffer: list[str] = []
    with out_path.open("w", encoding="utf-8") as f:
        with source_path.open("r", encoding="utf-8", errors="ignore") as src:
            for line in src:
                if line.strip() == marker:
                    text = "".join(buffer).strip()
                    buffer.clear()
                    if not text:
                        continue
                    f.write(text)
                    f.write("\n\n")
                    count += 1
                    if count >= max_examples:
                        break
                    continue
                buffer.append(line)

            if count < max_examples and buffer:
                text = "".join(buffer).strip()
                if text:
                    f.write(text)
                    f.write("\n\n")
                    count += 1

    return count


def main():
    args = parse_args()
    random.seed(args.seed)

    out_path = Path(args.out)
    ensure_parent(out_path)

    if args.dataset == "tinystories":
        count = export_tinystories(
            out_path=out_path,
            split=args.split,
            max_examples=args.max_examples,
            seed=args.seed,
        )
    elif args.dataset == "tinystories_local":
        count = export_local_tinystories(
            out_path=out_path,
            source_file=args.source_file,
            max_examples=args.max_examples,
        )
    else:
        count = export_local_text(
            out_path=out_path,
            local_glob=args.local_glob,
        )

    print(f"wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()
