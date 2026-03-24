# Mini Infer

Small continuous-batching inference playground with two model backends:

- `hf`: real Hugging Face causal LMs like Qwen
- `toy`: a small local GPT for learning and experimentation

## Repo Layout

- root: app runtime and package modules
- `models/`: toy model implementations
- `training/`: dataset prep and toy model training utilities
- `scripts/`: benchmark and local client helpers
- `data/generated/`: generated corpora and downloaded datasets

## Run The Server

Real model:

```bash
PYTHONPATH=.. MODEL_BACKEND=hf MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
.venv/bin/python -m uvicorn mini_infer.app:app
```

Toy model:

```bash
PYTHONPATH=.. MODEL_BACKEND=toy .venv/bin/python -m uvicorn mini_infer.app:app
```

Toy model from a trained checkpoint:

```bash
PYTHONPATH=.. MODEL_BACKEND=toy MODEL_NAME=checkpoints/toy-gpt-d256-l4-h8-seq256-step120k.pt \
.venv/bin/python -m uvicorn mini_infer.app:app
```

## Train The Toy Model

The trainer expects one plain-text file.

Quick start on Apple Silicon:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 5000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --out checkpoints/toy-gpt-d256-l4-h8-seq256.pt
```

Larger but still practical config for an M1 Ultra:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 10000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 512 \
  --n-layers 6 \
  --n-head 8 \
  --out checkpoints/toy-gpt-d512-l6-h8-seq256.pt
```

If MPS memory gets tight, drop `--batch-size` to `16`.

Resume a longer run:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 10000 \
  --batch-size 32 \
  --resume checkpoints/toy-gpt-d256-l4-h8-seq256.pt \
  --out checkpoints/toy-gpt-d256-l4-h8-seq256.pt
```

A full overnight run:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 120000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --eval-every 500 \
  --eval-batches 10 \
  --sample-every 2000 \
  --sample-tokens 120 \
  --out checkpoints/toy-gpt-d256-l4-h8-seq256-step120k.pt
```

Aggressive 10-hour run on an M1 Ultra:

Use this if you want the strongest practical pretrain run, not just the biggest model:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train_full.txt \
  --device mps \
  --steps 80000 \
  --batch-size 4 \
  --seq-len 256 \
  --d-embed 1280 \
  --n-layers 16 \
  --n-head 20 \
  --eval-every 500 \
  --eval-batches 10 \
  --sample-every 2000 \
  --sample-tokens 120 \
  --out checkpoints/toy-gpt-d1280-l16-h20-seq256-step80k.pt
```

Max 10-hour run I would try on this setup:

Use this if you want to push the machine harder and are willing to trade some training efficiency for a larger model:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train_full.txt \
  --device mps \
  --steps 50000 \
  --batch-size 2 \
  --seq-len 256 \
  --d-embed 1536 \
  --n-layers 18 \
  --n-head 24 \
  --eval-every 500 \
  --eval-batches 10 \
  --sample-every 2000 \
  --sample-tokens 120 \
  --out checkpoints/toy-gpt-d1536-l18-h24-seq256-step50k.pt
```

Resume the same run later with the same checkpoint:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 160000 \
  --batch-size 32 \
  --resume checkpoints/toy-gpt-d256-l4-h8-seq256-step120k.pt \
  --out checkpoints/toy-gpt-d256-l4-h8-seq256-step160k.pt
```

If you change `--seq-len`, `--d-embed`, `--n-layers`, or `--n-head`, start a new checkpoint instead of resuming:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 60000 \
  --batch-size 16 \
  --seq-len 256 \
  --d-embed 512 \
  --n-layers 6 \
  --n-head 8 \
  --eval-every 500 \
  --eval-batches 10 \
  --sample-every 2000 \
  --sample-tokens 120 \
  --out checkpoints/toy-gpt-d512-l6-h8-seq256-step60k.pt
```

Checkpoint naming tip:

- Use filenames that encode the shape, for example `toy-gpt-d256-l4-h8-seq256-step120k.pt`
- If you change `--seq-len`, `--d-embed`, `--n-layers`, or `--n-head`, start a new checkpoint name

## Prepare A Better Training Corpus

Install the dataset helper first:

```bash
.venv/bin/pip install datasets
```

Export a stronger starter corpus from TinyStories:

```bash
.venv/bin/python -m training.prepare_toy_data \
  --dataset tinystories \
  --split train \
  --max-examples 50000 \
  --out data/generated/tinystories_train.txt
```

If you already downloaded TinyStories locally with `hf download`, convert the local train file into your own generated corpus:

```bash
.venv/bin/python -m training.prepare_toy_data \
  --dataset tinystories_local \
  --source-file data/generated/TinyStories/TinyStories-train.txt \
  --max-examples 50000 \
  --out data/generated/tinystories_train.txt
```

And for a validation file:

```bash
.venv/bin/python -m training.prepare_toy_data \
  --dataset tinystories_local \
  --source-file data/generated/TinyStories/TinyStories-valid.txt \
  --max-examples 5000 \
  --out data/generated/tinystories_valid.txt
```

You can also build a corpus from local text files:

```bash
.venv/bin/python -m training.prepare_toy_data \
  --dataset local_text \
  --local-glob 'data/raw/*.txt' \
  --out data/generated/local_corpus.txt
```

## Prepare Chat SFT Data

For a chat-oriented toy model, export `smol-smoltalk` as JSONL:

```bash
.venv/bin/python -m training.prepare_chat_data \
  --dataset smol_smoltalk \
  --split train \
  --max-examples 50000 \
  --out data/generated/smol_smoltalk_train.jsonl
```

## SFT The Toy Model

Train on assistant responses only:

```bash
.venv/bin/python -m training.train_toy_sft \
  --data-file data/generated/smol_smoltalk_train.jsonl \
  --device mps \
  --steps 20000 \
  --batch-size 16 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --eval-every 200 \
  --sample-every 1000 \
  --out checkpoints/toy-gpt-sft-d256-l4-h8-seq256.pt
```

Resume:

```bash
.venv/bin/python -m training.train_toy_sft \
  --data-file data/generated/smol_smoltalk_train.jsonl \
  --device mps \
  --steps 40000 \
  --batch-size 16 \
  --resume checkpoints/toy-gpt-sft-d256-l4-h8-seq256.pt \
  --out checkpoints/toy-gpt-sft-d256-l4-h8-seq256-step40k.pt
```

Best overnight path from this repo:

1. Pretrain the toy model on TinyStories:

```bash
.venv/bin/python -m training.train_toy \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 120000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --eval-every 500 \
  --eval-batches 10 \
  --sample-every 2000 \
  --sample-tokens 120 \
  --out checkpoints/toy-gpt-d256-l4-h8-seq256-step120k.pt
```

2. Then initialize SFT from that checkpoint and train on `smol-smoltalk`:

```bash
.venv/bin/python -m training.train_toy_sft \
  --data-file data/generated/smol_smoltalk_train.jsonl \
  --device mps \
  --steps 30000 \
  --batch-size 16 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --init-from checkpoints/toy-gpt-d256-l4-h8-seq256-step120k.pt \
  --eval-every 200 \
  --sample-every 1000 \
  --out checkpoints/toy-gpt-sft-d256-l4-h8-seq256-step30k.pt
```

## Benchmark

```bash
.venv/bin/python -m scripts.benchmark
```

Only benchmark `/generate`:

```bash
.venv/bin/python -m scripts.benchmark --mode generate
```

Tune request load:

```bash
.venv/bin/python -m scripts.benchmark \
  --num-requests 64 \
  --concurrency 16 \
  --max-new-tokens 64
```

## Test Client

Quick manual test:

```bash
.venv/bin/python -m scripts.test_client
```

Use your own prompts file:

```bash
.venv/bin/python -m scripts.test_client \
  --prompts-file data/raw/prompts.txt \
  --max-new-tokens 32
```

## EC2 CUDA Setup

From the repo root:

1. Create and activate a venv.
2. Install a CUDA-compatible PyTorch build from the official PyTorch selector.
3. Install the rest of the repo requirements.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install torch for your CUDA version from the official selector:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Then start the server:

```bash
PYTHONPATH=.. MODEL_BACKEND=hf MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
python -m uvicorn mini_infer.app:app --host 0.0.0.0 --port 8000
```

Notes:

- `requirements.txt` intentionally does not pin `torch`, because the right wheel depends on the EC2 CUDA stack.
- `protobuf` is included because some Hugging Face models need it at load time.
- CPU-only and CUDA instances both work with the current loader path.
