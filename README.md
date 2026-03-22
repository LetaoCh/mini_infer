# Mini Infer

Small continuous-batching inference playground with two model backends:

- `hf`: real Hugging Face causal LMs like Qwen
- `toy`: a small local GPT for learning and experimentation

## Run The Server

Real model:

```bash
MODEL_BACKEND=hf MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
.venv/bin/python -m uvicorn mini_infer.app:app
```

Toy model:

```bash
MODEL_BACKEND=toy .venv/bin/python -m uvicorn mini_infer.app:app
```

Toy model from a trained checkpoint:

```bash
MODEL_BACKEND=toy MODEL_NAME=checkpoints/toy-gpt.pt \
.venv/bin/python -m uvicorn mini_infer.app:app
```

## Train The Toy Model

The trainer expects one plain-text file.

Quick start on Apple Silicon:

```bash
.venv/bin/python train_toy.py \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 5000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 256 \
  --n-layers 4 \
  --n-head 8 \
  --out checkpoints/toy-gpt.pt
```

Larger but still practical config for an M1 Ultra:

```bash
.venv/bin/python train_toy.py \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 10000 \
  --batch-size 32 \
  --seq-len 256 \
  --d-embed 512 \
  --n-layers 6 \
  --n-head 8 \
  --out checkpoints/toy-gpt-512d.pt
```

If MPS memory gets tight, drop `--batch-size` to `16`.

Resume a longer run:

```bash
.venv/bin/python train_toy.py \
  --text-file data/generated/tinystories_train.txt \
  --device mps \
  --steps 10000 \
  --batch-size 32 \
  --resume checkpoints/toy-gpt.pt \
  --out checkpoints/toy-gpt.pt
```

## Prepare A Better Training Corpus

Install the dataset helper first:

```bash
.venv/bin/pip install datasets
```

Export a stronger starter corpus from TinyStories:

```bash
.venv/bin/python prepare_toy_data.py \
  --dataset tinystories \
  --split train \
  --max-examples 50000 \
  --out data/generated/tinystories_train.txt
```

If you already downloaded TinyStories locally with `hf download`, convert the local train file into your own generated corpus:

```bash
.venv/bin/python prepare_toy_data.py \
  --dataset tinystories_local \
  --source-file data/generated/TinyStories/TinyStories-train.txt \
  --max-examples 50000 \
  --out data/generated/tinystories_train.txt
```

And for a validation file:

```bash
.venv/bin/python prepare_toy_data.py \
  --dataset tinystories_local \
  --source-file data/generated/TinyStories/TinyStories-valid.txt \
  --max-examples 5000 \
  --out data/generated/tinystories_valid.txt
```

You can also build a corpus from local text files:

```bash
.venv/bin/python prepare_toy_data.py \
  --dataset local_text \
  --local-glob 'data/raw/*.txt' \
  --out data/generated/local_corpus.txt
```

## Benchmark

```bash
.venv/bin/python benchmark.py
```
