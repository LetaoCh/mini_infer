import asyncio
import argparse
import os

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 30.0
REQUESTS = [
    ("What is batching?", 12),
    ("Explain speculative decoding briefly.", 18),
    ("Write a haiku about GPUs.", 10),
    ("Why does continuous batching help throughput?", 16),
    ("Give one sentence about KV cache.", 12),
    ("What is prefill vs decode?", 16),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Send a few simple test requests to the server.")
    parser.add_argument("--base-url", default=BASE_URL, help="Server base URL.")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Request timeout in seconds.")
    parser.add_argument(
        "--prompts-file",
        help="Optional text file with one prompt per line. Uses defaults when omitted.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for every prompt.",
    )
    return parser.parse_args()


def format_result(data: dict) -> str:
    total = data["completion_time"] - data["arrival_time"]
    preview = data["text"].replace("\n", " ")
    return (
        f"req={data['request_id']} "
        f"reason={data['finish_reason']} "
        f"tokens={data['generated_tokens']} "
        f"wait={data['batching_delay']:.2f}s "
        f"run={data['processing_delay']:.2f}s "
        f"total={total:.2f}s "
        f"text={preview}"
    )


async def fetch_stats(client: httpx.AsyncClient, base_url: str, label: str):
    response = await client.get(f"{base_url}/stats.json")
    response.raise_for_status()
    data = response.json()
    print(
        f"[{label}] submitted={data['submitted_total']} "
        f"completed={data['completed_total']} "
        f"rejected={data['rejected_total']} "
        f"pending={data['pending_queue']} "
        f"active={data['active_requests']} "
        f"prefill={data['prefill_mode']} "
        f"decode={data['decode_mode']}"
    )


async def send_one(client: httpx.AsyncClient, base_url: str, index: int, prompt: str, max_new_tokens: int):
    try:
        response = await client.post(
            f"{base_url}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = error.response.text.strip()
        print(f"req={index} status={error.response.status_code} error={detail}")
        return
    except Exception as error:
        print(f"req={index} error={error}")
        return

    print(format_result(response.json()))


def load_requests(prompts_file: str | None, max_new_tokens: int | None):
    if prompts_file is None:
        if max_new_tokens is None:
            return REQUESTS
        return [(prompt, max_new_tokens) for prompt, _tokens in REQUESTS]

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)

    token_budget = max_new_tokens if max_new_tokens is not None else 32
    return [(prompt, token_budget) for prompt in prompts]


async def main():
    args = parse_args()
    requests = load_requests(args.prompts_file, args.max_new_tokens)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        await fetch_stats(client, args.base_url, "before")

        tasks = [
            send_one(client, args.base_url, index, prompt, max_new_tokens)
            for index, (prompt, max_new_tokens) in enumerate(requests, start=1)
        ]
        await asyncio.gather(*tasks)

        await fetch_stats(client, args.base_url, "after")


if __name__ == "__main__":
    asyncio.run(main())
