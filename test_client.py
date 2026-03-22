import asyncio
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


async def fetch_stats(client: httpx.AsyncClient, label: str):
    response = await client.get(f"{BASE_URL}/stats.json")
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


async def send_one(client: httpx.AsyncClient, index: int, prompt: str, max_new_tokens: int):
    try:
        response = await client.post(
            f"{BASE_URL}/generate",
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


async def main():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        await fetch_stats(client, "before")

        tasks = [
            send_one(client, index, prompt, max_new_tokens)
            for index, (prompt, max_new_tokens) in enumerate(REQUESTS, start=1)
        ]
        await asyncio.gather(*tasks)

        await fetch_stats(client, "after")


if __name__ == "__main__":
    asyncio.run(main())
