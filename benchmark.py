import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PROMPT = os.getenv("PROMPT", "Explain what continuous batching is in one short paragraph.")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))

NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "32"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))
TIMEOUT = float(os.getenv("TIMEOUT", "120.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "0.25"))


@dataclass
class GenerateMetrics:
    request_id: Optional[int]
    ok: bool
    error: Optional[str]

    latency: float
    output_tokens: int

    batching_delay: Optional[float] = None
    processing_delay: Optional[float] = None
    finish_reason: Optional[str] = None
    retries: int = 0


@dataclass
class StreamMetrics:
    request_id: Optional[int]
    ok: bool
    error: Optional[str]

    ttft: Optional[float]
    latency: float
    output_tokens: int
    finish_reason: Optional[str] = None
    retries: int = 0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def retry_delay(attempt: int) -> float:
    return RETRY_BACKOFF_SEC * (2**attempt)


async def bench_generate_one(client: httpx.AsyncClient, prompt: str, max_new_tokens: int) -> GenerateMetrics:
    start = time.perf_counter()
    retries = 0
    while True:
        try:
            resp = await client.post(
                f"{BASE_URL}/generate",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            )
            latency = time.perf_counter() - start

            resp.raise_for_status()
            data = resp.json()

            return GenerateMetrics(
                request_id=data.get("request_id"),
                ok=True,
                error=None,
                latency=latency,
                output_tokens=data["generated_tokens"],
                batching_delay=data.get("batching_delay"),
                processing_delay=data.get("processing_delay"),
                finish_reason=data.get("finish_reason"),
                retries=retries,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 503 and retries < MAX_RETRIES:
                await asyncio.sleep(retry_delay(retries))
                retries += 1
                continue

            latency = time.perf_counter() - start
            return GenerateMetrics(
                request_id=None,
                ok=False,
                error=str(error),
                latency=latency,
                output_tokens=0,
                retries=retries,
            )
        except Exception as error:
            latency = time.perf_counter() - start
            return GenerateMetrics(
                request_id=None,
                ok=False,
                error=str(error),
                latency=latency,
                output_tokens=0,
                retries=retries,
            )


async def bench_stream_one(client: httpx.AsyncClient, prompt: str, max_new_tokens: int) -> StreamMetrics:
    start = time.perf_counter()
    retries = 0
    while True:
        first_token_time = None
        output_tokens = 0
        request_id = None
        finish_reason = None
        server_generated_tokens = None

        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/generate_stream",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue

                    payload = json.loads(line[len("data: ") :])

                    if "token" in payload:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens += 1

                    if payload.get("done"):
                        request_id = payload.get("request_id")
                        finish_reason = payload.get("finish_reason")
                        server_generated_tokens = payload.get("generated_tokens")

            latency = time.perf_counter() - start
            ttft = None if first_token_time is None else (first_token_time - start)

            if server_generated_tokens is not None and server_generated_tokens != output_tokens:
                print(
                    f"[WARN] stream token mismatch: counted={output_tokens}, "
                    f"server={server_generated_tokens}, request_id={request_id}"
                )

            return StreamMetrics(
                request_id=request_id,
                ok=True,
                error=None,
                ttft=ttft,
                latency=latency,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retries=retries,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 503 and retries < MAX_RETRIES:
                await asyncio.sleep(retry_delay(retries))
                retries += 1
                continue

            latency = time.perf_counter() - start
            return StreamMetrics(
                request_id=None,
                ok=False,
                error=str(error),
                ttft=None,
                latency=latency,
                output_tokens=0,
                retries=retries,
            )
        except Exception as error:
            latency = time.perf_counter() - start
            return StreamMetrics(
                request_id=None,
                ok=False,
                error=str(error),
                ttft=None,
                latency=latency,
                output_tokens=0,
                retries=retries,
            )


async def run_with_limit(coro_factory, num_requests: int, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(i: int):
        async with sem:
            return await coro_factory(i)

    tasks = [asyncio.create_task(wrapped(i)) for i in range(num_requests)]
    return await asyncio.gather(*tasks)


def print_generate_summary(results: list[GenerateMetrics], total_wall_time: float):
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]

    latencies = [r.latency for r in ok]
    token_counts = [r.output_tokens for r in ok]
    batching_delays = [r.batching_delay for r in ok if r.batching_delay is not None]
    processing_delays = [r.processing_delay for r in ok if r.processing_delay is not None]
    retries = [r.retries for r in ok]

    total_tokens = sum(token_counts)
    rps = len(ok) / total_wall_time if total_wall_time > 0 else 0.0
    tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    print("\n=== /generate summary ===")
    print(f"successful requests: {len(ok)}")
    print(f"failed requests:     {len(fail)}")
    print(f"total wall time:     {total_wall_time:.3f}s")
    print(f"requests/sec:        {rps:.2f}")
    print(f"tokens/sec:          {tps:.2f}")

    if latencies:
        print(f"latency avg:         {statistics.mean(latencies):.3f}s")
        print(f"latency p50:         {percentile(latencies, 0.50):.3f}s")
        print(f"latency p95:         {percentile(latencies, 0.95):.3f}s")

    if batching_delays:
        print(f"batching avg:        {statistics.mean(batching_delays):.3f}s")
    if processing_delays:
        print(f"processing avg:      {statistics.mean(processing_delays):.3f}s")
    if retries:
        print(f"retry avg:           {statistics.mean(retries):.2f}")

    if fail:
        print("\nSample errors:")
        for r in fail[:3]:
            print(f"  - {r.error}")


def print_stream_summary(results: list[StreamMetrics], total_wall_time: float):
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]

    latencies = [r.latency for r in ok]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    token_counts = [r.output_tokens for r in ok]
    retries = [r.retries for r in ok]

    total_tokens = sum(token_counts)
    rps = len(ok) / total_wall_time if total_wall_time > 0 else 0.0
    tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    print("\n=== /generate_stream summary ===")
    print(f"successful requests: {len(ok)}")
    print(f"failed requests:     {len(fail)}")
    print(f"total wall time:     {total_wall_time:.3f}s")
    print(f"requests/sec:        {rps:.2f}")
    print(f"tokens/sec:          {tps:.2f}")

    if ttfts:
        print(f"TTFT avg:            {statistics.mean(ttfts):.3f}s")
        print(f"TTFT p50:            {percentile(ttfts, 0.50):.3f}s")
        print(f"TTFT p95:            {percentile(ttfts, 0.95):.3f}s")

    if latencies:
        print(f"latency avg:         {statistics.mean(latencies):.3f}s")
        print(f"latency p50:         {percentile(latencies, 0.50):.3f}s")
        print(f"latency p95:         {percentile(latencies, 0.95):.3f}s")
    if retries:
        print(f"retry avg:           {statistics.mean(retries):.2f}")

    if fail:
        print("\nSample errors:")
        for r in fail[:3]:
            print(f"  - {r.error}")


async def run_generate_benchmark():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        async def one(_i: int):
            return await bench_generate_one(client, PROMPT, MAX_NEW_TOKENS)

        wall_start = time.perf_counter()
        results = await run_with_limit(one, NUM_REQUESTS, CONCURRENCY)
        total_wall_time = time.perf_counter() - wall_start

    print_generate_summary(results, total_wall_time)


async def run_stream_benchmark():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        async def one(_i: int):
            return await bench_stream_one(client, PROMPT, MAX_NEW_TOKENS)

        wall_start = time.perf_counter()
        results = await run_with_limit(one, NUM_REQUESTS, CONCURRENCY)
        total_wall_time = time.perf_counter() - wall_start

    print_stream_summary(results, total_wall_time)


async def main():
    print(f"BASE_URL={BASE_URL}")
    print(
        f"NUM_REQUESTS={NUM_REQUESTS}, CONCURRENCY={CONCURRENCY}, "
        f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}, MAX_RETRIES={MAX_RETRIES}"
    )

    await run_generate_benchmark()
    await run_stream_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
