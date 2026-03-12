import asyncio
import httpx


def format_result(data: dict) -> str:
    total = data["completion_time"] - data["arrival_time"]
    return (
        f"req {data['request_id']} "
        f"wait={data['batching_delay']:.2f}s "
        f"run={data['processing_delay']:.2f}s "
        f"total={total:.2f}s "
        f"text={data['text']}"
    )


async def send_one(client: httpx.AsyncClient, i: int, max_new_tokens: int):
    r = await client.post(
        "http://127.0.0.1:8000/generate",
        json={
            "prompt": f"req{i}",
            "max_new_tokens": max_new_tokens,
        },
    )
    print(format_result(r.json()))


async def main():
    async with httpx.AsyncClient() as client:
        tasks = [
            send_one(client, 0, 2),
            send_one(client, 1, 5),
            send_one(client, 2, 3),
            send_one(client, 3, 7),
            send_one(client, 4, 2),
            send_one(client, 5, 6),
        ]
        await asyncio.gather(*tasks)


asyncio.run(main())
