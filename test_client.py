import asyncio
import httpx


async def send_one(client: httpx.AsyncClient, i: int, max_new_tokens: int):
    r = await client.post(
        "http://127.0.0.1:8000/generate",
        json={
            "prompt": f"req{i}",
            "max_new_tokens": max_new_tokens,
        },
    )
    print(r.json())


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