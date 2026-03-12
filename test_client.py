import asyncio
import httpx


async def main():
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(10):
            tasks.append(
                client.post(
                    "http://127.0.0.1:8000/generate",
                    json={"prompt": f"req{i}", "max_new_tokens": 5},
                )
            )

        responses = await asyncio.gather(*tasks)

        for r in responses:
            print(r.json())


asyncio.run(main())
