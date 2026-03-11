import asyncio
import time
import random
from typing import Tuple, Dict

from .types import InferenceRequest, InferenceResult


class MiniInferenceEngine:
    def __init__(self, max_requests):
        self.pending = asyncio.Queue(max_requests)
        self.active: dict[int, asyncio.Future] = {}
        self.current_request_id = 0
        self.engine_task = None

    async def start(self):
        self.engine_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        self.engine_task.cancel()
        try:
            await self.engine_task
        except asyncio.CancelledError:
            pass

    async def submit_request(self, prompt: str, max_new_tokens: int) -> InferenceResult:
        request = InferenceRequest(self.current_request_id, prompt, max_new_tokens, time.time())
        self.current_request_id += 1
        future = asyncio.Future()

        self.active[request.request_id] = future

        await self.pending.put(request)

        result = await future
        return InferenceResult(request.request_id, result, "")

    async def _engine_loop(self):
        while True:
            request = await self.pending.get()
            request_id = request.request_id
            future = self.active[request_id]
            await asyncio.sleep(2)
            future.set_result(f"fake completion for: {request.prompt}")
            self.active.pop(request_id)
