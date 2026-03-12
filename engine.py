import asyncio
import time
import itertools

from typing import List

from .types import InferenceRequest, InferenceResult


class MiniInferenceEngine:
    def __init__(self, max_requests, batch_size=1):
        self.batch_size = batch_size
        self.max_wait_time = 0.05

        self.pending = asyncio.Queue(max_requests)
        self.batches = asyncio.Queue(max_requests)
        self.active: dict[int, asyncio.Future] = {}
        self.engine_tasks: List[asyncio.Task] = []
        self._request_id_gen = itertools.count(1)

    async def start(self):
        if self.engine_tasks:
            return

        engine_loop_task = asyncio.create_task(self._engine_loop())
        batcher_task = asyncio.create_task(self._batching())
        self.engine_tasks = [engine_loop_task, batcher_task]

    async def stop(self):
        for task in self.engine_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.engine_tasks = []

        for future in self.active.values():
            if not future.done():
                future.set_exception(RuntimeError("engine stopped"))
        self.active.clear()

    async def submit_request(self, prompt: str, max_new_tokens: int) -> InferenceResult:
        req_id = next(self._request_id_gen)
        now = time.time()
        request = InferenceRequest(req_id, prompt, max_new_tokens, now)
        print(f"[request {req_id}] arrived at {now:.6f}")

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        self.active[req_id] = future
        try:
            await self.pending.put(request)
            result = await future
            return result
        finally:
            self.active.pop(request.request_id, None)

    async def _batching(self):
        while True:
            batch = []
            first_request = await self.pending.get()
            first_arrival_time = first_request.arrival_time
            deadline = first_arrival_time + self.max_wait_time
            batch.append(first_request)

            while len(batch) < self.batch_size:
                try:
                    req = self.pending.get_nowait()
                    batch.append(req)
                except asyncio.QueueEmpty:
                    break

            while len(batch) < self.batch_size:
                now = time.time()
                remaining = deadline - now
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.pending.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            batch_start_time = time.time()
            for request in batch:
                request.batch_start_time = batch_start_time
            request_ids = [request.request_id for request in batch]
            print(f"[batch] start={batch_start_time:.6f} size={len(batch)} requests={request_ids}")
            await self.batches.put(batch)

    async def _process(self, batch):
        for request in batch:
            request_id = request.request_id

            future = self.active.get(request_id)
            if future is None or future.done():
                continue

            try:
                completion_time = time.time()
                batch_start_time = request.batch_start_time or completion_time
                future.set_result(
                    InferenceResult(
                        request_id=request.request_id,
                        text=f"fake completion for: {request.prompt}",
                        finish_reason="completed",
                        arrival_time=request.arrival_time,
                        batch_start_time=batch_start_time,
                        completion_time=completion_time,
                        batching_delay=batch_start_time - request.arrival_time,
                        processing_delay=completion_time - batch_start_time,
                    )
                )
                print(
                    f"[request {request_id}] completed at {completion_time:.6f} "
                    f"batching_delay={batch_start_time - request.arrival_time:.6f}s "
                    f"processing_delay={completion_time - batch_start_time:.6f}s"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                self.active.pop(request_id, None)

    async def _engine_loop(self):
        while True:
            batch = await self.batches.get()
            await self._process(batch)
