import asyncio
import time
import itertools
import random

from typing import List

from .types import InferenceRequest, InferenceResult, ActiveRequest


class MiniInferenceEngine:
    def __init__(self, max_requests, batch_size=1):
        self.batch_size = batch_size
        self.max_wait_time = 0.05
        self._request_id_gen = itertools.count(1)

        self.pending = asyncio.Queue(max_requests)
        self.active: dict[int, asyncio.Future] = {}
        self.active_slots = []

        self.scheduler_task = None

    async def start(self):
        if self.scheduler_task is not None and not self.scheduler_task.done():
            return
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        if self.scheduler_task is not None and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        self.scheduler_task = None

        for fut in self.active.values():
            if not fut.done():
                fut.set_exception(RuntimeError("engine stopped"))
        self.active.clear()
        self.active_slots.clear()

    async def submit_request(self, prompt: str, max_new_tokens: int) -> InferenceResult:
        req_id = next(self._request_id_gen)
        now = time.time()
        request = InferenceRequest(req_id, prompt, max_new_tokens, now)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        self.active[req_id] = future
        try:
            await self.pending.put(request)
            result = await future
            return result
        finally:
            self.active.pop(request.request_id, None)

    async def _scheduler_loop(self):
        while True:
            await self._maybe_wait_for_first_request()
            await self._admit_new_requests()
            await self._run_decode_step()
            self._finalize_finished_requests()

    async def _maybe_wait_for_first_request(self):
        if self.active_slots:
            return
        request = await self.pending.get()
        active_request = ActiveRequest(request)
        self.active_slots.append(active_request)

    async def _admit_new_requests(self):
        while len(self.active_slots) < self.batch_size:
            try:
                request = self.pending.get_nowait()
            except asyncio.QueueEmpty:
                break
            request.batch_start_time = time.time()
            active_request = ActiveRequest(request)
            self.active_slots.append(active_request)

    async def _run_decode_step(self):
        if not self.active_slots:
            return
        await asyncio.sleep(random.randint(0, 100) / 100)
        for r in self.active_slots:
            r.generated_tokens += 1
            r.output_text += f"<tok {r.generated_tokens}>"

    def _finalize_finished_requests(self):
        still_active = []
        for active_r in self.active_slots:
            r = active_r.request
            r_id = r.request_id

            if active_r.generated_tokens < r.max_new_tokens:
                still_active.append(active_r)
                continue

            fut = self.active.get(r_id)
            if fut is None or fut.done():
                continue

            completion_time = time.time()
            batch_start_time = r.batch_start_time or completion_time
            fut.set_result(
                InferenceResult(
                    request_id=r.request_id,
                    text=f"{r.prompt} -> {active_r.output_text}",
                    finish_reason="completed",
                    arrival_time=r.arrival_time,
                    batch_start_time=batch_start_time,
                    completion_time=completion_time,
                    batching_delay=batch_start_time - r.arrival_time,
                    processing_delay=completion_time - batch_start_time,
                )
            )
            self.active.pop(r_id, None)

        self.active_slots = still_active
