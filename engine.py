import asyncio
import itertools
import time

from typing import List

from .types import InferenceRequest, InferenceResult, OverloadedError, ActiveRequest


class MiniInferenceEngine:
    def __init__(self, max_requests, batch_size=1):
        self.batch_size = batch_size
        self.max_wait_time = 0.05
        self.decode_step_time = 0.05  # fixed decode latency
        self._request_id_gen = itertools.count(1)

        self.pending = asyncio.Queue(max_requests)
        self.active: dict[int, asyncio.Future] = {}
        self.active_slots: List[ActiveRequest] = []

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

        try:
            self.pending.put_nowait(request)
        except asyncio.QueueFull:
            raise OverloadedError("system full, try again later")

        self.active[req_id] = future
        try:
            print(f"[QUEUE] request {req_id} queued")
            result = await future
            return result
        finally:
            self.active.pop(request.request_id, None)

    async def _scheduler_loop(self):
        while True:
            if not self.active_slots:
                await self._maybe_wait_for_first_request()
                await self._admit_new_requests(initial_fill=True)
            else:
                await self._admit_new_requests(initial_fill=False)

            await self._run_decode_step()
            self._finalize_finished_requests()

    async def _maybe_wait_for_first_request(self):
        if self.active_slots:
            return

        request = await self.pending.get()
        request.admit_time = time.time()
        self.active_slots.append(ActiveRequest(request))

        print(f"[ADMIT] request {request.request_id} admitted (first slot)")

    async def _admit_new_requests(self, initial_fill: bool):
        deadline = None
        if initial_fill and self.active_slots:
            first_req = self.active_slots[0].request
            base_time = first_req.admit_time or time.time()
            deadline = base_time + self.max_wait_time

        while len(self.active_slots) < self.batch_size:
            try:
                request = self.pending.get_nowait()
            except asyncio.QueueEmpty:
                if not initial_fill or deadline is None:
                    break

                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                try:
                    request = await asyncio.wait_for(self.pending.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

            request.admit_time = time.time()
            self.active_slots.append(ActiveRequest(request))
            print(f"[ADMIT] request {request.request_id} admitted")

    async def _run_decode_step(self):
        if not self.active_slots:
            return

        tick_start = time.time()
        for active_r in self.active_slots:
            if active_r.request.service_start_time is None:
                active_r.request.service_start_time = tick_start

        print(f"[DECODE] tick | active={len(self.active_slots)}")
        await asyncio.sleep(self.decode_step_time)

        for active_r in self.active_slots:
            active_r.generated_tokens += 1
            active_r.output_text += f"<tok {active_r.generated_tokens}>"

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
            service_start_time = r.service_start_time or completion_time

            fut.set_result(
                InferenceResult(
                    request_id=r.request_id,
                    text=f"{r.prompt} -> {active_r.output_text}",
                    finish_reason="completed",
                    arrival_time=r.arrival_time,
                    service_start_time=service_start_time,
                    completion_time=completion_time,
                    batching_delay=service_start_time - r.arrival_time,
                    processing_delay=completion_time - service_start_time,
                )
            )

            print(f"[FINISH] request {r_id} finished")
            self.active.pop(r_id, None)

        self.active_slots = still_active
