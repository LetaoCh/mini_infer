import asyncio
import itertools
import time

from .types import InferenceRequest, InferenceResult, ActiveRequest


class MiniInferenceEngine:
    def __init__(self, max_requests: int, batch_size: int = 4):
        self.batch_size = batch_size
        self.pending: asyncio.Queue[InferenceRequest] = asyncio.Queue(max_requests)
        self.active: dict[int, asyncio.Future] = {}
        self.active_slots: list[ActiveRequest] = []

        self.scheduler_task: asyncio.Task | None = None
        self._request_id_gen = itertools.count(1)

        # Stage 3 knobs
        self.decode_step_time = 0.2
        self.max_admit_per_tick = batch_size

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

        request = InferenceRequest(
            request_id=req_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            arrival_time=now,
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        self.active[req_id] = future
        try:
            await self.pending.put(request)
            print(f"req {req_id} queued")
            result = await future
            return result
        finally:
            self.active.pop(req_id, None)

    async def _scheduler_loop(self):
        while True:
            if not self.active_slots:
                first_req = await self.pending.get()
                first_req.batch_start_time = time.time()
                self.active_slots.append(ActiveRequest(request=first_req))

            await self._admit_new_requests()

            await self._run_decode_step()

            self._finalize_finished_requests()

    async def _admit_new_requests(self):
        admitted = 0
        while (
            len(self.active_slots) < self.batch_size
            and admitted < self.max_admit_per_tick
        ):
            try:
                req = self.pending.get_nowait()
            except asyncio.QueueEmpty:
                break

            future = self.active.get(req.request_id)
            if future is None or future.done() or future.cancelled():
                continue

            req.batch_start_time = time.time()
            self.active_slots.append(ActiveRequest(request=req))
            print(
                f"req {req.request_id} started"
                f" after {req.batch_start_time - req.arrival_time:.2f}s"
            )
            admitted += 1

    async def _run_decode_step(self):
        if not self.active_slots:
            return

        # fake one decode iteration for all active requests
        await asyncio.sleep(self.decode_step_time)

        for slot in self.active_slots:
            slot.generated_tokens += 1
            slot.output_text += f"<tok{slot.generated_tokens}>"

    def _finalize_finished_requests(self):
        still_active: list[ActiveRequest] = []

        for slot in self.active_slots:
            req = slot.request
            future = self.active.get(req.request_id)

            if future is None or future.done() or future.cancelled():
                continue

            if slot.generated_tokens >= req.max_new_tokens:
                completion_time = time.time()
                batch_start_time = req.batch_start_time or completion_time

                future.set_result(
                    InferenceResult(
                        request_id=req.request_id,
                        text=f"{req.prompt} -> {slot.output_text}",
                        finish_reason="completed",
                        arrival_time=req.arrival_time,
                        batch_start_time=batch_start_time,
                        completion_time=completion_time,
                        batching_delay=batch_start_time - req.arrival_time,
                        processing_delay=completion_time - batch_start_time,
                    )
                )
                print(
                    f"req {req.request_id} done"
                    f" wait={batch_start_time - req.arrival_time:.2f}s"
                    f" run={completion_time - batch_start_time:.2f}s"
                )
            else:
                still_active.append(slot)

        self.active_slots = still_active
