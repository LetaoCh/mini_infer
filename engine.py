import asyncio
import itertools
import time
import torch

from typing import List

from .model import load_model_bundle
from .types import (
    ActiveRequest,
    InferenceRequest,
    InferenceResult,
    OverloadedError,
    RequestContext,
)


class MiniInferenceEngine:
    def __init__(self, max_requests, batch_size=1, model_name=None):
        self.batch_size = batch_size
        self.max_wait_time = 0.05
        self._request_id_gen = itertools.count(1)

        self.pending = asyncio.Queue(max_requests)
        self.active: dict[int, RequestContext] = {}
        self.active_slots: List[ActiveRequest] = []

        self.scheduler_task = None

        self.submitted_total = 0
        self.completed_total = 0
        self.rejected_total = 0
        self.decode_ticks_total = 0

        bundle = load_model_bundle(model_name) if model_name else load_model_bundle()
        self.tokenizer = bundle.tokenizer
        self.model = bundle.model
        self.device = bundle.device
        self.model_name = bundle.model_name

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

        for ctx in self.active.values():
            if not ctx.future.done():
                ctx.future.set_exception(RuntimeError("engine stopped"))
            ctx.token_queue.put_nowait(None)

        self.active.clear()
        self.active_slots.clear()

    def _build_request(self, prompt: str, max_new_tokens: int):
        req_id = next(self._request_id_gen)
        now = time.time()

        request = InferenceRequest(req_id, prompt, max_new_tokens, now)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        ctx = RequestContext(request_id=req_id, future=future)

        return request, ctx

    def _enqueue_request(self, request: InferenceRequest, ctx: RequestContext) -> None:
        try:
            self.pending.put_nowait(request)
        except asyncio.QueueFull:
            self.rejected_total += 1
            raise OverloadedError("system full, try again later")

        self.active[request.request_id] = ctx
        self.submitted_total += 1
        print(f"[QUEUE] request {request.request_id} queued")

    async def submit_request(self, prompt: str, max_new_tokens: int) -> InferenceResult:
        request, ctx = self._build_request(prompt, max_new_tokens)
        self._enqueue_request(request, ctx)

        try:
            result = await ctx.future
            return result
        finally:
            self.active.pop(request.request_id, None)

    async def submit_streaming_request(self, prompt: str, max_new_tokens: int) -> RequestContext:
        request, ctx = self._build_request(prompt, max_new_tokens)
        self._enqueue_request(request, ctx)
        return ctx

    def cancel_request(self, request_id: int):
        ctx = self.active.get(request_id)
        if ctx is None:
            return
        ctx.cancelled = True

    async def _scheduler_loop(self):
        while True:
            if not self.active_slots:
                await self._maybe_wait_for_first_request()
                await self._admit_new_requests(initial_fill=True)
            else:
                await self._admit_new_requests(initial_fill=False)

            self._remove_cancelled_requests()
            await self._run_generation_step()
            self._finalize_finished_requests()

    async def _maybe_wait_for_first_request(self):
        if self.active_slots:
            return

        request = await self.pending.get()
        request.admit_time = time.time()
        self.active_slots.append(ActiveRequest(request=request))
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
            self.active_slots.append(ActiveRequest(request=request))
            print(f"[ADMIT] request {request.request_id} admitted")

    def _pick_next_token_id(self, next_token_logits):
        return torch.argmax(next_token_logits, dim=-1)

    def _pick_next_token_id_sampling(self, next_token_logits):
        probs = torch.softmax(next_token_logits / 0.8, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id

    async def _run_prefill_for_request(self, active_r, ctx):
        # prompt = active_r.request.prompt
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(self.device)
        # attention_mask = inputs["attention_mask"].to(self.device)

        prompt = active_r.request.prompt

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        next_token_logits = outputs.logits[:, -1, :]  # (B, T, C) -> (B, C)
        next_token_id = self._pick_next_token_id(next_token_logits)

        active_r.past_key_values = outputs.past_key_values
        active_r.last_token_id = next_token_id
        active_r.prefill_done = True

        generated_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        active_r.output_text += generated_token
        active_r.generated_tokens += 1
        ctx.token_queue.put_nowait(generated_token)

    async def _run_decode_for_request(self, active_r, ctx):
        decode_input_ids = active_r.last_token_id.unsqueeze(0)  # [1] -> [1,1]

        with torch.no_grad():
            outputs = self.model(
                input_ids=decode_input_ids,
                past_key_values=active_r.past_key_values,
                use_cache=True,
            )

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = self._pick_next_token_id(next_token_logits)

        active_r.past_key_values = outputs.past_key_values
        active_r.last_token_id = next_token_id

        generated_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        active_r.output_text += generated_token
        active_r.generated_tokens += 1
        ctx.token_queue.put_nowait(generated_token)

    async def _run_generation_step(self):
        if not self.active_slots:
            return

        tick_start = time.time()
        self.decode_ticks_total += 1

        for active_r in self.active_slots:
            if active_r.request.service_start_time is None:
                active_r.request.service_start_time = tick_start

        print(f"[PREFILL/DECODE] tick | active={len(self.active_slots)}")

        for active_r in self.active_slots:
            ctx = self.active.get(active_r.request.request_id)
            if ctx is None:
                continue

            if not active_r.prefill_done:
                await self._run_prefill_for_request(active_r, ctx)
            else:
                await self._run_decode_for_request(active_r, ctx)

    def _finalize_finished_requests(self):
        still_active = []

        for active_r in self.active_slots:
            r = active_r.request
            r_id = r.request_id

            last_token_id = None
            if active_r.last_token_id is not None:
                last_token_id = active_r.last_token_id.item()

            hit_eos = self.tokenizer.eos_token_id is not None and last_token_id == self.tokenizer.eos_token_id
            hit_max_tokens = active_r.generated_tokens >= r.max_new_tokens
            print("last_token_id:", last_token_id, "eos_token_id:", self.tokenizer.eos_token_id)

            if not hit_eos and not hit_max_tokens:
                still_active.append(active_r)
                continue

            ctx = self.active.get(r_id)
            if ctx is None or ctx.future.done():
                continue

            completion_time = time.time()
            service_start_time = r.service_start_time or completion_time
            self.completed_total += 1

            ctx.token_queue.put_nowait(None)
            ctx.future.set_result(
                InferenceResult(
                    request_id=r.request_id,
                    text=active_r.output_text,
                    finish_reason="eos" if hit_eos else "length",
                    arrival_time=r.arrival_time,
                    service_start_time=service_start_time,
                    completion_time=completion_time,
                    batching_delay=service_start_time - r.arrival_time,
                    processing_delay=completion_time - service_start_time,
                    generated_tokens=active_r.generated_tokens,
                )
            )

            print(f"[FINISH] request {r_id} finished")

        self.active_slots = still_active

    def get_stats(self):
        return {
            "submitted_total": self.submitted_total,
            "completed_total": self.completed_total,
            "rejected_total": self.rejected_total,
            "decode_ticks_total": self.decode_ticks_total,
            "pending_queue": self.pending.qsize(),
            "active_requests": len(self.active_slots),
        }

    def _remove_cancelled_requests(self):
        still_active = []

        for active_r in self.active_slots:
            r_id = active_r.request.request_id
            ctx = self.active.get(r_id)

            if ctx is not None and ctx.cancelled:
                if not ctx.future.done():
                    ctx.future.set_exception(asyncio.CancelledError())

                ctx.token_queue.put_nowait(None)
                print(f"[CANCEL] request {r_id} cancelled")
                self.active.pop(r_id, None)
                continue

            still_active.append(active_r)

        self.active_slots = still_active
