import asyncio
import itertools
import time
import torch
import torch.nn.functional as F

from typing import List, Literal
from transformers.cache_utils import DynamicCache

from .model import load_model_bundle
from .state import (
    ActiveRequest,
    InferenceRequest,
    InferenceResult,
    OverloadedError,
    RequestContext,
)

PrefillMode = Literal["single", "batched"]
DecodeMode = Literal["single", "batched"]


class MiniInferenceEngine:
    def __init__(
        self,
        max_requests,
        batch_size=1,
        model_name=None,
        server_profile: str = "custom",
        prefill_mode: PrefillMode = "batched",
        decode_mode: DecodeMode = "batched",
        tick_log_every: int = 1,
    ):
        self.max_requests = max_requests
        self.batch_size = batch_size
        self.max_wait_time = 0.05
        self._request_id_gen = itertools.count(1)
        self.server_profile = server_profile
        self.prefill_mode = self._validate_prefill_mode(prefill_mode)
        self.decode_mode = self._validate_decode_mode(decode_mode)
        self.tick_log_every = max(1, tick_log_every)
        self.start_time = None

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

    def _validate_prefill_mode(self, mode: str) -> PrefillMode:
        if mode not in {"single", "batched"}:
            raise ValueError(f"invalid prefill_mode: {mode}")
        return mode

    def _validate_decode_mode(self, mode: str) -> DecodeMode:
        if mode not in {"single", "batched"}:
            raise ValueError(f"invalid decode_mode: {mode}")
        return mode

    async def start(self):
        if self.scheduler_task is not None and not self.scheduler_task.done():
            return
        self.start_time = time.time()
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.scheduler_task.add_done_callback(self._handle_scheduler_done)
        self._log(
            "ENGINE",
            status="started",
            model=self.model_name,
            device=self.device.type,
            profile=self.server_profile,
            batch_size=self.batch_size,
            prefill=self.prefill_mode,
            decode=self.decode_mode,
            tick_log_every=self.tick_log_every,
        )

    async def stop(self):
        if self.scheduler_task is not None:
            if not self.scheduler_task.done():
                self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.scheduler_task = None
        self._log("ENGINE", status="stopped")

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
            self._log("QUEUE", status="rejected", request=request.request_id, pending=self.pending.qsize())
            raise OverloadedError("system full, try again later")

        self.active[request.request_id] = ctx
        self.submitted_total += 1
        self._log(
            "QUEUE",
            status="queued",
            request=request.request_id,
            pending=self.pending.qsize(),
            prompt=self._prompt_preview(request.prompt),
        )

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

    def _handle_scheduler_done(self, task: asyncio.Task):
        if task.cancelled():
            return

        exc = task.exception()
        if exc is not None:
            self._log("SCHEDULER", status="crashed", error=repr(exc))

    def _log(self, event: str, **fields):
        timestamp = time.strftime("%H:%M:%S")
        suffix = ""
        if fields:
            suffix = " | " + " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"[{timestamp}] {event}{suffix}")

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
        self._log(
            "ADMIT",
            request=request.request_id,
            slot="first",
            active=len(self.active_slots),
            pending=self.pending.qsize(),
        )

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
            self._log(
                "ADMIT",
                request=request.request_id,
                active=len(self.active_slots),
                pending=self.pending.qsize(),
            )

    def _pick_next_token_id(self, next_token_logits):
        return torch.argmax(next_token_logits, dim=-1)

    def _pick_next_token_id_sampling(self, next_token_logits):
        probs = torch.softmax(next_token_logits / 0.8, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id

    def _build_prompt_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prompt_preview(self, prompt: str, max_len: int = 48) -> str:
        compact = " ".join(prompt.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def _emit_generated_token(self, active_r: ActiveRequest, ctx: RequestContext, token_id: torch.Tensor):
        generated_token = self.tokenizer.decode(token_id[0], skip_special_tokens=True)
        active_r.output_text += generated_token
        active_r.generated_tokens += 1
        ctx.token_queue.put_nowait(generated_token)

    def _normalize_past_key_values(self, past_key_values):
        if past_key_values is None:
            return None

        normalized = []
        for layer in past_key_values:
            if len(layer) < 2:
                raise ValueError(f"unexpected cache layer format: {type(layer)}")
            normalized.append((layer[0], layer[1]))
        return tuple(normalized)

    def _build_model_cache(self, past_key_values):
        if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return DynamicCache(past_key_values, config=self.model.config)

    def _split_past_key_values(self, batched_past_key_values, sequence_lengths):
        per_request = [[] for _ in range(len(sequence_lengths))]

        for layer_k, layer_v in batched_past_key_values:
            for i, seq_len in enumerate(sequence_lengths):
                per_request[i].append(
                    (
                        layer_k[i : i + 1, :, :seq_len, :],
                        layer_v[i : i + 1, :, :seq_len, :],
                    )
                )

        return [tuple(x) for x in per_request]

    def _merge_past_key_values(self, active_requests):
        if not active_requests:
            return (), []

        cache_lengths = [active_r.past_key_values[0][0].shape[2] for active_r in active_requests]
        max_cache_len = max(cache_lengths)
        num_layers = len(active_requests[0].past_key_values)

        merged = []
        for layer_idx in range(num_layers):
            keys = []
            values = []

            for active_r, cache_len in zip(active_requests, cache_lengths):
                layer_k, layer_v = active_r.past_key_values[layer_idx]
                pad_len = max_cache_len - cache_len

                if pad_len > 0:
                    layer_k = F.pad(layer_k, (0, 0, pad_len, 0))
                    layer_v = F.pad(layer_v, (0, 0, pad_len, 0))

                keys.append(layer_k)
                values.append(layer_v)

            merged_k = torch.cat(keys, dim=0)
            merged_v = torch.cat(values, dim=0)
            merged.append((merged_k, merged_v))

        return tuple(merged), cache_lengths

    def _split_decode_past_key_values(self, batched_past_key_values, prior_cache_lengths):
        per_request = [[] for _ in range(len(prior_cache_lengths))]

        for layer_k, layer_v in batched_past_key_values:
            for i, cache_len in enumerate(prior_cache_lengths):
                keep_len = cache_len + 1
                per_request[i].append(
                    (
                        layer_k[i : i + 1, :, -keep_len:, :],
                        layer_v[i : i + 1, :, -keep_len:, :],
                    )
                )

        return [tuple(x) for x in per_request]

    def _tokenize_prompts(self, active_requests: list[ActiveRequest], padding: bool):
        texts = []
        for active_r in active_requests:
            texts.append(self._build_prompt_text(active_r.request.prompt))

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=True,
        )
        return (
            inputs["input_ids"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )

    def _run_prefill_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def _apply_prefill_result(
        self,
        active_r: ActiveRequest,
        ctx: RequestContext,
        next_token_id: torch.Tensor,
        past_key_values,
    ):
        active_r.past_key_values = past_key_values
        active_r.last_token_id = next_token_id
        active_r.prefill_done = True
        self._emit_generated_token(active_r, ctx, next_token_id)

    async def _run_prefill_single(self, prefill_pairs):
        for active_r, ctx in prefill_pairs:
            input_ids, attention_mask = self._tokenize_prompts([active_r], padding=False)
            outputs = self._run_prefill_forward(input_ids, attention_mask)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self._pick_next_token_id_sampling(next_token_logits)
            past_key_values = self._normalize_past_key_values(outputs.past_key_values)

            self._apply_prefill_result(active_r, ctx, next_token_id, past_key_values)

    async def _run_prefill_batched(self, prefill_pairs):
        if not prefill_pairs:
            return

        active_requests = [active_r for active_r, _ctx in prefill_pairs]
        input_ids, attention_mask = self._tokenize_prompts(active_requests, padding=True)
        outputs = self._run_prefill_forward(input_ids, attention_mask)
        past_key_values = self._normalize_past_key_values(outputs.past_key_values)
        last_positions = attention_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(input_ids.size(0), device=self.device)
        next_token_logits = outputs.logits[batch_indices, last_positions, :]
        next_token_ids = self._pick_next_token_id_sampling(next_token_logits)
        sequence_lengths = attention_mask.sum(dim=1).tolist()
        per_request_pkv = self._split_past_key_values(
            past_key_values,
            sequence_lengths,
        )

        for i, (active_r, ctx) in enumerate(prefill_pairs):
            token_id = next_token_ids[i : i + 1]  # keep shape [1]
            self._apply_prefill_result(active_r, ctx, token_id, per_request_pkv[i])

    async def _run_decode_batched(self, decode_pairs):
        if not decode_pairs:
            return

        active_requests = [active_r for active_r, _ctx in decode_pairs]
        batch_input_ids = torch.stack(
            [active_r.last_token_id for active_r in active_requests],
            dim=0,
        )

        batch_past_key_values, cache_lengths = self._merge_past_key_values(active_requests)
        max_cache_len = max(cache_lengths)

        attention_mask = torch.zeros(
            len(active_requests),
            max_cache_len + 1,
            dtype=torch.long,
            device=self.device,
        )
        for i, cache_len in enumerate(cache_lengths):
            attention_mask[i, max_cache_len - cache_len :] = 1

        position_ids = torch.tensor(
            cache_lengths,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._build_model_cache(batch_past_key_values),
                use_cache=True,
            )

        past_key_values = self._normalize_past_key_values(outputs.past_key_values)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = self._pick_next_token_id_sampling(next_token_logits)
        split_past_key_values = self._split_decode_past_key_values(
            past_key_values,
            cache_lengths,
        )

        for i, (active_r, ctx) in enumerate(decode_pairs):
            token_id = next_token_ids[i : i + 1]
            active_r.last_token_id = token_id
            active_r.past_key_values = split_past_key_values[i]
            self._emit_generated_token(active_r, ctx, token_id)

    async def _run_decode_single(self, decode_pairs):
        for active_r, ctx in decode_pairs:
            decode_input_ids = active_r.last_token_id.unsqueeze(0)  # [1] -> [1,1]

            with torch.no_grad():
                outputs = self.model(
                    input_ids=decode_input_ids,
                    past_key_values=self._build_model_cache(active_r.past_key_values),
                    use_cache=True,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self._pick_next_token_id_sampling(next_token_logits)

            active_r.past_key_values = self._normalize_past_key_values(outputs.past_key_values)
            active_r.last_token_id = next_token_id
            self._emit_generated_token(active_r, ctx, next_token_id)

    def _collect_generation_pairs(self):
        prefill_pairs = []
        decode_pairs = []

        for active_r in self.active_slots:
            ctx = self.active.get(active_r.request.request_id)
            if ctx is None:
                continue

            if not active_r.prefill_done:
                prefill_pairs.append((active_r, ctx))
            else:
                decode_pairs.append((active_r, ctx))

        return prefill_pairs, decode_pairs

    async def _run_prefill_step(self, prefill_pairs):
        if self.prefill_mode == "single":
            await self._run_prefill_single(prefill_pairs)
            return
        await self._run_prefill_batched(prefill_pairs)

    async def _run_decode_step(self, decode_pairs):
        if self.decode_mode == "single":
            await self._run_decode_single(decode_pairs)
            return
        await self._run_decode_batched(decode_pairs)

    async def _run_generation_step(self):
        if not self.active_slots:
            return

        tick_start = time.time()
        self.decode_ticks_total += 1

        for active_r in self.active_slots:
            if active_r.request.service_start_time is None:
                active_r.request.service_start_time = tick_start

        prefill_pairs, decode_pairs = self._collect_generation_pairs()
        should_log_tick = bool(prefill_pairs) or self.decode_ticks_total % self.tick_log_every == 0
        if should_log_tick:
            self._log(
                "TICK",
                tick=self.decode_ticks_total,
                active=len(self.active_slots),
                pending=self.pending.qsize(),
                prefill=len(prefill_pairs),
                decode=len(decode_pairs),
                prefill_mode=self.prefill_mode,
                decode_mode=self.decode_mode,
            )
        await self._run_prefill_step(prefill_pairs)
        await self._run_decode_step(decode_pairs)

        await asyncio.sleep(0)

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

            if not hit_eos and not hit_max_tokens:
                still_active.append(active_r)
                continue

            ctx = self.active.get(r_id)
            if ctx is None or ctx.future.done():
                continue

            completion_time = time.time()
            service_start_time = r.service_start_time or completion_time
            self.completed_total += 1
            finish_reason = "eos" if hit_eos else "length"

            ctx.token_queue.put_nowait(None)
            ctx.future.set_result(
                InferenceResult(
                    request_id=r.request_id,
                    text=active_r.output_text,
                    finish_reason=finish_reason,
                    arrival_time=r.arrival_time,
                    service_start_time=service_start_time,
                    completion_time=completion_time,
                    batching_delay=service_start_time - r.arrival_time,
                    processing_delay=completion_time - service_start_time,
                    generated_tokens=active_r.generated_tokens,
                )
            )

            self._log(
                "FINISH",
                request=r_id,
                reason=finish_reason,
                tokens=active_r.generated_tokens,
                batching_ms=int((service_start_time - r.arrival_time) * 1000),
                processing_ms=int((completion_time - service_start_time) * 1000),
            )

        self.active_slots = still_active

    def get_stats(self):
        active_details = []
        for active_r in self.active_slots:
            req = active_r.request
            active_details.append(
                {
                    "request_id": req.request_id,
                    "state": "decode" if active_r.prefill_done else "prefill",
                    "generated_tokens": active_r.generated_tokens,
                    "max_new_tokens": req.max_new_tokens,
                    "prompt_preview": self._prompt_preview(req.prompt, max_len=64),
                }
            )

        uptime_sec = 0.0
        if self.start_time is not None:
            uptime_sec = max(0.0, time.time() - self.start_time)

        return {
            "server_profile": self.server_profile,
            "model_name": self.model_name,
            "device": self.device.type,
            "batch_size": self.batch_size,
            "queue_capacity": self.max_requests,
            "submitted_total": self.submitted_total,
            "completed_total": self.completed_total,
            "rejected_total": self.rejected_total,
            "decode_ticks_total": self.decode_ticks_total,
            "pending_queue": self.pending.qsize(),
            "active_requests": len(self.active_slots),
            "prefill_mode": self.prefill_mode,
            "decode_mode": self.decode_mode,
            "tick_log_every": self.tick_log_every,
            "uptime_sec": round(uptime_sec, 2),
            "active_request_ids": [item["request_id"] for item in active_details],
            "active_details": active_details,
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
                self._log("CANCEL", request=r_id)
                self.active.pop(r_id, None)
                continue

            still_active.append(active_r)

        self.active_slots = still_active
