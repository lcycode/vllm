# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exclusive, bounded turn snapshots on vLLM's existing block pool."""

import itertools
from collections import OrderedDict
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from . import compression_config, worker
from .allocator import ResidentFullAttentionManager
from .middleware import HTTP_SESSION_ARG, PROTECTED_TOKENS_ARG

logger = init_logger(__name__)
SessionKey = tuple[str | None, str | tuple[str, ...]]


def session_key(req: Request) -> SessionKey | None:
    sp = req.sampling_params
    if (
        sp is None
        or req.resumable
        or req.mm_features
        or req.lora_request
        or req.prompt_token_ids is None
        or req.prompt_embeds is not None
        or sp.prompt_logprobs is not None
        or req.skip_reading_prefix_cache
    ):
        return None
    extra = sp.extra_args or {}
    identity = extra.get(HTTP_SESSION_ARG)
    if (
        isinstance(identity, list)
        and len(identity) >= 2
        and all(isinstance(part, str) and part for part in identity)
    ):
        return (req.cache_salt, tuple(identity))
    key = extra.get("kv_session")
    return (req.cache_salt, key) if isinstance(key, str) and key else None


@dataclass
class Snapshot:
    handle: int  # worker-local policy state
    # Exact forwarded prefix. The final sampled token has no KV yet.
    tokens: list[int]
    protected: int
    retained: int
    blocks: list[list[KVCacheBlock]]


class SessionCacheManager(KVCacheManager):
    """Idle sessions own private pages which can be evicted under pressure."""

    def bind(self, rpc, config):
        self.max_idle = config.get("max_idle_sessions", 16)
        self.protected_tokens = config.get("protected_tokens")
        if type(self.max_idle) is not int or self.max_idle < 1:
            raise ValueError("max_idle_sessions must be a positive integer")
        if self.protected_tokens is not None and (
            type(self.protected_tokens) is not int or self.protected_tokens < 0
        ):
            raise ValueError("protected_tokens must be a nonnegative integer")
        self.rpc = rpc
        (self.full_groups,) = rpc(worker.initialize_worker, args=(config,))
        self.num_evicted_tokens: dict[str, int] = {}
        managers = list(self.coordinator.single_type_managers)
        for gid in self.full_groups:
            native = managers[gid]
            assert type(native) is FullAttentionManager
            managers[gid] = ResidentFullAttentionManager(
                native, self.num_evicted_tokens
            )
        self.coordinator.single_type_managers = self.managers = tuple(managers)
        self.idle: OrderedDict[SessionKey, Snapshot] = OrderedDict()
        self.claimed: dict[str, tuple[SessionKey, Snapshot]] = {}
        self.active: dict[str, tuple[SessionKey, int]] = {}
        self.handles = itertools.count()

    def discard(self, snapshot):
        for blocks in snapshot.blocks:
            self.block_pool.free_blocks(reversed(blocks))
        self.rpc(worker.discard_parked, args=(snapshot.handle,))

    def evict_oldest(self):
        if not self.idle:
            return False
        _, snapshot = self.idle.popitem(last=False)
        self.discard(snapshot)
        return True

    def protected_length(self, request: Request, default: int) -> int:
        assert request.sampling_params is not None
        extra = request.sampling_params.extra_args or {}
        count = min(request.num_prompt_tokens, extra.get(PROTECTED_TOKENS_ARG, default))
        if self.protected_tokens is not None:
            count = min(count, self.protected_tokens)
        return count

    def get_computed_blocks(self, request):
        key = session_key(request)
        snap = self.idle.get(key)
        if snap is None or request.num_preemptions:
            return super().get_computed_blocks(request)
        seen = len(snap.tokens)
        ids = request.prompt_token_ids
        # A new request needs at least one suffix token to produce fresh logits.
        if (
            ids is None
            or len(ids) <= seen
            or self.protected_length(request, snap.protected) != snap.protected
            or ids[:seen] != snap.tokens
        ):
            logger.info("KV session prefix miss for request %s", request.request_id)
            return super().get_computed_blocks(request)

        req_id = request.request_id
        del self.idle[key]
        for gid, manager in enumerate(self.managers):
            manager.req_to_blocks[req_id] = snap.blocks[gid]
        self.num_evicted_tokens[req_id] = seen - snap.retained
        self.claimed[req_id] = (key, snap)
        # These pages are already owned. Only the logical hit count is new.
        return self.empty_kv_cache_blocks, seen

    def can_fit_full_sequence(self, *args, **kwargs):
        while not super().can_fit_full_sequence(*args, **kwargs):
            if not self.evict_oldest():
                return False
        return True

    def allocate_slots(self, *args, **kwargs):
        while (blocks := super().allocate_slots(*args, **kwargs)) is None:
            if not self.evict_oldest():
                break
        return blocks

    def after_schedule(self, output, requests):
        for new in output.scheduled_new_reqs:
            req = requests[new.req_id]
            key = session_key(req)
            if key is None:
                continue
            claimed = self.claimed.pop(new.req_id, None)
            protected = req.num_prompt_tokens
            previous = self.idle.get(key)
            if (
                previous
                and req.prompt_token_ids[: previous.protected]
                == previous.tokens[: previous.protected]
            ):
                # A template rewrite can invalidate generated KV while leaving
                # the original system and task boundary intact.
                protected = previous.protected
            protected = self.protected_length(req, protected)
            handle = None
            if claimed is not None:
                _, snap = claimed
                protected, handle = snap.protected, snap.handle
                logger.info(
                    "KV session resumed %s: logical=%d resident=%d suffix=%d",
                    new.req_id,
                    len(snap.tokens),
                    snap.retained,
                    req.num_prompt_tokens - len(snap.tokens),
                )
            self.rpc(
                worker.start_session,
                args=(new.req_id, protected, req.num_prompt_tokens, handle),
            )
            self.active[new.req_id] = (key, protected)
        # Admission can fail after the prefix lookup, e.g. due to token or
        # page limits. Return its exclusive claim for retry or pressure eviction.
        for req_id, (key, snap) in self.claimed.items():
            for manager in self.managers:
                manager.req_to_blocks.pop(req_id)
            self.num_evicted_tokens.pop(req_id)
            self.idle[key] = snap
        self.claimed.clear()

    def compression_tables(self, req_id, seen):
        tables = {}
        resident = seen - self.num_evicted_tokens.get(req_id, 0)
        for gid, size in self.full_groups.items():
            group = self.managers[gid].req_to_blocks[req_id]
            if resident > len(group) * size or any(
                b.is_null or b.ref_cnt != 1 for b in group
            ):
                raise RuntimeError(
                    "compression requires an exclusive, complete page table"
                )
            tables[gid] = [b.block_id for b in group]
        return tables

    def release_tail(self, req_id, retained):
        freed = 0
        for gid, size in self.full_groups.items():
            used = (retained + size - 1) // size
            group = self.managers[gid].req_to_blocks[req_id]
            tail = group[used:]
            del group[used:]
            self.block_pool.free_blocks(reversed(tail))
            freed += len(tail)
        return freed

    def compact_active(self, req_ids, requests):
        work = {}
        for req_id in req_ids:
            if req_id in self.active and req_id in requests:
                seen = requests[req_id].num_computed_tokens
                work[req_id] = (seen, self.compression_tables(req_id, seen))
        if not work:
            return []
        (results,) = self.rpc(worker.compact_active, args=(work,))
        failed = []
        for req_id, (seen, _) in work.items():
            retained = results[req_id]
            if retained is None:
                failed.append(req_id)
                continue
            self.release_tail(req_id, retained)
            self.num_evicted_tokens[req_id] = seen - retained
        return failed

    def finish(self, request):
        req_id = request.request_id
        active = self.active.get(req_id)
        if active is None:
            return
        if (
            request.num_computed_tokens < request.num_prompt_tokens
            or request.status
            not in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
        ):
            self.rpc(worker.discard_active, args=(req_id,))
            del self.active[req_id]
            return
        key, protected = active
        seen = request.num_computed_tokens
        blocks = [manager.req_to_blocks[req_id] for manager in self.managers]
        tables = self.compression_tables(req_id, seen)
        handle = next(self.handles)
        last_sample = (
            request.all_token_ids[seen] if request.num_tokens == seen + 1 else None
        )
        (result,) = self.rpc(
            worker.park_session, args=(req_id, handle, seen, tables, last_sample)
        )
        if result is None:
            del self.active[req_id]
            return
        before, retained, stats = result
        freed = self.release_tail(req_id, retained)
        # Transfer ownership out of the completed request before native free().
        # Recurrent pages follow the session intact, including their logical layout.
        for manager in self.managers:
            del manager.req_to_blocks[req_id]
        if key in self.idle:
            self.discard(self.idle.pop(key))
        while len(self.idle) >= self.max_idle:
            self.evict_oldest()
        self.idle[key] = Snapshot(
            handle, request.all_token_ids[:seen], protected, retained, blocks
        )
        del self.active[req_id]
        logger.info(
            "KV session parked %s: logical=%d resident=%d->%d freed_pages=%d %s",
            req_id,
            seen,
            before,
            retained,
            freed,
            stats,
        )

    def reset_active(self, request):
        if active := self.active.get(request.request_id):
            # Native preemption replays the full logical transcript. Discard
            # the old score coordinates and retain the original prompt boundary.
            self.rpc(
                worker.start_session,
                args=(request.request_id, active[1], request.num_tokens),
            )
            logger.info("KV session preemption rebuild for %s", request.request_id)

    def free(self, request: Request) -> None:
        super().free(request)
        self.num_evicted_tokens.pop(request.request_id, None)


class CompressionScheduler(Scheduler):
    kv_cache_manager_cls = SessionCacheManager
    kv_cache_manager: SessionCacheManager

    def bind_worker_rpc(self, rpc):
        config = compression_config(self.vllm_config)
        if config is None:
            raise ValueError("CompressionScheduler requires a kv_compression config")
        self.kv_cache_manager.bind(rpc, config)

    @staticmethod
    def validate_config(config, compression):
        from vllm import envs
        from vllm.platforms import current_platform

        if not isinstance(compression.get("policy"), str):
            raise ValueError("kv_compression requires a policy class path")
        unknown = set(compression) - {
            "policy",
            "options",
            "max_idle_sessions",
            "protected_tokens",
        }
        if unknown:
            raise ValueError(f"unknown kv_compression options: {sorted(unknown)}")
        parallel = config.parallel_config
        if (
            not current_platform.is_cuda()
            or envs.VLLM_USE_V2_MODEL_RUNNER
            or parallel.world_size != 1
            or parallel.data_parallel_size != 1
            or parallel.use_ubatching
            or config.scheduler_config.async_scheduling
            or config.cache_config.enable_prefix_caching
            or not config.model_config.enforce_eager
            or config.speculative_config
            or config.kv_transfer_config
            or config.ec_transfer_config
            or config.model_config.is_encoder_decoder
            or config.model_config.enable_return_routed_experts
            or config.model_config.runner_type != "generate"
        ):
            raise ValueError(
                "kv_compression requires V1 single-GPU CUDA generation with "
                "--enforce-eager --no-async-scheduling --no-enable-prefix-caching "
                "and CompressionScheduler. Speculation, microbatching, routed expert "
                "capture and cache transfer are unsupported."
            )

    def schedule(self):
        output = super().schedule()
        self.kv_cache_manager.after_schedule(output, self.requests)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        failures = []
        if req_ids := model_runner_output.kv_compression_req_ids:
            # Process the boundary before native stop handling can park a snapshot.
            failed = self.kv_cache_manager.compact_active(req_ids, self.requests)
            if failed:
                failures = self.finish_requests(failed, RequestStatus.FINISHED_ERROR)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for req_id, client in failures:
            if client not in outputs:
                outputs[client] = EngineCoreOutputs()
            outputs[client].outputs.append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                )
            )
        return outputs

    def _free_blocks(self, request):
        self.kv_cache_manager.finish(request)
        super()._free_blocks(request)

    def _preempt_request(self, request, timestamp):
        self.kv_cache_manager.reset_active(request)
        super()._preempt_request(request, timestamp)

    def reset_prefix_cache(self, *args, **kwargs):
        while self.kv_cache_manager.evict_oldest():
            pass
        return super().reset_prefix_cache(*args, **kwargs)
