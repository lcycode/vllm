# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side observations, physical addressing and synchronous page compaction."""

import logging
from dataclasses import dataclass
from functools import partial

import torch

from .pages import compact_layer, kernel_pages, validate_keep
from .policy import KVCompressionPolicy, PagedQuery

logger = logging.getLogger(__name__)


@dataclass
class Session:
    policy: KVCompressionPolicy
    protected: int
    decode_from: int
    evicted: int = 0


class CompressionRunner:
    def __init__(self, runner, config):
        from vllm.tokenizers import cached_tokenizer_from_config
        from vllm.utils.import_utils import resolve_obj_by_qualname
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        self.runner = runner
        self.policy_cls = resolve_obj_by_qualname(config["policy"])
        if not issubclass(self.policy_cls, KVCompressionPolicy):
            raise ValueError("policy must implement KVCompressionPolicy")
        self.options = config.get("options", {})
        self.tokenizer = cached_tokenizer_from_config(runner.model_config)
        self.full_groups: dict[int, int] = {}
        self.layers = []
        self.active: dict[str, Session] = {}
        self.parked: dict[int, Session] = {}
        self.query_rows = []
        self.logit_rows = []
        self.collecting = False
        self.offsets = None
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "none":
                continue
            if (
                type(spec) is not FullAttentionSpec
                or spec.sliding_window is not None
                or spec.attention_chunk_size is not None
            ):
                raise ValueError(f"unsupported compression cache spec: {spec}")
            self.full_groups[gid] = spec.block_size
            for name in group.layer_names:
                layer = runner.compilation_config.static_forward_context[name]
                impl = layer.impl
                if not (
                    isinstance(impl, FlashAttentionImpl)
                    and impl.attn_type == AttentionType.DECODER
                    and impl.sliding_window == (-1, -1)
                    and impl.alibi_slopes is None
                    and not impl.logits_soft_cap
                    and impl.sinks is None
                    and impl.kv_sharing_target_layer_name is None
                    and layer.kv_cache.ndim == 5
                    and layer.kv_cache.dtype in (torch.float16, torch.bfloat16)
                ):
                    raise ValueError(
                        f"{name} requires unshared fp16/bf16 full FlashAttention"
                    )
                self.layers.append((gid, layer))
        if not self.layers:
            raise ValueError(
                "KV compression requires at least one full attention layer"
            )
        # Check server policy options before accepting any requests.
        self.new_policy()

    def new_policy(self):
        return self.policy_cls(
            tokenizer=self.tokenizer,
            num_layers=len(self.layers),
            device=self.runner.device,
            **self.options,
        )

    def prepare_slots(self, num_reqs, starts, positions, req_indices, scheduled):
        """positions stays logical for RoPE, input IDs, and recurrent states."""
        batch = self.runner.input_batch
        self.query_rows.clear()
        self.logit_rows.clear()
        offsets = []
        cursor = 0
        for row, req_id in enumerate(batch.req_ids[:num_reqs]):
            n = int(scheduled[row])
            state = self.active.get(req_id)
            offsets.append(state.evicted if state else 0)
            if state is not None:
                start = int(batch.num_computed_tokens_cpu[row])
                req = self.runner.requests[req_id]
                if start >= max(req.num_prompt_tokens, state.decode_from):
                    assert n == 1  # synchronous decoding without speculation
                    if state.policy.wants_query(int(batch.token_ids_cpu[row, start])):
                        self.query_rows.append(
                            (state, row, cursor, start + 1 - state.evicted)
                        )
                if state.policy.needs_logits and start + n == req.num_tokens:
                    self.logit_rows.append((state, row))
            cursor += n

        collecting = bool(self.query_rows)
        if collecting != self.collecting:
            for index, (_, layer) in enumerate(self.layers):
                layer.impl.kv_query_observer = (
                    partial(self.observe, index) if collecting else None
                )
            self.collecting = collecting

        self.offsets = None
        physical = positions
        if any(offsets):
            self.offsets_cpu = torch.tensor(offsets, dtype=torch.int32)
            self.offsets = self.offsets_cpu.to(positions.device)
            physical = positions - self.offsets[req_indices.long()]
        for gid, table in enumerate(batch.block_table.block_tables):
            table.compute_slot_mapping(
                num_reqs, starts, physical if gid in self.full_groups else positions
            )

    def adjust_metadata(self, gid, cm):
        if gid in self.full_groups and self.offsets is not None:
            # Eager execution has one metadata row per scheduled request.
            assert cm.num_reqs == len(self.offsets_cpu)
            cm.seq_lens = cm.seq_lens - self.offsets
            cm._seq_lens_cpu = cm._seq_lens_cpu - self.offsets_cpu
            cm._num_computed_tokens_cpu = cm._num_computed_tokens_cpu - self.offsets_cpu
            cm._num_computed_tokens_cache = None
            cm.max_seq_len = int(cm._seq_lens_cpu.max())

    def observe(self, index, query, cache, block_table, lse, scale):
        size = cache.shape[2]
        for state, row, token, length in self.query_rows:
            pages = block_table[row, : (length + size - 1) // size]
            state.policy.observe(
                index,
                PagedQuery(query[token], cache, pages, length, lse[:, token], scale),
            )

    def after_forward(self, logits):
        for state, _, _, _ in self.query_rows:
            state.policy.end_query()
        for state, row in self.logit_rows:
            state.policy.record_logits(logits[row])
        self.query_rows.clear()
        self.logit_rows.clear()

    def compact(self, state: Session, length: int, tables: dict[int, list[int]]) -> int:
        keep = state.policy.finish_turn(length, state.protected)
        if keep is None:
            return length
        validate_keep(
            keep,
            len(self.layers),
            length,
            state.protected,
            self.layers[0][1].kv_cache.device,
        )
        page_tables: dict[int, torch.Tensor] = {}
        for indices, (gid, layer) in zip(keep, self.layers):
            cache = layer.kv_cache
            if gid not in page_tables:
                page_tables[gid] = kernel_pages(
                    tables[gid], self.full_groups[gid], cache
                )
            compact_layer(cache, page_tables[gid], indices, state.protected)
        state.policy.apply_keep(keep)
        return keep.shape[1]

    def park(self, req_id, handle, seen, tables):
        state = self.active[req_id]
        length = seen - state.evicted
        result = None
        try:
            retained = self.compact(state, length, tables)
            result = length, retained, state.policy.stats()
        except Exception:
            # A copy can fail after some layers have moved. Retire this session
            # and let the scheduler finish the request through its native free().
            logger.exception(
                "Discarding KV session after failed compression for %s", req_id
            )
        # The scheduler can give the released pages to another request as soon
        # as this RPC returns. All model writes and relocations must be visible.
        torch.cuda.synchronize(self.runner.device)
        del self.active[req_id]
        if result is not None:
            state.evicted = seen - retained
            self.parked[handle] = state
        return result


# Module-level functions are serialized by Executor.collective_rpc. Tensor
# data and policy instances stay on the worker side for their entire lifetime.
def initialize_worker(worker, config):
    if worker.use_v2_model_runner:
        raise ValueError("KV compression requires VLLM_USE_V2_MODEL_RUNNER=0")
    adapter = CompressionRunner(worker.model_runner, config)
    worker.model_runner.kv_compression = adapter
    return adapter.full_groups


@torch.inference_mode()
def start_session(worker, req_id, protected, decode_from, handle=None):
    adapter = worker.model_runner.kv_compression
    if handle is None:
        state = Session(adapter.new_policy(), protected, decode_from)
    else:
        state = adapter.parked.pop(handle)
        state.decode_from = decode_from
    adapter.active[req_id] = state


@torch.inference_mode()
def park_session(worker, req_id, handle, seen, tables):
    return worker.model_runner.kv_compression.park(req_id, handle, seen, tables)


def discard_active(worker, req_id):
    worker.model_runner.kv_compression.active.pop(req_id, None)


def discard_parked(worker, handle):
    del worker.model_runner.kv_compression.parked[handle]
