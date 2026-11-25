# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real V1 allocator and worker lifecycle, with CPU tensors in place of CUDA."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_compression import get_compression_scheduler, worker
from vllm.v1.kv_compression.middleware import (
    HTTP_SESSION_ARG,
    SessionMiddleware,
    attach_http_session,
)
from vllm.v1.kv_compression.scheduler import CompressionScheduler, SessionCacheManager
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .kv_compression_utils import bind_cpu_worker


def test_middleware_resets_cancelled_sessions_and_keeps_other_sampling_args():
    enabled = SimpleNamespace(additional_config={"kv_compression": {}})
    disabled = SimpleNamespace(additional_config={})
    # skip_clone=True is used by the native HTTP request types.
    params = SamplingParams(skip_clone=True, extra_args={"custom": 1})

    async def cancelled(scope, receive, send):
        clone = params.clone()
        attach_http_session(disabled, clone)
        assert clone.extra_args == {"custom": 1}
        attach_http_session(enabled, clone)
        assert clone.extra_args == {"custom": 1, HTTP_SESSION_ARG: ["codex", "abc"]}
        assert params.extra_args == {"custom": 1}
        raise asyncio.CancelledError

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await SessionMiddleware(cancelled)(
                {"type": "http", "headers": [(b"thread-id", b"abc")]}, None, None
            )
        # The same params can carry an offline session only outside HTTP.
        clone = params.clone()
        clone.extra_args = {
            **params.extra_args,
            "kv_session": "offline",
            HTTP_SESSION_ARG: ["codex", "abc"],
        }
        attach_http_session(enabled, clone)
        assert clone.extra_args == {"custom": 1, "kv_session": "offline"}

        async def anonymous(scope, receive, send):
            attach_http_session(enabled, clone)
            assert clone.extra_args == {"custom": 1}

        await SessionMiddleware(anonymous)({"type": "http", "headers": []}, None, None)

    asyncio.run(run())


def test_compression_scheduler_selection(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.v1.core.sched.scheduler import Scheduler

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    assert get_compression_scheduler(config) is None
    config.additional_config = {"kv_compression": {"policy": "example.KeepLast"}}
    config.scheduler_config.async_scheduling = False
    config.cache_config.enable_prefix_caching = False
    config.model_config = SimpleNamespace(
        enforce_eager=True,
        is_encoder_decoder=False,
        enable_return_routed_experts=False,
        runner_type="generate",
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert get_compression_scheduler(config) is CompressionScheduler
    config.scheduler_config.scheduler_cls = CompressionScheduler
    assert get_compression_scheduler(config) is CompressionScheduler
    config.scheduler_config.scheduler_cls = Scheduler
    with pytest.raises(ValueError, match="requires CompressionScheduler"):
        get_compression_scheduler(config)


def test_worker_binds_full_attention_and_detects_local_mask_specs():
    from dataclasses import replace

    from vllm.v1.attention.backends.flash_attn import AttentionType, FlashAttentionImpl

    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl.attn_type, impl.sliding_window = AttentionType.DECODER, (-1, -1)
    impl.alibi_slopes = impl.sinks = impl.kv_sharing_target_layer_name = None
    impl.logits_soft_cap = 0
    layer = SimpleNamespace(
        impl=impl, kv_cache=torch.zeros(2, 4, 16, 2, 8, dtype=torch.float16)
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    full = KVCacheGroupSpec(["attn"], spec)
    ssm = KVCacheGroupSpec(
        ["ssm"], MambaSpec(block_size=64, shapes=((1,),), dtypes=(torch.float32,))
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(skip_tokenizer_init=True),
        kv_cache_config=KVCacheConfig(4, [], [ssm, full]),
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer}),
        device=torch.device("cpu"),
    )
    stub = SimpleNamespace(use_v2_model_runner=False, model_runner=runner)
    config = {"policy": "vllm.v1.kv_compression.policy.KVCompressionPolicy"}
    assert worker.initialize_worker(stub, config) == {1: 16}
    assert runner.kv_compression.layers == [(1, layer)]
    # A disabled hybrid allocator can represent a local mask as a FullAttentionSpec.
    for local in (
        replace(spec, sliding_window=16),
        replace(spec, attention_chunk_size=16),
    ):
        runner.kv_cache_config = KVCacheConfig(
            4, [], [KVCacheGroupSpec(["attn"], local)]
        )
        with pytest.raises(ValueError, match="unsupported compression cache spec"):
            worker.initialize_worker(stub, config)


def request(req_id, ids, key="agent", salt=None):
    sp = SamplingParams(max_tokens=32, extra_args={"kv_session": key})
    return Request(req_id, ids, sp, None, cache_salt=salt)


def schedule_new(cache, *requests):
    new = [SimpleNamespace(req_id=r.request_id) for r in requests]
    cache.after_schedule(
        SimpleNamespace(scheduled_new_reqs=new), {r.request_id: r for r in requests}
    )


@pytest.fixture
def engine(monkeypatch):
    # Seven usable pages across both groups (block 0 is the null block).
    full = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    recurrent = MambaSpec(block_size=64, shapes=((1,),), dtypes=(torch.float32,))
    groups = [KVCacheGroupSpec(["attn"], full), KVCacheGroupSpec(["ssm"], recurrent)]
    manager = SessionCacheManager(
        KVCacheConfig(8, [], groups), 64, 2, enable_caching=False
    )
    recurrent_manager = manager.coordinator.single_type_managers[1]
    adapter, kv = bind_cpu_worker(manager, monkeypatch)
    assert manager.managers[1] is recurrent_manager
    assert manager.managers[0].block_pool is recurrent_manager.block_pool
    a = request("A", [1, 2])
    manager.allocate_slots(a, 10)
    schedule_new(manager, a)
    a.append_output_token_ids(list(range(3, 12)))
    a.num_computed_tokens = 10  # token 11 sampled, no KV yet
    old_pages, recurrent_pages = manager.get_blocks("A").get_block_ids()
    for pos in range(10):
        kv[:, old_pages[pos // 2], pos % 2] = float(pos + 1)
    a.status = RequestStatus.FINISHED_STOPPED
    manager.finish(a)
    manager.free(a)
    return manager, adapter, kv, old_pages, recurrent_pages


def test_another_request_uses_returned_pages_then_only_suffix_is_allocated(engine):
    manager, adapter, kv, original, recurrent_pages = engine
    assert manager.block_pool.get_num_free_blocks() == 4
    b = request("B", [10] * 5, key=None)
    manager.allocate_slots(b, 5)
    b_attn, b_ssm = manager.get_blocks("B").get_block_ids()
    assert set(original[2:]) <= set(b_attn + b_ssm)
    kv[:, b_attn + b_ssm] = -1
    manager.free(b)
    a = request("A2", list(range(1, 14)))
    hits, seen = manager.get_computed_blocks(a)
    assert seen == 10 and hits is manager.empty_kv_cache_blocks
    assert manager.can_fit_full_sequence(a, seen, hits)
    manager.allocate_slots(a, a.num_tokens - seen, seen, hits)
    schedule_new(manager, a)
    new, ssm = manager.get_blocks("A2").get_block_ids()
    assert new[:2] == original[:2] and ssm == recurrent_pages
    assert len(new) == 3  # 3 survivors + 3 suffix tokens, not 13 logical tokens
    assert kv[0, new[0], :, 0, 0].tolist() == [1, 2]
    assert kv[0, new[1], 0, 0, 0] == 10
    assert adapter.active["A2"].evicted == 7
    # A native preemption rebuilds the logical prefix with fresh policy state.
    manager.reset_active(a)
    manager.free(a)
    assert adapter.active["A2"].evicted == 0
    assert adapter.active["A2"].protected == 2
    assert not manager.managers[0].num_evicted_tokens
    a.status = RequestStatus.FINISHED_ABORTED
    manager.finish(a)
    assert not manager.idle and not adapter.active and not adapter.parked


def test_prefix_misses_failed_admission_and_pressure_release(engine):
    manager, adapter, _, _, _ = engine
    ids = list(range(1, 13))
    for r in (
        request("edit", [99, *ids[1:]]),
        request("short", ids[:10]),
        request("salt", ids, salt="other"),
        request("branch", ids, key="other"),
    ):
        assert manager.get_computed_blocks(r)[1] == 0
    waiting = request("waiting", ids)
    assert manager.get_computed_blocks(waiting)[1] == 10
    # A second branch cannot claim the same physical pages concurrently.
    assert manager.get_computed_blocks(request("fork", ids))[1] == 0
    schedule_new(manager)  # no token budget / insufficient capacity
    assert not manager.claimed and not manager.managers[0].num_evicted_tokens
    assert len(manager.idle) == len(adapter.parked) == 1
    big = request("big", [7] * 12, key=None)
    assert manager.can_fit_full_sequence(big)
    assert not manager.idle and not adapter.parked
    assert manager.allocate_slots(big, 12) is not None
    assert manager.block_pool.get_num_free_blocks() == 0
    assert manager.get_computed_blocks(waiting)[1] == 0


@pytest.mark.parametrize("sync_fails", [False, True])
def test_failed_copy_discards_snapshot_only_after_device_ack(
    engine, monkeypatch, sync_fails
):
    manager, adapter, kv, _, _ = engine
    r = request("A3", list(range(1, 14)))
    hit, seen = manager.get_computed_blocks(r)
    manager.allocate_slots(r, 3, seen, hit)
    schedule_new(manager, r)
    r.num_computed_tokens = 13
    r.status = RequestStatus.FINISHED_STOPPED
    owned = manager.get_blocks("A3").get_block_ids()
    free = manager.block_pool.get_num_free_blocks()

    def fail(*args):
        kv[:, owned[0][0]] = -2  # Simulate a partially written layer.
        raise RuntimeError("copy failed")

    def synchronize(_):
        assert manager.block_pool.get_num_free_blocks() == free
        if sync_fails:
            raise RuntimeError("device synchronization failed")

    monkeypatch.setattr(worker, "compact_layer", fail)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    if sync_fails:
        with pytest.raises(RuntimeError, match="device synchronization failed"):
            manager.finish(r)
        assert "A3" in manager.active and "A3" in adapter.active
        assert manager.get_blocks("A3").get_block_ids() == owned
        assert manager.block_pool.get_num_free_blocks() == free
    else:
        manager.finish(r)
        manager.free(r)
        assert not manager.active and not adapter.active
        assert manager.block_pool.get_num_free_blocks() == 7
        assert manager.get_computed_blocks(request("retry", list(range(1, 16))))[1] == 0
    assert not manager.idle and not adapter.parked


def test_native_schedule_to_stop_to_chunked_continuation(engine):
    fixture_manager, adapter, _, _, _ = engine
    fixture_manager.evict_oldest()
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.model_config = MagicMock(
        is_multimodal_model=False,
        is_encoder_decoder=False,
        max_model_len=64,
        enable_return_routed_experts=False,
        skip_tokenizer_init=True,
    )
    config.cache_config.enable_prefix_caching = False
    config.cache_config.num_gpu_blocks = 8
    config.scheduler_config.long_prefill_token_threshold = 2
    scheduler = CompressionScheduler(
        config,
        fixture_manager.kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=2,
        log_stats=False,
    )
    cache = scheduler.kv_cache_manager
    cache.bind(
        fixture_manager.rpc, {"policy": "example.KeepLast", "protected_tokens": 2}
    )
    a = request("first", [1, 2, 3, 4, 5])
    a.max_tokens = 3
    scheduler.add_request(a)

    def step(req_id, expected_count, sampled):
        batch = scheduler.schedule()
        assert batch.num_scheduled_tokens == {req_id: expected_count}
        scheduler.update_from_output(
            batch,
            ModelRunnerOutput(
                req_ids=[req_id],
                req_id_to_index={req_id: 0},
                sampled_token_ids=[sampled],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        return batch

    step("first", 2, [])
    step("first", 2, [])
    step("first", 1, [6])
    step("first", 1, [7])
    step("first", 1, [8])  # 7 computed, last sample still has no KV
    assert a.request_id not in scheduler.requests
    assert cache.idle[None, "agent"].retained == 3
    b = request("second", list(range(1, 11)))
    scheduler.add_request(b)
    first_suffix = step("second", 2, [])
    start = first_suffix.scheduled_new_reqs[0].num_computed_tokens
    assert start == 7
    assert b.num_cached_tokens == 7
    full, recurrent = MagicMock(), MagicMock()
    adapter.runner.input_batch = SimpleNamespace(
        req_ids=["second"],
        num_computed_tokens_cpu=[start],
        block_table=SimpleNamespace(block_tables=[full, recurrent]),
    )
    adapter.runner.requests = {"second": b}
    positions = torch.arange(start, start + first_suffix.num_scheduled_tokens["second"])
    adapter.prepare_slots(1, torch.tensor([0, 2]), positions, torch.tensor([0, 0]), [2])
    assert positions.tolist() == [7, 8]
    assert full.compute_slot_mapping.call_args.args[2].tolist() == [3, 4]
    assert recurrent.compute_slot_mapping.call_args.args[2] is positions
    step("second", 1, [11])
    scheduler.finish_requests("second", RequestStatus.FINISHED_ABORTED)
    assert cache.block_pool.get_num_free_blocks() == 7 and not cache.active


def test_flash_forward_observer_selects_correct_request_query_and_lse(monkeypatch):
    from functools import partial

    from vllm.v1.attention.backends import flash_attn as fa

    # Replace only the CUDA call. Exercise the real FlashAttention forward and
    # runner-side slicing with a two-token prefill ahead of an action query.
    impl = fa.FlashAttentionImpl.__new__(fa.FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    impl.attn_type = fa.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    impl.num_kv_heads = impl.dcp_world_size = 1
    impl.scale, impl.sliding_window = 0.5, (-1, -1)
    impl.logits_soft_cap, impl.alibi_slopes, impl.sinks = 0.0, None, None
    meta = SimpleNamespace(
        num_actual_tokens=3,
        use_cascade=False,
        causal=True,
        query_start_loc=torch.tensor([0, 2, 3]),
        max_query_len=2,
        seq_lens=torch.tensor([2, 3]),
        max_seq_len=3,
        block_table=torch.tensor([[1, 0], [3, 2]]),
        scheduler_metadata=None,
        max_num_splits=0,
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    query = torch.arange(12).reshape(3, 2, 2).float()
    kv = torch.zeros(2, 4, 2, 1, 2)
    output = torch.empty_like(query)
    lse = torch.arange(6).reshape(2, 3).float()
    flags = []

    def flash(**kw):
        flags.append("return_softmax_lse" in kw)
        return (output, lse) if kw.get("return_softmax_lse") else output

    monkeypatch.setattr(fa, "flash_attn_varlen_func", flash, raising=False)
    adapter = worker.CompressionRunner.__new__(worker.CompressionRunner)
    policy = MagicMock()
    adapter.query_rows = [(SimpleNamespace(policy=policy), 1, 2, 3)]
    for hook in (None, partial(adapter.observe, 0)):
        impl.kv_query_observer = hook
        assert impl.forward(layer, query, None, None, kv, meta, output) is output
    assert flags == [False, True]
    index, observed = policy.observe.call_args.args
    assert index == 0 and observed.cache is kv and observed.length == 3
    assert observed.block_table.tolist() == [3, 2]
    torch.testing.assert_close(observed.query, query[2])
    torch.testing.assert_close(observed.logsumexp, lse[:, 2])
