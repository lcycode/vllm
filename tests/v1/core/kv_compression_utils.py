# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.kv_compression import worker
from vllm.v1.kv_compression.policy import KVCompressionPolicy


class KeepLast(KVCompressionPolicy):
    def finish_turn(self, length, protected):
        keep = list(range(protected))
        if length > protected:
            keep.append(length - 1)
        return torch.tensor([keep], dtype=torch.long)


def bind_cpu_worker(cache, monkeypatch, *, protected_tokens=None):
    """Use the real page operations on a CPU tensor for full-attention group 0."""
    config = cache.kv_cache_config
    block_size = config.kv_cache_groups[0].kv_cache_spec.block_size
    kv = torch.zeros(2, config.num_blocks, block_size, 1, 1)
    adapter = worker.CompressionRunner.__new__(worker.CompressionRunner)
    adapter.active, adapter.parked = {}, {}
    adapter.full_groups = {0: block_size}
    adapter.query_rows, adapter.logit_rows, adapter.collecting = [], [], False
    adapter.new_policy = lambda: KeepLast(tokenizer=None, num_layers=1, device="cpu")
    adapter.layers = [(0, SimpleNamespace(kv_cache=kv))]
    adapter.runner = SimpleNamespace(device=torch.device("cpu"))
    stub = SimpleNamespace(model_runner=SimpleNamespace(kv_compression=adapter))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)

    def rpc(method, args):
        if method is worker.initialize_worker:
            return [adapter.full_groups]
        return [method(stub, *args)]

    cache.bind(
        rpc, {"policy": "example.KeepLast", "protected_tokens": protected_tokens}
    )
    return adapter, kv
