# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate logical token counts for groups whose KV pages were compacted."""

from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


class ResidentFullAttentionManager(FullAttentionManager):
    def __init__(self, native: FullAttentionManager, evicted: dict[str, int]):
        # Installed once during engine initialization, before any page is owned.
        super().__init__(
            native.kv_cache_spec,
            native.block_pool,
            native.enable_caching,
            native.kv_cache_group_id,
            native.dcp_world_size,
            native.pcp_world_size,
        )
        self.num_evicted_tokens = evicted

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        offset = self.num_evicted_tokens.get(request_id, 0)
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens - offset,
            new_computed_blocks,
            total_computed_tokens - offset,
            num_tokens_main_model - offset,
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        offset = self.num_evicted_tokens.get(request_id, 0)
        return super().allocate_new_blocks(
            request_id, num_tokens - offset, num_tokens_main_model - offset
        )
