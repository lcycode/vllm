# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU policy interface. One instance follows one exact token history."""

from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import Self


@dataclass(frozen=True)
class PagedQuery:
    """Read-only view valid during observe(). Q and K already include RoPE."""

    query: torch.Tensor  # [query heads, head dim]
    cache: torch.Tensor  # [2, kernel pages, page size, KV heads, head dim]
    block_table: torch.Tensor  # kernel page ids covering this request
    length: int  # visible resident tokens, including the query itself
    logsumexp: torch.Tensor  # [query heads], from the same attention forward
    scale: float


@dataclass(frozen=True)
class PagedQueryBatch:
    """One layer, one action query per request, all fields in policy order."""

    query: torch.Tensor  # [requests, query heads, head dim]
    cache: torch.Tensor  # shared physical pages, as in PagedQuery
    block_table: torch.Tensor  # [requests, max kernel pages used in this batch]
    lengths: tuple[int, ...]  # resident lengths, available without a device sync
    seq_used: torch.Tensor  # the same lengths on device, int32
    logsumexp: torch.Tensor  # [requests, query heads]
    scale: float


class KVCompressionPolicy:
    """FullKV by default. Subclass to collect observations and select survivors.

    Each decode token passes through wants_query() before the forward. If it
    returns True, observe() receives the corresponding query from every full
    attention layer. The engine owns the page tables and all KV writes.
    """

    needs_logits = False

    def __init__(self, *, tokenizer, num_layers: int, device: torch.device, **options):
        if options:
            raise ValueError(f"Unknown compression policy options: {sorted(options)}")

    def wants_query(self, token_id: int) -> bool:
        return False

    def observe(self, layer: int, query: PagedQuery) -> None:
        pass

    @classmethod
    def observe_batch(
        cls, layer: int, policies: list[Self], batch: PagedQueryBatch
    ) -> None:
        """Override for batched scoring. Existing observers get one row each."""
        size = batch.cache.shape[2]
        for i, (policy, length) in enumerate(zip(policies, batch.lengths)):
            pages = batch.block_table[i, : (length + size - 1) // size]
            policy.observe(
                layer,
                PagedQuery(
                    batch.query[i],
                    batch.cache,
                    pages,
                    length,
                    batch.logsumexp[i],
                    batch.scale,
                ),
            )

    def end_query(self) -> None:
        """Finish an observed decode token after all layers have run."""

    def check_boundary(
        self, length: int, protected: int, *, unforwarded_token_id: int | None = None
    ) -> None:
        """Check for a boundary after end_query(), or on a final sample without KV.

        length counts resident slots. The final sample has no attention row and
        is excluded from length and from the reusable prefix.
        """

    def record_logits(self, logits: torch.Tensor) -> None:
        pass

    def on_sample(
        self, token_id: int, length: int, protected: int
    ) -> torch.Tensor | None:
        """Observe a newly sampled token before its forward. Optionally select KV
        to compact immediately. length is the number of already resident slots.
        """
        return None

    def finish_turn(self, length: int, protected: int) -> torch.Tensor | None:
        """Sorted resident indices [full layers, retained tokens], or None."""
        return None

    def apply_keep(self, keep: torch.Tensor) -> None:
        """Remap policy state to the same resident indices as the engine."""

    def stats(self) -> dict[str, Any]:
        return {}
