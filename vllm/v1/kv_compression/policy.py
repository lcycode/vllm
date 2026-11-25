# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU policy interface. One instance follows one exact token history."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PagedQuery:
    """Read-only view valid during observe(). Q and K already include RoPE."""

    query: torch.Tensor  # [query heads, head dim]
    cache: torch.Tensor  # [2, kernel pages, page size, KV heads, head dim]
    block_table: torch.Tensor  # kernel page ids covering this request
    length: int  # visible resident tokens, including the query itself
    logsumexp: torch.Tensor  # [query heads], from the same attention forward
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

    def end_query(self) -> None:
        """Finish an observed decode token after all layers have run."""

    def record_logits(self, logits: torch.Tensor) -> None:
        pass

    def finish_turn(self, length: int, protected: int) -> torch.Tensor | None:
        """Sorted resident indices [full layers, retained tokens], or None."""
        return None

    def apply_keep(self, keep: torch.Tensor) -> None:
        """Remap policy state to the same resident indices as the engine."""

    def stats(self) -> dict[str, Any]:
        return {}
