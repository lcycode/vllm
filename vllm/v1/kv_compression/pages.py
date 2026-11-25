# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def kernel_pages(page_ids: list[int], allocation_size: int, cache: torch.Tensor):
    """Translate allocator pages to the kernel's (possibly smaller) pages."""
    size = cache.shape[2]
    if allocation_size % size:
        raise ValueError("allocation pages must contain whole kernel pages")
    ratio = allocation_size // size
    pages = torch.tensor(page_ids, dtype=torch.long, device=cache.device)
    if ratio == 1:
        return pages
    return (pages[:, None] * ratio + torch.arange(ratio, device=cache.device)).ravel()


def validate_keep(keep, layers, length, protected, device):
    if (
        not isinstance(keep, torch.Tensor)
        or keep.dtype != torch.long
        or keep.device != device
        or keep.ndim != 2
        or keep.shape[0] != layers
        or not protected <= keep.shape[1] <= length
    ):
        raise ValueError("expected an int64 keep matrix on the cache device")
    if keep.shape[1] == 0:
        return
    valid = (
        (keep[:, 0] >= 0).all()
        & (keep[:, -1] < length).all()
        & (keep[:, 1:] > keep[:, :-1]).all()
        & (keep[:, :protected] == torch.arange(protected, device=device)).all()
    )
    if not valid:
        raise ValueError(
            "keep indices must be ordered, in bounds and preserve the prefix"
        )


def compact_layer(
    cache: torch.Tensor, pages: torch.Tensor, keep: torch.Tensor, protected: int = 0
):
    """Pack K and V into leading slots of a strided [2,B,S,H,D] cache.

    Sorted unique survivors satisfy keep[i] >= i. Gathering each batch before
    its write handles overlap within the batch and bounds temporary storage.
    """
    size = cache.shape[2]
    for first in range(protected, len(keep), 128):
        src = keep[first : first + 128]
        dst = torch.arange(first, first + len(src), device=src.device)
        cache[:, pages[dst // size], dst % size] = cache[
            :, pages[src // size], src % size
        ]
