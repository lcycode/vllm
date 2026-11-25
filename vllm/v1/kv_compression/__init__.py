# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-boundary KV compression for the V1 GPU model runner."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .scheduler import CompressionScheduler


def compression_config(config: "VllmConfig") -> dict[str, Any] | None:
    additional = config.additional_config
    options = additional.get("kv_compression") if isinstance(additional, dict) else None
    if options is not None and not isinstance(options, dict):
        raise ValueError("kv_compression must be a configuration object")
    return options


def get_compression_scheduler(
    config: "VllmConfig",
) -> type["CompressionScheduler"] | None:
    options = compression_config(config)
    if options is None:
        return None

    from .scheduler import CompressionScheduler

    sched = config.scheduler_config
    if (
        sched.scheduler_cls is not None
        and sched.get_scheduler_cls() is not CompressionScheduler
    ):
        raise ValueError("kv_compression requires CompressionScheduler")
    CompressionScheduler.validate_config(config, options)
    return CompressionScheduler
