# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Carry a harness's native session identity to the engine request."""

from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

from . import compression_config

HTTP_SESSION_ARG = "_kv_compression_http_session"
PROTECTED_TOKENS_ARG = "kv_protected_tokens"
_session = ContextVar[tuple[str, ...] | None]("kv_compression_session", default=None)


def _session_from_headers(headers: Mapping[str, str]) -> tuple[str, ...]:
    if session := headers.get("x-opencode-session-id", "").strip():
        return ("opencode", session)
    if session := headers.get("x-claude-code-session-id", "").strip():
        if agent := headers.get("x-claude-code-agent-id", "").strip():
            return ("claude-code", session, agent)
        return ("claude-code", session)
    if thread := headers.get("thread-id", "").strip():
        return ("codex", thread)
    return ()  # Anonymous HTTP request. None is reserved for offline callers.


class SessionMiddleware:
    """ASGI middleware for vllm serve --middleware. Reads only HTTP headers."""

    def __init__(self, app: "ASGIApp"):
        self.app = app

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        from starlette.datastructures import Headers

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = _session.set(_session_from_headers(Headers(scope=scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _session.reset(token)


def attach_http_session(config: "VllmConfig", params: "SamplingParams") -> None:
    session = _session.get()
    if (not session and not params.extra_args) or compression_config(config) is None:
        return
    # Native HTTP requests shallow-clone SamplingParams. Only headers may
    # identify an HTTP session, even if vllm_xargs contains an offline key.
    extra = (params.extra_args or {}).copy()
    if PROTECTED_TOKENS_ARG in extra:
        count = extra[PROTECTED_TOKENS_ARG]
        if type(count) is not int or count < 0:
            raise ValueError(f"{PROTECTED_TOKENS_ARG} must be a nonnegative integer")
    if session is not None:
        extra.pop("kv_session", None)
    if session:
        extra[HTTP_SESSION_ARG] = list(session)
    else:
        extra.pop(HTTP_SESSION_ARG, None)
    params.extra_args = extra or None
