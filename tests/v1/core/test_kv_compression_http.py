# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay native HTTP tool turns with real rendering, scheduling and CPU pages."""

import asyncio
import json

import httpx
import pytest
import torch
from openai import AsyncOpenAI
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.platforms import current_platform
from vllm.renderers import renderer_from_config
from vllm.sampling_params import RequestOutputKind
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_compression.scheduler import CompressionScheduler, session_key
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager

from .kv_compression_utils import bind_cpu_worker

TOOL = {
    "name": "read_status",
    "description": "Read status",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
TOOL_TEXT = (
    "<tool_call>\n<function=read_status>\n"
    "<parameter=path>tests</parameter>\n</function>\n</tool_call>"
)
TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:"
    "{% if m.tool_calls %}<think></think>{% for call in m.tool_calls %}"
    "<tool_call>\n<function={{ call.function.name }}>\n"
    "<parameter=path>{{ call.function.arguments.path }}</parameter>\n"
    "</function>\n</tool_call>"
    "{% endfor %}{% else %}{{ m.content or '' }}{% endif %}{{ '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}assistant:<think>{% endif %}"
)
MIDDLEWARE = "vllm.v1.kv_compression.middleware.SessionMiddleware"


class ScriptedEngine:
    """Replace model sampling with known token IDs, keeping the native scheduler."""

    errored = False
    io_processor = None

    def __init__(self, config, monkeypatch):
        self.vllm_config = config
        self.model_config = config.model_config
        self.renderer = renderer_from_config(config)
        self.input_processor = InputProcessor(config, self.renderer)
        spec = FullAttentionSpec(
            block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        pages = KVCacheConfig(64, [], [KVCacheGroupSpec(["attn"], spec)])
        self.scheduler = CompressionScheduler(
            config,
            pages,
            StructuredOutputManager(config),
            block_size=8,
            log_stats=False,
        )
        bind_cpu_worker(
            self.scheduler.kv_cache_manager, monkeypatch, protected_tokens=2
        )
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreRequest)
        self.requests = []

    async def is_tracing_enabled(self):
        return False

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        # The asynchronous engine consumes a stream in a different task from
        # the HTTP route. Interleave requests before reading the context.
        await asyncio.sleep(0)
        core = self.input_processor.process_inputs(
            request_id, prompt, sampling_params, ("generate",)
        )
        request = Request.from_engine_core_request(
            self.decoder.decode(self.encoder.encode(core)), block_hasher=None
        )
        self.scheduler.add_request(request)
        tokenizer = self.renderer.get_tokenizer()
        ids = core.prompt_token_ids
        text = "</think>" + (
            "all green" if "tool:green" in tokenizer.decode(ids) else TOOL_TEXT
        )
        output_ids = tokenizer.encode(text, add_special_tokens=False)
        for token in [*output_ids, tokenizer.eos_token_id]:
            batch = self.scheduler.schedule()
            self.scheduler.update_from_output(
                batch,
                ModelRunnerOutput(
                    req_ids=[request_id],
                    req_id_to_index={request_id: 0},
                    sampled_token_ids=[[token]],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        assert request.is_finished()
        self.requests.append(request)
        pieces = [(text, output_ids)]
        if sampling_params.output_kind == RequestOutputKind.DELTA:
            pieces = [(tokenizer.decode([token]), [token]) for token in output_ids]
            pieces.append(("", [tokenizer.eos_token_id]))
        for i, (piece, token_ids) in enumerate(pieces):
            finished = i == len(pieces) - 1
            yield RequestOutput(
                request_id,
                None,
                ids,
                None,
                [
                    CompletionOutput(
                        0,
                        piece,
                        token_ids,
                        None,
                        None,
                        finish_reason="stop" if finished else None,
                    )
                ],
                finished=finished,
                num_cached_tokens=request.num_cached_tokens,
            )


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    # A source-only checkout has no installed platform metadata for CLI defaults.
    if not current_platform.device_type:
        monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0")
    markers = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["<eos>", *markers]
    raw = Tokenizer(models.BPE({s: i for i, s in enumerate(alphabet)}, []))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="<eos>",
        additional_special_tokens=markers,
        chat_template=TEMPLATE,
    )
    tokenizer.save_pretrained(tmp_path)
    Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
    ).save_pretrained(tmp_path)
    model = ModelConfig(model=str(tmp_path), dtype="float32", enforce_eager=True)
    config = VllmConfig(model_config=model, device_config=DeviceConfig("cpu"))
    config.cache_config.enable_prefix_caching = False
    config.cache_config.num_gpu_blocks = 64
    config.additional_config = {"kv_compression": {"policy": "example.KeepLast"}}
    engine = ScriptedEngine(config, monkeypatch)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        [
            "--model",
            str(tmp_path),
            "--served-model-name",
            "agent",
            "--middleware",
            MIDDLEWARE,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--structured-outputs-config",
            '{"reasoning_parser":"qwen3"}',
        ]
    )
    app = build_app(args, ("generate",), model)
    return app, engine, args


async def replay_responses(client, thread_id, *, salt=None, rewrite=False):
    user = {"role": "user", "content": f"check {thread_id} {salt or 'default'}"}
    opts = {
        "model": "agent",
        "tools": [{"type": "function", **TOOL}],
        "store": False,
        "extra_headers": {"thread-id": thread_id},
        "extra_body": {"cache_salt": salt} if salt else {},
    }
    first_body = {**opts["extra_body"], "vllm_xargs": {"kv_protected_tokens": 1}}
    stream = await client.responses.create(
        input=[user], stream=True, **{**opts, "extra_body": first_body}
    )
    async with stream:
        events = [event async for event in stream]
    assert events[-1].type == "response.completed"
    (call,) = [
        event.item
        for event in events
        if event.type == "response.output_item.done"
        and event.item.type == "function_call"
    ]
    assert call.name == TOOL["name"] and json.loads(call.arguments) == {"path": "tests"}
    second = await client.responses.create(
        input=[
            {**user, "content": "edited history"} if rewrite else user,
            call.to_dict(),
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": "green",
            },
        ],
        **opts,
    )
    assert second.output_text == "all green"


async def replay_chat(client, headers):
    user = {
        "role": "user",
        "content": "check opencode" if headers else "check anonymous",
    }
    opts = {
        "model": "agent",
        "tools": [{"type": "function", "function": {**TOOL, "strict": True}}],
        "extra_headers": headers,
    }
    async with client.beta.chat.completions.stream(
        messages=[user], extra_body={"vllm_xargs": {"kv_protected_tokens": 0}}, **opts
    ) as stream:
        completion = await stream.get_final_completion()
    (call,) = completion.choices[0].message.tool_calls
    assert call.function.name == TOOL["name"]
    assert json.loads(call.function.arguments) == {"path": "tests"}
    second = await client.chat.completions.create(
        messages=[
            user,
            {"role": "assistant", "tool_calls": [call.to_dict()]},
            {"role": "tool", "tool_call_id": call.id, "content": "green"},
        ],
        **opts,
    )
    assert second.choices[0].message.content == "all green"


async def replay_messages(http, agent=None):
    headers = {"x-claude-code-session-id": "same"}
    if agent:
        headers["x-claude-code-agent-id"] = agent
    user = {"role": "user", "content": f"check claude {agent or 'main'}"}
    opts = {
        "model": "agent",
        "max_tokens": 128,
        "tools": [
            {
                "name": TOOL["name"],
                "description": TOOL["description"],
                "input_schema": TOOL["parameters"],
            }
        ],
    }
    first = await http.post(
        "/v1/messages",
        headers=headers,
        json={**opts, "messages": [user], "stream": True},
    )
    first.raise_for_status()
    events = [
        json.loads(line[6:])
        for line in first.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "message_stop"
    (call,) = [
        event["content_block"]
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    ]
    call["input"] = json.loads(
        "".join(
            event["delta"]["partial_json"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
    )
    assert call["name"] == TOOL["name"] and call["input"] == {"path": "tests"}
    result = {"type": "tool_result", "tool_use_id": call["id"], "content": "green"}
    second = await http.post(
        "/v1/messages",
        headers=headers,
        json={
            **opts,
            "messages": [
                user,
                {"role": "assistant", "content": [call]},
                {"role": "user", "content": [result]},
            ],
        },
    )
    second.raise_for_status()
    assert [b["text"] for b in second.json()["content"] if b["type"] == "text"] == [
        "all green"
    ]


def test_native_tool_turns_resume_only_their_own_session(http_server):
    app, engine, args = http_server

    async def run():
        await init_app_state(engine, app.state, args, ("generate",))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            client = AsyncOpenAI(
                base_url="http://test/v1",
                api_key="local",
                http_client=http,
                max_retries=0,
            )
            await asyncio.gather(
                replay_responses(client, "same"),
                replay_responses(client, "same", salt="private"),
                replay_responses(client, "edited", rewrite=True),
                replay_chat(client, {"x-opencode-session-id": "same"}),
                replay_messages(http),
                replay_messages(http, "one"),
                replay_messages(http, "two"),
                replay_chat(client, {}),
            )

    asyncio.run(run())
    turns = {}
    for req in engine.requests:
        turns.setdefault(session_key(req), []).append(req)
    edited = (None, ("codex", "edited"))
    sessions = {
        (None, ("codex", "same")),
        ("private", ("codex", "same")),
        edited,
        (None, ("opencode", "same")),
        (None, ("claude-code", "same")),
        (None, ("claude-code", "same", "one")),
        (None, ("claude-code", "same", "two")),
    }
    assert set(turns) == sessions | {None}
    for key, (first, second) in turns.items():
        assert first.num_cached_tokens == 0
        seen = first.num_computed_tokens
        matches = second.prompt_token_ids[:seen] == first.all_token_ids[:seen]
        assert matches == (key != edited)
        assert second.num_cached_tokens == (seen if matches and key is not None else 0)
    cache = engine.scheduler.kv_cache_manager
    assert set(cache.idle) == sessions
    # Clients supply their boundary on the first turn. The second inherits it.
    for (_, (client_kind, *_)), snap in cache.idle.items():
        expected = {"codex": 1, "opencode": 0, "claude-code": 2}[client_kind]
        assert snap.protected == expected
        assert snap.retained == expected + 1  # KeepLast retains one unprotected slot
    assert not cache.active and not cache.claimed
