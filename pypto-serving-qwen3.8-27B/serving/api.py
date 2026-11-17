"""OpenAI-style completions and chat over the resident PyPTO engine."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import resource
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .engine import MODEL_NAME, InferenceEngine
from .distributed import ServingConfig, SamplingParams


class CompletionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_NAME
    prompt: str | list[int]
    repetition_penalty: Literal[1.0] = 1.0
    logprobs: None = None
    max_tokens: int = Field(default=32, ge=1, le=262144,
                            validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    temperature: float = Field(default=0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    ignore_eos: bool = False
    return_token_ids: Literal[True] = True
    stream_options: dict[str, bool] | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[TextContent]


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_NAME
    messages: list[ChatMessage] = Field(min_length=1)
    chat_template_kwargs: dict[str, bool] | None = None
    max_tokens: int = Field(default=32, ge=1, le=262144,
                            validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    temperature: float = Field(default=0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    ignore_eos: bool = False
    return_token_ids: Literal[True] = True
    stream_options: dict[str, bool] | None = None
    stream: bool = False
    stop: str | list[str] | None = None


def stop_strings(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    if len(values) > 8 or any(not item for item in values):
        raise HTTPException(400, "Stop must contain 1-8 nonempty strings")
    return values


def validate(body) -> SamplingParams:
    if body.model != MODEL_NAME:
        raise HTTPException(400, "Unsupported model")
    if body.stream_options is not None and set(body.stream_options) - {"include_usage"}:
        raise HTTPException(400, "Unsupported stream option")
    template = getattr(body, "chat_template_kwargs", None)
    if template is not None and (set(template) - {"enable_thinking"} or template.get("enable_thinking", False)):
        raise HTTPException(400, "This service uses the non-thinking chat template")
    try:
        return SamplingParams(body.temperature, body.top_p, body.top_k, body.seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServingConfig.from_env()
    engine = InferenceEngine(config)
    await asyncio.to_thread(engine.start)
    app.state.engine = engine
    try:
        yield
    finally:
        await asyncio.to_thread(engine.stop)


app = FastAPI(title=MODEL_NAME, lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    engine: InferenceEngine = app.state.engine
    if engine.scheduler.error is not None or engine.scheduler.stopping.is_set():
        raise HTTPException(503, "PyPTO engine unavailable")
    return {"status": "ready", "model": MODEL_NAME, "device": engine.device_id,
            **engine.scheduler.counts(), "pid": os.getpid(),
            "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


@app.get("/metrics")
async def metrics() -> dict:
    return app.state.engine.snapshot()


async def next_event(job, request: Request):
    while True:
        if await request.is_disconnected():
            job.cancelled.set()
            return "cancelled", None
        try:
            return await asyncio.to_thread(job.events.get, True, 0.2)
        except queue.Empty:
            continue


async def respond(job, request: Request, chat: bool):
    engine: InferenceEngine = app.state.engine
    response_id = ("chatcmpl-" if chat else "cmpl-") + job.request_id
    created = int(time.time())
    object_name = "chat.completion" if chat else "text_completion"

    async def stream_events():
        emitted = ""
        generated: list[int] = []
        finished = None
        max_hold = max((len(item) for item in job.stop_strings), default=0)
        try:
            if chat:
                yield "data: " + json.dumps({"id": response_id, "object": "chat.completion.chunk",
                                             "created": created, "model": MODEL_NAME,
                                             "prompt_token_ids": job.prompt_ids,
                                             "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                          "finish_reason": None}]}) + "\n\n"
            while True:
                kind, value = await next_event(job, request)
                if kind == "token":
                    generated = list(value["generated"])
                    finished = value["finish_reason"]
                    text = value["text"]
                    if value["stop_match"]:
                        text = text.split(value["stop_match"], 1)[0]
                    safe = len(text) if finished else max(0, len(text) - max_hold + 1)
                    if not text.startswith(emitted):
                        raise RuntimeError("Tokenizer output changed an already streamed prefix")
                    delta = text[len(emitted):safe]
                    emitted += delta
                    choice = ({"index": 0, "delta": {"content": delta},
                               "token_ids": [value["token_id"]], "finish_reason": finished}
                              if chat else
                              {"index": 0, "text": delta,
                               "token_ids": [value["token_id"]], "finish_reason": finished})
                    yield "data: " + json.dumps({"id": response_id,
                                                 "object": object_name + ".chunk",
                                                 "created": created, "model": MODEL_NAME,
                                                 "choices": [choice]}, ensure_ascii=False) + "\n\n"
                elif kind == "done":
                    usage = {"prompt_tokens": len(job.prompt_ids),
                             "completion_tokens": len(generated),
                             "total_tokens": len(job.prompt_ids) + len(generated)}
                    if getattr(job, "include_usage", False):
                        yield "data: " + json.dumps({"id": response_id, "object": object_name + ".chunk",
                                                 "created": created, "model": MODEL_NAME,
                                                 "choices": [], "usage": usage,
                                                 "metrics": value["metrics"]}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                elif kind == "error":
                    yield "data: " + json.dumps({"error": str(value)}) + "\n\n"
                    return
                else:
                    return
        finally:
            job.cancelled.set()

    if request is None:
        raise RuntimeError("HTTP request context is required")
    if getattr(job, "stream", False):
        return StreamingResponse(stream_events(), media_type="text/event-stream")
    try:
        generated: list[int] = []
        finish_reason = None
        stop_match = None
        timing = None
        while True:
            kind, value = await next_event(job, request)
            if kind == "token":
                generated = list(value["generated"])
                finish_reason = value["finish_reason"]
                stop_match = value["stop_match"]
            elif kind == "done":
                timing = value["metrics"]
                break
            elif kind == "error":
                raise HTTPException(500, str(value))
            else:
                raise HTTPException(499, "Client disconnected")
        text = job.text.split(stop_match, 1)[0] if stop_match else job.text
        usage = {"prompt_tokens": len(job.prompt_ids), "completion_tokens": len(generated),
                 "total_tokens": len(job.prompt_ids) + len(generated)}
        choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                   "token_ids": generated, "finish_reason": finish_reason}
                  if chat else {"index": 0, "text": text,
                                "token_ids": generated, "finish_reason": finish_reason})
        return {"id": response_id, "object": object_name, "created": created,
                "model": MODEL_NAME, "choices": [choice], "usage": usage,
                "prompt_token_ids": job.prompt_ids,
                "metrics": timing}
    finally:
        job.cancelled.set()


@app.post("/v1/completions")
async def completions(body: CompletionInput, request: Request):
    sampling = validate(body)
    engine: InferenceEngine = app.state.engine
    try:
        stops = stop_strings(body.stop)
        job = (engine.submit_text(body.prompt, body.max_tokens, stops, sampling, body.ignore_eos)
               if isinstance(body.prompt, str) else
               engine.submit_tokens(body.prompt, body.max_tokens, stops, sampling, body.ignore_eos))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job.stream = body.stream
    job.include_usage = bool((body.stream_options or {}).get("include_usage", False))
    return await respond(job, request, False)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatInput, request: Request):
    sampling = validate(body)
    engine: InferenceEngine = app.state.engine
    try:
        messages = [{"role": message.role,
                     "content": message.content if isinstance(message.content, str)
                     else "".join(part.text for part in message.content)}
                    for message in body.messages]
        job = engine.submit_chat(messages,
                                 body.max_tokens, stop_strings(body.stop), sampling, body.ignore_eos)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job.stream = body.stream
    job.include_usage = bool((body.stream_options or {}).get("include_usage", False))
    return await respond(job, request, True)
