"""OpenAI-style completions and chat over the resident PyPTO engine."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import resource
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .engine import MODEL_NAME, InferenceEngine
from .distributed import ReplicaRouter, TensorParallelEngine


class CompletionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_NAME
    prompt: str | list[int]
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = 0
    stream: bool = False
    stop: str | list[str] | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_NAME
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = 0
    stream: bool = False
    stop: str | list[str] | None = None


def stop_strings(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    if len(values) > 8 or any(not item for item in values):
        raise HTTPException(400, "Stop must contain 1-8 nonempty strings")
    return values


def validate(model: str, temperature: float) -> None:
    if model != MODEL_NAME:
        raise HTTPException(400, "Unsupported model")
    if temperature != 0:
        raise HTTPException(400, "Only greedy temperature=0 is supported")


@asynccontextmanager
async def lifespan(app: FastAPI):
    assignment = os.environ.get("TASK_DEVICE", "0")
    devices = tuple(int(value) for value in assignment.split(","))
    dp = int(os.environ.get("DP", "1"))
    tp = int(os.environ.get("TP", "1"))
    root = Path(os.environ["MODEL_ROOT"])
    max_active = int(os.environ.get("MAX_ACTIVE", "2"))
    prefix_setting = os.environ.get("PREFIX_CACHE", "1")
    if prefix_setting not in ("0", "1"):
        raise RuntimeError("PREFIX_CACHE must be 0 or 1")
    prefix_cache = prefix_setting == "1"
    if dp == 1 and tp == 1 and len(devices) == 1:
        engine = InferenceEngine(root, devices[0], max_active=max_active,
                                 prefix_cache=prefix_cache)
    elif dp == 2 and tp == 1 and len(devices) == 2:
        engine = ReplicaRouter(root, devices, max_active=max_active,
                               prefix_cache=prefix_cache)
    elif dp == 1 and tp == 2 and len(devices) == 2:
        engine = TensorParallelEngine(root, devices, max_active=max_active,
                                      prefix_cache=prefix_cache)
    else:
        raise RuntimeError("DP/TP topology does not match assigned device count")
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


def visible_text(engine: InferenceEngine, token_ids: list[int], stop_match: str | None) -> str:
    text = engine.decode(token_ids)
    return text.split(stop_match, 1)[0] if stop_match else text


async def respond(job, request: Request, chat: bool):
    engine: InferenceEngine = app.state.engine
    response_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
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
                                             "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                          "finish_reason": None}]}) + "\n\n"
            while True:
                kind, value = await next_event(job, request)
                if kind == "token":
                    generated = list(value["generated"])
                    finished = value["finish_reason"]
                    text = visible_text(engine, generated, value["stop_match"])
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
        text = visible_text(engine, generated, stop_match)
        usage = {"prompt_tokens": len(job.prompt_ids), "completion_tokens": len(generated),
                 "total_tokens": len(job.prompt_ids) + len(generated)}
        choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                   "token_ids": generated, "finish_reason": finish_reason}
                  if chat else {"index": 0, "text": text,
                                "token_ids": generated, "finish_reason": finish_reason})
        return {"id": response_id, "object": object_name, "created": created,
                "model": MODEL_NAME, "choices": [choice], "usage": usage,
                "metrics": timing}
    finally:
        job.cancelled.set()


@app.post("/v1/completions")
async def completions(body: CompletionInput, request: Request):
    validate(body.model, body.temperature)
    engine: InferenceEngine = app.state.engine
    try:
        stops = stop_strings(body.stop)
        job = (engine.submit_text(body.prompt, body.max_tokens, stops)
               if isinstance(body.prompt, str) else
               engine.submit_tokens(body.prompt, body.max_tokens, stops))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job.stream = body.stream
    return await respond(job, request, False)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatInput, request: Request):
    validate(body.model, body.temperature)
    engine: InferenceEngine = app.state.engine
    try:
        job = engine.submit_chat([message.model_dump() for message in body.messages],
                                 body.max_tokens, stop_strings(body.stop))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job.stream = body.stream
    return await respond(job, request, True)
