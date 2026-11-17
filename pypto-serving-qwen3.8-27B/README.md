# Qwen3.8-27B PyPTO Serving

A standalone BF16 text-generation service for Qwen3.8-27B on two Ascend A3
devices with TP2/DP1/EP1. It supports unified prefill/decode execution,
continuous batching, prefix caching and three-token MTP speculation.

## Model and execution

The external checkpoint uses the Qwen3.5 text architecture: 64 layers,
including 48 Gated DeltaNet layers and 16 full-attention layers, a hidden size
of 5120 and a vocabulary size of 248320. Backbone matrices and attention heads
are sharded across both tensor-parallel ranks.

PyPTO L3 `DistributedWorker` runs a unified target `model_fwd` and a separate
MTP drafter. Target execution packs prefill and decode/verification tokens
into one forward, with request buckets of 1, 4, 16 and 32. Packed token rows
are aligned to 16; kernel tiles and row alignment are separate from the
scheduler's per-request prefill chunk length.

| Capacity setting | Default |
| --- | ---: |
| Maximum model length | 133120 |
| Per-step token budget | 16384 |
| Maximum active requests | 32 |
| Extra per-request prefill cap | 0 (budget-driven) |
| MTP draft tokens | 3 |
| Prefix cache | Enabled |

The model-length limit applies to prompt plus generated tokens. KV capacity
also limits admission: configuring 32 active requests does not reserve enough
KV pages for 32 requests at the maximum model length. Requests fail explicitly
when the required pages cannot be allocated.

The runtime manages paged KV, convolution and recurrent state, 32-token prefix
checkpoints, MTP acceptance/rejection recovery and request-state ownership.
Weights and reusable hidden-state consumers stay on the device. Greedy sampling
uses device reductions with compact per-rank results; stochastic sampling and
parts of TP/MTP coordination retain host-side processing.

Optimized kernels include GQA decode attention, C16 GDN for long sequences,
fused NZ FFN gate/up, row-size-specific matrix kernels, T16 convolution,
M128 projection prefixes with tail handling and a two-stage FFN down K pipeline.

## Environment

Use Python 3.12, an Ascend driver and the compatible PyPTO toolchain. Install
the full PyPTO revision recorded in `pypto.pin` together with its matching
Simpler runtime, PTOAS and pto-isa dependencies, then install this package's
Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The pinned runtime's device-access path was validated with CANN 9.0. Other
CANN versions require separate validation. Model weights, wheels, virtual
environments and toolchains are external to this package. Serving does not
require a checkout of pypto-lib or pypto-serving.

Generated compilation artifacts use an external user-cache directory by
default. Set `PYPTO_SERVING_CACHE_DIR` to choose a persistent location. A new
source path or toolchain identity can require compilation before the service
becomes ready.

## Start the service

Activate the matching environment and use `task-submit` to allocate a pair of
devices that can communicate. For a deployment authorized to use physical
devices 8 and 9:

```bash
export TASKQUEUE_DEVICE_CONF=/absolute/path/to/task-submit.conf
task-submit --device 8,9 --max-time 0 --timeout 0 --env TASKQUEUE_DEVICE_CONF --run \
  'bash /absolute/path/to/pypto-serving-qwen3.8-27B/run.sh \
    --model-root /absolute/path/to/Qwen3.8-27B --devices 8,9 \
    --tp 2 --dp 1 --ep 1 --max-model-len 133120 \
    --max-num-batched-tokens 16384 --max-num-seqs 32 \
    --long-prefill-token-threshold 0 \
    --speculative-method qwen3_5_mtp --num-speculative-tokens 3 \
    --prefix-cache 1 --host 127.0.0.1 --port 8000 \
    --served-model-name qwen3.8-27B'
```

`--devices` must exactly match the scheduler's `TASK_DEVICE` assignment.
`run.sh` maps those physical devices to logical ranks inside the process.
Stop the service through its owning job so that the service, workers and
allocated resources are released together.

## HTTP API

The service exposes `/health`, `/metrics`, `/v1/completions` and
`/v1/chat/completions`. The default served model name is `qwen3.8-27B`.
Completions accept text or token IDs. Chat requests support system, user and
assistant messages with text content or text content parts, using the
checkpoint's non-thinking chat template.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27B","messages":[{"role":"user","content":"What is the capital of France? Answer briefly."}],"max_tokens":64,"temperature":0}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27B","messages":[{"role":"user","content":"Explain tensor parallelism briefly."}],"max_tokens":64,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Supported sampling parameters are `temperature`, `top_p`, `top_k` and an
optional request-local `seed`. A temperature of zero selects greedy sampling.
`max_completion_tokens` is an alias for `max_tokens`, whose default is 32.
Prompt plus requested output must fit within the configured model length.

`stop` accepts one string or up to eight nonempty strings. EOS stopping is
enabled by default; `ignore_eos=true` supports fixed-output-length workloads.
Streaming usage is returned when `stream_options.include_usage=true` and
counts only committed tokens. Set `--num-speculative-tokens 0` to disable MTP
for diagnostic runs, and `--prefix-cache 0` to disable prefix caching.

`/metrics` reports request timing and state, prefix-cache activity, actual
batch/slot mappings, speculative acceptance, memory and source/compiled-artifact
provenance.

## Prefill chunk scheduling

`--long-prefill-token-threshold 0` uses budget-driven chunking. Each step first
reserves capacity for existing decode/MTP verification work, then assigns the
remaining token budget to prefill requests using bounded round-robin fairness.
A request can finish its remaining prompt in one step when the budget permits.
The budget is not divided equally among all active requests.

A positive threshold, such as 1024 or 2048, adds a per-request cap. The actual
chunk length also depends on the remaining prompt, available budget and prefix
checkpoints. Requests that are not selected continue in later steps. The
unified target entry retains mixed prefill/decode execution.

The threshold must be a nonnegative integer no greater than `--max-model-len`.
Invalid values are rejected before model loading or device initialization.
Zero removes the extra request cap; allocation and compilation dimensions
remain bounded by the configured context and token budget.

## Performance and validation

The best recorded run used BF16, A3 TP2/DP1/EP1, concurrency 32, 8192 input
tokens and 512 output tokens. It enabled a shared 32-token prefix and MTP3,
with a token budget of 16384 and prefill threshold zero. All 50 measured
requests succeeded after ten total warmups: one 33-token prefix seed and nine
full requests. Results are from one measured round.

| Metric | Result |
| --- | ---: |
| Output throughput (tokens/s) | 96.0348 |
| TTFT P50 (ms) | 30173.947 |
| TTFT P95 (ms) | 116877.546 |
| TPOT P50 (ms) | 230.661 |
| TPOT P95 (ms) | 367.885 |

These results were measured on November 16, 2026. TTFT uses the arrival of
the first generated token ID. TPOT is the interval between the last and first
generated token IDs divided by output length minus one. Role-only, usage-only
and empty control events do not count as the first token.

Fresh short Chinese and English HTTP generation from this checkout passed on
A3 TP2 on November 17, with prefix caching, MTP3 and normal EOS stopping.
Both requests finished with `finish_reason=stop`. Local CPU checks covered
scheduler budgeting, greedy MTP verification, prefix-page ownership and source
identity. Full dataset accuracy and final independent deployment acceptance
remain pending.
