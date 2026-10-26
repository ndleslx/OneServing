# Qwen3.8-27B PyPTO text serving

This self-contained source package adapts the completed v2 service to the
`pypto-serving-{model-name}` delivery layout. It loads a read-only external
Qwen3.8-27B checkpoint and serves text completion and chat through PyPTO. It
does not bundle weights, PyPTO, or a Python environment. The v2 source was
validated before this layout/runtime conversion. This converted package passed
short end-to-end NPU smoke tests on 2026-10-26; the full v2 acceptance suite
has not been replayed against this directory.

## Model and deployment

The checkpoint contract is 64 BF16 text layers, hidden size 5120, vocabulary
248320, with three linear-attention layers followed by one full-attention layer
per group. The loader checks checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and config/index/tokenizer
digests. Maximum context is 512 tokens. RoPE tables are generated from the
checkpoint config at startup.

Supported topologies are TP1/DP1, TP2/DP1, and TP1/DP2; EP is 1 because this
model has no expert layers. Assign one physical NPU per TP/DP rank. The
single-device and TP runners both use PyPTO L3 `DistributedWorker`. The
scheduler admits at most two active requests per engine and accepts concurrent
queued requests. The source includes continuous batching, 16-token chunked
prefill, paged KV cache and prefix reuse, asynchronous request scheduling,
HTTP/SSE chat, and TTFT/TPOT metrics. The API emits partial text on each token
and uses the checkpoint chat template. Sampling is greedy only
(`temperature=0`); top-p, top-k, seed, speculative decode, PP, and EP are not
implemented.

## Verification scope

Using the PyPTO version in `pypto.pin`, TP1/DP1 started on one assigned NPU
and returned two generated tokens from `/v1/completions`. TP2/DP1 and TP1/DP2
each started on two assigned NPUs and completed a normal completion, two
concurrent completions, SSE chat, `/health`, `/metrics`, and clean shutdown.
The smoke tests used two-token outputs. Long-context behavior, prefix cache
reuse, cancellation, numerical parity, and throughput have not been rechecked
after this package conversion.

## Install and start

Use an Ascend Python 3.12 environment with compatible CANN and platform Torch.
Install PyPTO at the full commit in `pypto.pin`, together with its pinned
Simpler, PTOAS, and pto-isa stack. Then install this package's other dependencies:

```bash
python -m pip install -r requirements.txt
./run.sh --model-root /absolute/path/to/Qwen3.8-27B --tp 1 --dp 1 --ep 1 \
  --host 127.0.0.1 --port 8000
```

`run.sh --help` lists the options. It accepts `--` before options. By default,
the service uses device 0; set `TASK_DEVICE=2`, use `--devices 2`, or run inside
an allocation that provides `TASK_DEVICE`. For TP2 use two assigned devices:

```bash
./run.sh --model-root /absolute/path/to/Qwen3.8-27B --tp 2 --dp 1 --ep 1 --devices 2,3
```

## Requests

The service provides `/health`, `/metrics`, `/v1/completions`, and
`/v1/chat/completions`. Request `model` defaults to `Qwen3.8-27B-PyPTO`.
`prompt` is text or token IDs; chat `messages` accept system/user/assistant
roles. `max_tokens` defaults to 32. `stream=true` selects SSE. Optional `stop`
accepts one string or up to eight strings. Only `temperature=0` is accepted.

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":16,"temperature":0}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'
```

The package does not depend on a pypto-lib or pypto-serving source checkout.
