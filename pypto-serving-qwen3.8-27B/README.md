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
scheduler defaults to two active requests per engine; TP1 accepts up to the
compiled batch capacity of eight, and TP2 retains its two-request limit.
The source includes continuous batching, 16-token chunked
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
Those smoke tests used two-token outputs. The TP1 mixed execution section below
records the subsequent numerical, prefix reuse and throughput validation.

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


## Experimental TP1 mixed execution

Enable with `PYPTO_QWEN_MIXED=1`. TP1 accepts `MAX_ACTIVE` up to the existing
model capacity (`BATCH=8`); the default remains two. TP2 retains its original
two-active-request limit. For example:

```bash
MAX_ACTIVE=8 PYPTO_QWEN_MIXED=1 ./run.sh \
  --model-root /absolute/path/to/Qwen3.8-27B --tp 1 --dp 1 --ep 1
```

With mixed execution enabled, prefill, decode and mixed batches use the same
graph. Effective token rows stay contiguous from embedding through all layers
and the output head. RMSNorm, Attention input/output projections and MLP run on
this layout, using 32-row projection tiles and padding only the tail. Attention
cores operate on request segments and write only their valid rows; hidden
activations are not packed into and scattered out of the MLP at each layer.

Each request keeps its resident convolution and recurrent state slot. A
`state_slots` array maps this batch's request order to those resident slots,
while `packed_offsets` marks the token segments. Reordering or replacing batch
members therefore does not copy recurrent state through the host. KV pages
continue to use each request's block table. Output logits and diagnostic traces
are addressed by resident slot. The disabled path retains the original graphs
for numerical and performance comparisons. A mixed runner loads only the mixed
graph; comparison tests call `start(include_reference_graphs=True)` to load the
original graphs alongside it.

The graph is experimental and disabled by default. Run the numerical and
performance checks below on the pinned stack for this implementation.

The 2026-11-05 TP1/DP1 validation passed host metadata tests, graph compilation,
PTOAS assembly and five full-network NPU correctness tests. On a fixed
eight-request HTTP trace, four repeated trials compared the original and packed
graphs at matching active-request limits. Output throughput improved by 40.3%
at capacity two and 93.1% at capacity eight; TPOT P95 fell by 34.7% and 48.5%,
respectively. All 128 measured requests succeeded with identical tokens. See the
[validation report](docs/mixed-execution-report.md).

Host metadata tests need only Python:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_mixed_batch.py -v
```

On the pinned PyPTO stack, compiler checks require no NPU or weights:

```bash
PYPTO_QWEN_TEST_COMPILE=1 PYTHONPATH=. \
  python -m unittest discover -s tests -p test_mixed_compile.py -v
```

For NPU tests, use a checkpoint that passes the existing loader checks:

```bash
PYPTO_QWEN_TEST_MODEL_ROOT=/absolute/path/to/Qwen3.8-27B \
PYPTO_QWEN_TEST_DEVICE=1 PYTHONPATH=. \
  python -m unittest discover -s tests -p test_mixed_npu.py -v
```

These compare logits, tokens, effective KV and conv/recurrent state against the
baseline, including multi-request batches, slot permutations, compaction and
prefix reuse. To compare a fixed workload after warmup, use the same environment:

```bash
PYTHONPATH=. python tests/benchmark_mixed_npu.py --prefills 4 --decodes 2
```

The benchmark reports input token rate and step latency; it does not measure
HTTP throughput, TTFT or TPOT. KV pool capacity still limits aggregate context
length across active requests.
