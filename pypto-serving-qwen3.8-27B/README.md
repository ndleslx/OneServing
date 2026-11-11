# Qwen3.8-27B PyPTO serving

This package includes the recent source-compiled single-device Qwen optimizations and keeps the existing multi-device implementation as a fallback. Weights and the Ascend/PyPTO environment remain external.

## Run

Install PyPTO at the full commit in `pypto.pin`, its pinned Simpler/PTOAS dependencies, and `requirements.txt` in the Ascend environment. The optimized path requires Python 3.11 or later and the validated NZ binding contract. Its logical-shape compatibility is installed within the service process; it does not edit installed library files or require a harness checkout.

```bash
./run.sh --model-root /absolute/path/to/Qwen3.8-27B --tp 1 --dp 1 --ep 1 \
  --devices 4 --max-active 8 --port 8000
```

`--backend auto` selects the optimized backend for TP1/DP1 and retains the existing backend for TP2/DP1 or TP1/DP2. `--backend legacy` selects the previous source explicitly. The active-request default remains 2; use `--max-active 4` or `8` for the intended concurrent load. Assign devices explicitly; no devices or model locations are hard-coded.

The optimized backend supports 2048 total input and output tokens and uses prefix reuse disabled (`PREFIX_CACHE=0`, the optimized default). The existing backend retains its original 512-token limit and prefix-cache behavior. The checkpoint identity remains revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 64 BF16 text layers, hidden size 5120, and vocabulary 248320.

## Included optimizations

The optimized runner compiles its graphs directly from `serving/optimized/kernels`, loads NZ weight banks through the packaged loader, uses 16/64-token batched and 512-token single-request prefill paths, rotates large prefill with decode, and preserves all resident owners when only part of the active set is selected. Decode uses the fused model/device greedy graph. Integer local state compaction avoids allocating a full duplicate recurrent-state bank.

The newer P128 shared-input gate/up and SwiGLU FFN model is included with its numerical evidence. It can be enabled with `--mixed-prefill 1`; the default remains disabled because the newly ported full service combination has not completed device lifecycle validation. `--async-prepare 1` enables the separately qualified host lookahead preparation algorithm and retains its runtime-version checks; it also defaults disabled. Neither option registers the rejected owned323 feedback path, prepare397, packed-prefill, or persistent multi-step decode.

## Numerical and performance evidence

Original source qualifications include the complete 44+3 model campaign (2175 device calls), the later integrated FFN campaign (1396 calls), and FFN whole-mixed-step checks (294 calls). The FFN model step improved 6.25–8.19% in six tested mixed cases. That result is not an HTTP TPOT improvement.

Ten exported graphs were freshly CPU compiled and their complete metadata matched the actual qualified graphs. All 575 generated PTO files matched after removing debug locations and renaming SSA values consistently; model operations, precision conversions, rounding, layout and loops were retained. The export does not include old prebuilt binaries or absolute harness/run/data paths.

The host source port passes scheduler/owner mock tests, including cancellation, EOS, budgets, error joining and preservation of unscheduled resident state. These checks establish source and host behavior; the newly assembled standalone service has not been requalified on a device. Historical model/HTTP evidence applies to its original frozen source, not automatically to this new packaging. See `docs/optimized-validation.md` for the exact source and evidence mapping. Keep deployment acceptance separate from the source PR.

## API

The service exposes `/health`, `/metrics`, `/v1/completions`, and `/v1/chat/completions`; requests are greedy (`temperature=0`). The optimized API supports normal and SSE text/chat, generated token IDs, stop strings/token IDs, `ignore_eos`, and checkpoint EOS configuration. Preserve the input-plus-output context limit. `run.sh --help` lists backend and topology options.

A short probe is available through `python benchmarks/verify_candidate.py --mode correctness` or `--mode benchmark`, with the external checkpoint selected by `PYPTO_HARNESS_MODEL_ROOT`. The probe propagates `--max-active` to `run.sh`; it is a small smoke/throughput probe rather than the full numerical or service acceptance suite.
