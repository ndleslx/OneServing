#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: ./run.sh --model-root PATH [options]
  --devices IDS                 Physical devices assigned by task-submit
  --tp 2 --dp 1 --ep 1          Deployment topology
  --max-model-len 133120         Prompt + completion position capacity
  --long-prefill-token-threshold 0    Budget-driven; positive values add a per-request cap
  --max-num-batched-tokens 16384 Scheduling token budget
  --max-num-seqs 32              Active request capacity
  --speculative-method qwen3_5_mtp --num-speculative-tokens 3
  --prefix-cache 1               Prefix cache (0 disables for diagnostics)
  --host 127.0.0.1 --port 8000 --served-model-name qwen3.8-27B
Activate the matching PyPTO environment before launching through task-submit.
EOF
}
model_root=""
tp=2; dp=1; ep=1
host=127.0.0.1; port=8000
max_model_len=133120; max_num_batched_tokens=16384; max_num_seqs=32
long_prefill_token_threshold=0
speculative_method=qwen3_5_mtp; num_speculative_tokens=3; prefix_cache=1
served_model_name=qwen3.8-27B
devices="${TASK_DEVICE:-}"
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --model-root|--devices|--tp|--dp|--ep|--host|--port|--max-model-len|--long-prefill-token-threshold|--max-num-batched-tokens|--max-num-seqs|--speculative-method|--num-speculative-tokens|--prefix-cache|--served-model-name)
      key="$1"; (($# >= 2)) || { echo "Missing value for $key" >&2; exit 2; }
      value="$2"
      case "$key" in
        --model-root) model_root="$value";; --devices) devices="$value";;
        --tp) tp="$value";; --dp) dp="$value";; --ep) ep="$value";;
        --host) host="$value";; --port) port="$value";;
        --max-model-len) max_model_len="$value";;
        --long-prefill-token-threshold) long_prefill_token_threshold="$value";;
        --max-num-batched-tokens) max_num_batched_tokens="$value";;
        --max-num-seqs) max_num_seqs="$value";;
        --speculative-method) speculative_method="$value";;
        --num-speculative-tokens) num_speculative_tokens="$value";;
        --prefix-cache) prefix_cache="$value";;
        --served-model-name) served_model_name="$value";;
      esac
      shift 2;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -d "$model_root" ]] || { echo 'A real external --model-root is required' >&2; exit 2; }
[[ "$tp,$dp,$ep" == 2,1,1 ]] || { echo 'This package requires TP2/DP1/EP1' >&2; exit 2; }
[[ -n "${TASK_DEVICE:-}" && "$devices" == "$TASK_DEVICE" ]] || {
  echo '--devices must exactly match the task-submit physical allocation' >&2; exit 2;
}
[[ "$devices" =~ ^[0-9]+,[0-9]+$ ]] || { echo 'Exactly two physical devices are required' >&2; exit 2; }
IFS=',' read -r first second <<< "$devices"
[[ "$first" != "$second" ]] || { echo 'Devices must be distinct' >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || { echo 'Invalid port' >&2; exit 2; }
# Validate the coupled limits before importing the model or initializing devices.
"${PYTHON:-python}" - "$long_prefill_token_threshold" "$max_model_len" <<'PREFILL_VALIDATION'
import re, sys
value, limit = sys.argv[1:]
message = "--long-prefill-token-threshold must be a non-negative integer <= --max-model-len (0: budget-driven)"
if not re.fullmatch(r"[+]?[0-9]+", value) or not re.fullmatch(r"[+]?[0-9]+", limit):
    raise SystemExit(message)
if not 0 <= int(value) <= int(limit) or int(limit) <= 0:
    raise SystemExit(message)
PREFILL_VALIDATION
export LONG_PREFILL_TOKEN_THRESHOLD="$long_prefill_token_threshold"
export ASCEND_RT_VISIBLE_DEVICES="$devices" TASK_PHYSICAL_DEVICES="$devices" TASK_DEVICE=0,1
export MODEL_ROOT="$model_root" TP="$tp" DP="$dp" EP="$ep"
export MAX_MODEL_LEN="$max_model_len" MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens" MAX_NUM_SEQS="$max_num_seqs"
export SPECULATIVE_METHOD="$speculative_method" NUM_SPECULATIVE_TOKENS="$num_speculative_tokens"
export PREFIX_CACHE="$prefix_cache" SERVED_MODEL_NAME="$served_model_name"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"
runtime_cache="${PYPTO_SERVING_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pypto-serving-qwen3.8-27B}"
mkdir -p "$runtime_cache/build" "$runtime_cache/jit"
runtime_cache="$(cd "$runtime_cache" && pwd)"
export PYPTO_CACHE_DIR="$runtime_cache/jit" PYPTO_CACHE=1
if [[ -z "${PYPTO_PROG_BUILD_DIR:-}" ]]; then
  cd "$runtime_cache/build"
fi
exec "${PYTHON:-python}" -m uvicorn serving.api:app --host "$host" --port "$port"
