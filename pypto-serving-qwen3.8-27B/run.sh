#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run.sh --model-root PATH --tp N --ep N --dp N [--host HOST --port PORT --devices IDS --max-active N --backend auto|optimized|legacy --mixed-prefill 0|1 --async-prepare 0|1]

Use --devices 0,1 or set TASK_DEVICE to the assigned device IDs. EP must be 1.
Supported topologies: TP=1 DP=1, TP=2 DP=1, TP=1 DP=2.
EOF
}

model_root=""
tp=1
ep=1
dp=1
host=127.0.0.1
port=8000
devices="${TASK_DEVICE:-0}"
max_active="${MAX_ACTIVE:-2}"
backend="${PYPTO_BACKEND:-auto}"
mixed_prefill="${PYPTO_MIXED_PREFILL:-0}"
async_prepare="${PYPTO_ASYNC_PREPARE:-0}"

while (($#)); do
  case "$1" in
    --) shift ;;
    --help|-h) usage; exit 0 ;;
    --model-root|--tp|--ep|--dp|--host|--port|--devices|--max-active|--backend|--mixed-prefill|--async-prepare)
      key="$1"
      if (($# < 2)); then echo "Missing value for $key" >&2; exit 2; fi
      value="$2"
      case "$key" in
        --model-root) model_root="$value" ;;
        --tp) tp="$value" ;;
        --ep) ep="$value" ;;
        --dp) dp="$value" ;;
        --host) host="$value" ;;
        --port) port="$value" ;;
        --devices) devices="$value" ;;
        --max-active) max_active="$value" ;;
        --backend) backend="$value" ;;
        --mixed-prefill) mixed_prefill="$value" ;;
        --async-prepare) async_prepare="$value" ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$model_root" || ! -d "$model_root" ]]; then
  echo "--model-root must identify an existing checkpoint directory" >&2
  exit 2
fi
if [[ "$ep" != 1 || ! "$tp" =~ ^[12]$ || ! "$dp" =~ ^[12]$ || "$((tp * dp))" -gt 2 ]]; then
  echo "Supported topologies are TP1/DP1, TP2/DP1, TP1/DP2, all with EP1" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "--port must be between 1 and 65535" >&2
  exit 2
fi
if [[ ! "$max_active" =~ ^[0-9]+$ ]] || ((max_active < 1 || max_active > 8)); then
  echo "--max-active must be between 1 and 8" >&2
  exit 2
fi
case "$backend" in
  auto|optimized|legacy) ;;
  *) echo "--backend must be auto, optimized, or legacy" >&2; exit 2 ;;
esac
if [[ ! "$mixed_prefill" =~ ^[01]$ || ! "$async_prepare" =~ ^[01]$ ]]; then
  echo "--mixed-prefill and --async-prepare must be 0 or 1" >&2
  exit 2
fi
IFS=',' read -r -a device_ids <<< "$devices"
if ((${#device_ids[@]} != tp * dp)); then
  echo "Device count must match TP * DP" >&2
  exit 2
fi
for device in "${device_ids[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ ]]; then
    echo "Device IDs must be nonnegative integers" >&2
    exit 2
  fi
done

export MODEL_ROOT="$model_root" TP="$tp" DP="$dp" EP="$ep" TASK_DEVICE="$devices" MAX_ACTIVE="$max_active"
export PYPTO_BACKEND="$backend" PYPTO_MIXED_PREFILL="$mixed_prefill" PYPTO_ASYNC_PREPARE="$async_prepare"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m uvicorn serving.api:app --host "$host" --port "$port"
