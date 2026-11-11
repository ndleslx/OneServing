#!/usr/bin/env python3
"""Start one candidate and run a fixed correctness or throughput probe.

The harness calls this script from an isolated candidate directory.  The
checkpoint stays outside the source tree and is selected with
``PYPTO_HARNESS_MODEL_ROOT`` (or ``MODEL_ROOT``).  The probe is intentionally
small; formal campaigns should increase rounds after the baseline noise floor
has been measured.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _request(port: int, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        result = json.loads(response.read())
    if not isinstance(result, dict) or not result.get("choices"):
        raise RuntimeError(f"invalid response: {result!r}")
    metrics = result.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise RuntimeError("missing metrics")
    token_ids = result["choices"][0].get("token_ids", [])
    if not isinstance(token_ids, list) or any(not isinstance(token, int) for token in token_ids):
        raise RuntimeError(f"invalid token_ids: {token_ids!r}")
    return {"tokens": len(token_ids), "token_ids": token_ids,
            "ttft_ms": metrics.get("ttft_ms"),
            "tpot_ms": metrics.get("tpot_ms")}


def _wait_ready(port: int, process: subprocess.Popen[str], timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready")


def _start(root: Path, model_root: Path, devices: str, tp: int, dp: int,
           port: int, max_active: int) -> subprocess.Popen[str]:
    command = [str(root / "run.sh"), "--model-root", str(model_root),
               "--tp", str(tp), "--dp", str(dp), "--ep", "1", "--devices", devices,
               "--port", str(port), "--max-active", str(max_active)]
    log_dir = Path(os.environ.get("PYPTO_HARNESS_RUN_DIR", "."))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"server-{uuid.uuid4().hex}.log").open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "benchmark"), required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default=os.environ.get("TASK_DEVICE", "4").split(",")[0])
    parser.add_argument("--max-active", type=int, default=int(os.environ.get("MAX_ACTIVE", "2")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("PYPTO_HARNESS_CONCURRENCY", "8")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("PYPTO_HARNESS_ROUNDS", "10")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("PYPTO_HARNESS_MAX_TOKENS", "16")))
    args = parser.parse_args()
    model_root = Path(os.environ.get("PYPTO_HARNESS_MODEL_ROOT", os.environ.get("MODEL_ROOT", ""))).resolve()
    if not model_root.is_dir():
        raise RuntimeError("PYPTO_HARNESS_MODEL_ROOT must point to the external checkpoint")
    root = Path(__file__).resolve().parents[1]
    devices = os.environ.get("PYPTO_HARNESS_DEVICES", args.device)
    tp = int(os.environ.get("PYPTO_HARNESS_TP", "1"))
    dp = int(os.environ.get("PYPTO_HARNESS_DP", "1"))
    process = _start(root, model_root, devices, tp, dp, args.port, args.max_active)
    try:
        _wait_ready(args.port, process)
        prompt = "Write one short sentence confirming that the PyPTO serving probe is ready."
        if args.mode == "correctness":
            result = _request(args.port, prompt, 2)
            if result["tokens"] < 1:
                raise RuntimeError("empty completion")
            expected = os.environ.get("PYPTO_HARNESS_EXPECTED_TOKENS")
            if expected:
                expected_ids = [int(value) for value in expected.split(",") if value.strip()]
                if result["token_ids"] != expected_ids:
                    raise RuntimeError(
                        f"token mismatch: expected {expected_ids}, got {result['token_ids']}"
                    )
            print(json.dumps({"correct": True, **result}))
            return 0

        for _ in range(2):
            _request(args.port, prompt, args.max_tokens)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_request, args.port, prompt, args.max_tokens)
                       for _ in range(args.concurrency * args.rounds)]
            results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        tokens = sum(int(result["tokens"]) for result in results)
        if tokens <= 0 or elapsed <= 0:
            raise RuntimeError("benchmark produced no tokens")
        ttft = [float(result["ttft_ms"]) for result in results if result["ttft_ms"] is not None]
        tpot = [float(result["tpot_ms"]) for result in results if result["tpot_ms"] is not None]
        print(json.dumps({"output_tok_s": tokens / elapsed,
                          "requests": len(results),
                          "concurrency": args.concurrency,
                          "elapsed_s": elapsed,
                          "ttft_ms_mean": sum(ttft) / len(ttft) if ttft else None,
                          "tpot_ms_mean": sum(tpot) / len(tpot) if tpot else None}))
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(2)
