# gguf_runner.py — runs on Raspberry Pi
# pip install llama-cpp-python psutil

import time
import threading
import psutil
from pathlib import Path
from llama_cpp import Llama
from external/llama.cpp/build/bin import llama-cli

# ── Test prompts used for benchmarking ───────────────────────────
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in one sentence.",
    "What is 17 multiplied by 8?",
]

# ── RAM Monitor ───────────────────────────────────────────────────
def _monitor_ram(stop_event: threading.Event, results: dict):
    """Background thread — samples process RAM every 0.5s, records peak."""
    process = psutil.Process()
    peak = 0.0
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 ** 2)   # MB
        if mem > peak:
            peak = mem
        time.sleep(0.5)
    results["peak_ram_mb"] = peak


# ── Main Runner ───────────────────────────────────────────────────
def run_gguf(model_path: Path, quant_type: str) -> dict:
    """
    Loads a GGUF model, runs TEST_PROMPTS, measures:
      - avg latency per token (ms)
      - throughput (tokens/sec)
      - peak RAM usage (MB)
      - RAM utilization (%)
    
    Returns a results dict ready to POST to the bridge.
    """
    print(f"[Runner] Loading {model_path.name} ...")
    total_ram_mb = psutil.virtual_memory().total / (1024 ** 2)

    # ── Start RAM monitor thread ──────────────────────────────────
    ram_results = {}
    stop_event  = threading.Event()
    ram_thread  = threading.Thread(
        target=_monitor_ram,
        args=(stop_event, ram_results),
        daemon=True,
    )
    ram_thread.start()

    try:
        # ── Load model ────────────────────────────────────────────
        llm = Llama(
            model_path   = str(model_path),
            n_ctx        = 512,       # context window
            n_threads    = 4,         # Pi 4 = 4 cores
            n_gpu_layers = 0,         # CPU only — Pi has no GPU
            verbose      = False,
        )
        print(f"[Runner] Model loaded. Running {len(TEST_PROMPTS)} prompts ...")

        # ── Inference loop ────────────────────────────────────────
        latencies  = []
        sample_out = ""

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"[Runner] Prompt {i+1}/{len(TEST_PROMPTS)}: {prompt[:40]}...")

            t_start = time.perf_counter()
            output  = llm(
                prompt,
                max_tokens = 64,
                echo       = False,
                stop       = ["\n\n"],   # stop at double newline
            )
            t_end = time.perf_counter()

            generated_tokens = output["usage"]["completion_tokens"]
            elapsed_ms       = (t_end - t_start) * 1000
            latency_per_tok  = elapsed_ms / max(generated_tokens, 1)
            latencies.append(latency_per_tok)

            print(f"[Runner] → {generated_tokens} tokens | "
                  f"{latency_per_tok:.1f} ms/tok")

            # Save first response as sample
            if not sample_out:
                sample_out = output["choices"][0]["text"].strip()

        # ── Stop RAM monitor ──────────────────────────────────────
        stop_event.set()
        ram_thread.join()

        # ── Compute final metrics ─────────────────────────────────
        avg_latency    = sum(latencies) / len(latencies)
        tokens_per_sec = 1000 / avg_latency if avg_latency > 0 else 0
        peak_ram       = ram_results.get("peak_ram_mb", 0.0)

        print(f"[Runner] Done — avg {avg_latency:.1f} ms/tok | "
              f"{tokens_per_sec:.1f} tok/s | "
              f"peak RAM {peak_ram:.0f} MB")

        return {
            "quant_type":      quant_type,
            "avg_latency_ms":  round(avg_latency, 2),
            "tokens_per_sec":  round(tokens_per_sec, 2),
            "peak_ram_mb":     round(peak_ram, 1),
            "total_ram_mb":    round(total_ram_mb, 1),
            "ram_utilization": round(peak_ram / total_ram_mb, 3),
            "sample_output":   sample_out[:200],
            "status":          "success",
            "error":           None,
        }

    # ── Error handling ────────────────────────────────────────────
    except MemoryError:
        stop_event.set()
        print(f"[Runner] OOM — model too large for this device")
        return {
            "quant_type":      quant_type,
            "avg_latency_ms":  0,
            "tokens_per_sec":  0,
            "peak_ram_mb":     round(total_ram_mb, 1),
            "total_ram_mb":    round(total_ram_mb, 1),
            "ram_utilization": 1.0,
            "sample_output":   "",
            "status":          "oom",
            "error":           "Out of memory — model too large for this device",
        }

    except Exception as e:
        stop_event.set()
        print(f"[Runner] FAILED — {e}")
        return {
            "quant_type":      quant_type,
            "avg_latency_ms":  0,
            "tokens_per_sec":  0,
            "peak_ram_mb":     0,
            "total_ram_mb":    round(total_ram_mb, 1),
            "ram_utilization": 0,
            "sample_output":   "",
            "status":          "failed",
            "error":           str(e),
        }