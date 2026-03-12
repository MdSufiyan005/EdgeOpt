# pi_client.py  — runs on RASPBERRY PI as a daemon
# pip install requests psutil llama-cpp-python huggingface_hub

import time
import json
import psutil
import requests
import threading
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ── Config ────────────────────────────────────────────────────────
SERVER_URL   = "http://YOUR_MAIN_MACHINE_IP:8080"  # ← set this
MODELS_DIR   = Path("/home/pi/models")
POLL_INTERVAL = 10   # seconds between polls
TEST_PROMPTS  = [
    "What is the capital of France?",
    "Explain what a neural network is in one sentence.",
    "What is 17 multiplied by 8?",
]
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────
def poll_for_model() -> dict | None:
    try:
        r = requests.get(f"{SERVER_URL}/poll-model", timeout=10)
        data = r.json()
        return data if data.get("model_ready") else None
    except Exception as e:
        print(f"[Pi] Poll error: {e}")
        return None


def download_model(hf_repo_id: str, filename: str) -> Path:
    print(f"[Pi] Downloading {filename} from {hf_repo_id} ...")
    local_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR),
    )
    print(f"[Pi] Downloaded to {local_path}")
    return Path(local_path)


def measure_ram_peak(stop_event: threading.Event, results: dict):
    """Runs in background thread, samples RAM every 0.5s."""
    process = psutil.Process()
    peak = 0.0
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 ** 2)   # MB
        if mem > peak:
            peak = mem
        time.sleep(0.5)
    results["peak_ram_mb"] = peak


def run_inference(model_path: Path, quant_type: str) -> dict:
    print(f"[Pi] Loading model: {model_path.name}")
    total_ram_mb = psutil.virtual_memory().total / (1024 ** 2)

    # Start RAM monitor thread
    ram_results  = {}
    stop_event   = threading.Event()
    ram_thread   = threading.Thread(
        target=measure_ram_peak, args=(stop_event, ram_results), daemon=True
    )
    ram_thread.start()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_threads=4,          # Pi 4 has 4 cores
            n_gpu_layers=0,       # CPU only on Pi
            verbose=False,
        )

        latencies   = []
        sample_out  = ""

        for prompt in TEST_PROMPTS:
            t_start = time.perf_counter()
            output  = llm(
                prompt,
                max_tokens=64,
                echo=False,
            )
            t_end = time.perf_counter()

            generated_tokens = output["usage"]["completion_tokens"]
            elapsed_ms = (t_end - t_start) * 1000
            latency_per_tok = elapsed_ms / max(generated_tokens, 1)
            latencies.append(latency_per_tok)

            if not sample_out:
                sample_out = output["choices"][0]["text"].strip()

        stop_event.set()
        ram_thread.join()

        avg_latency   = sum(latencies) / len(latencies)
        tokens_per_sec = 1000 / avg_latency if avg_latency > 0 else 0
        peak_ram      = ram_results.get("peak_ram_mb", 0.0)

        return {
            "quant_type":       quant_type,
            "avg_latency_ms":   round(avg_latency, 2),
            "tokens_per_sec":   round(tokens_per_sec, 2),
            "peak_ram_mb":      round(peak_ram, 1),
            "total_ram_mb":     round(total_ram_mb, 1),
            "ram_utilization":  round(peak_ram / total_ram_mb, 3),
            "sample_output":    sample_out[:200],
            "status":           "success",
            "error":            None,
        }

    except MemoryError:
        stop_event.set()
        return {
            "quant_type": quant_type, "avg_latency_ms": 0,
            "tokens_per_sec": 0, "peak_ram_mb": total_ram_mb,
            "total_ram_mb": total_ram_mb, "ram_utilization": 1.0,
            "sample_output": "", "status": "oom",
            "error": "Out of memory — model too large for this device",
        }
    except Exception as e:
        stop_event.set()
        return {
            "quant_type": quant_type, "avg_latency_ms": 0,
            "tokens_per_sec": 0, "peak_ram_mb": 0,
            "total_ram_mb": total_ram_mb, "ram_utilization": 0,
            "sample_output": "", "status": "failed", "error": str(e),
        }


def submit_results(results: dict, iteration: int):
    payload = {**results, "iteration": iteration}
    print(f"[Pi] Submitting results: {json.dumps(payload, indent=2)}")
    r = requests.post(f"{SERVER_URL}/submit-results", json=payload, timeout=30)
    r.raise_for_status()
    print(f"[Pi] Results accepted by server.")


# ── Main Daemon Loop ──────────────────────────────────────────────
def main():
    print(f"[Pi] AutoEdgeQuant client started. Polling {SERVER_URL} ...")
    seen_iteration = -1

    while True:
        model_info = poll_for_model()

        if model_info and model_info["iteration"] != seen_iteration:
            seen_iteration = model_info["iteration"]
            print(f"[Pi] New model detected: {model_info['quant_type']} "
                  f"(iteration {seen_iteration})")

            try:
                model_path = download_model(
                    model_info["hf_repo_id"],
                    model_info["filename"],
                )
                results = run_inference(model_path, model_info["quant_type"])
                submit_results(results, seen_iteration)

                # Optionally clean up model to save disk space
                model_path.unlink()
                print(f"[Pi] Cleaned up {model_path.name}")

            except Exception as e:
                print(f"[Pi] ERROR during pipeline: {e}")
                submit_results({
                    "quant_type": model_info.get("quant_type", "unknown"),
                    "avg_latency_ms": 0, "tokens_per_sec": 0,
                    "peak_ram_mb": 0, "total_ram_mb": 0,
                    "ram_utilization": 0, "sample_output": "",
                    "status": "failed", "error": str(e),
                }, seen_iteration)
        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()