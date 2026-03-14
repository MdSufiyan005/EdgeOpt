# pi_client.py  — runs on RASPBERRY PI as a daemon
# pip install requests psutil llama-cpp-python huggingface_hub

import time
import json
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from gguf_runner import run_gguf          

# ── Config ────────────────────────────────────────────────────────
SERVER_URL    = "http://107.20.22.244:8000"
MODELS_DIR    = Path("/home/pi/models")
POLL_INTERVAL = 10    # seconds between polls

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
        repo_id   = hf_repo_id,
        filename  = filename,
        local_dir = str(MODELS_DIR),
    )
    print(f"[Pi] Downloaded to {local_path}")
    return Path(local_path)


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

                # ── run_gguf handles llama/qwen/mistral/phi/gemma etc. ──
                results = run_gguf(model_path, model_info["quant_type"])
                submit_results(results, seen_iteration)

                # Clean up to save disk space on Pi
                model_path.unlink()
                print(f"[Pi] Cleaned up {model_path.name}")

            except Exception as e:
                print(f"[Pi] ERROR during pipeline: {e}")
                submit_results({
                    "quant_type":      model_info.get("quant_type", "unknown"),
                    "model_family":    "unknown",
                    "avg_latency_ms":  0,
                    "tokens_per_sec":  0,
                    "peak_ram_mb":     0,
                    "total_ram_mb":    0,
                    "ram_utilization": 0,
                    "sample_output":   "",
                    "status":          "failed",
                    "error":           str(e),
                }, seen_iteration)

        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()