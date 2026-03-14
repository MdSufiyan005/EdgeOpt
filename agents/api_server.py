# # api_server.py  — runs on your MAIN MACHINE
# # pip install fastapi uvicorn asyncio

# import asyncio
# from datetime import datetime
# from typing import Optional
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# import uvicorn

# app = FastAPI(title="AutoEdgeQuant Bridge API")

# # ────────────────────────────────────
# # ─────────
# # In-memory store (swap for Redis in production)
# # ─────────────────────────────────────────────
# _state = {
#     "pending_model":  None,   # set by system after HF upload
#     "pi_results":     None,   # set by Pi after inference
#     "result_event":   asyncio.Event(),  # graph waits on this
# }

# # ─────────────────────────────────────────────
# # Schemas
# # ─────────────────────────────────────────────
# class ModelNotification(BaseModel):
#     hf_repo_id:   str          # e.g. "youruser/llama-q4km"
#     quant_type:   str          # e.g. "Q4_K_M"
#     filename:     str          # e.g. "llama-q4km.gguf"
#     model_size_mb: float
#     iteration:    int

# class PiResults(BaseModel):
#     quant_type:        str
#     iteration:         int
#     avg_latency_ms:    float   # ms per token
#     tokens_per_sec:    float
#     peak_ram_mb:       float
#     total_ram_mb:      float
#     ram_utilization:   float   # 0.0 - 1.0
#     perplexity:        Optional[float] = None
#     sample_output:     str
#     status:            str     # "success" | "oom" | "failed"
#     error:             Optional[str] = None

# # ─────────────────────────────────────────────
# # Endpoints
# # ─────────────────────────────────────────────

# # Called by YOUR SYSTEM after upload_model_hf() succeeds
# @app.post("/notify-model")
# async def notify_model(payload: ModelNotification):
#     """System notifies Pi that a new model is ready on HuggingFace."""
#     _state["pending_model"] = payload.dict()
#     _state["pi_results"] = None
#     _state["result_event"].clear()   # reset so graph blocks on next wait
#     print(f"[API] Model ready: {payload.quant_type} @ {payload.hf_repo_id}")
#     return {"status": "notified", "model": payload.filename}


# # Called by PI to poll for a new model to download
# @app.get("/poll-model")
# async def poll_model():
#     """Pi polls this to check if a new model is waiting for it."""
#     if _state["pending_model"] is None:
#         return {"model_ready": False}
#     return {"model_ready": True, **_state["pending_model"]}


# # Called by PI after it finishes inference
# @app.post("/submit-results")
# async def submit_results(results: PiResults):
#     """Pi submits inference metrics after running the quantized model."""
#     _state["pi_results"] = results.dict()
#     _state["result_event"].set()    # unblocks the waiting graph
#     print(f"[API] Results received: {results.avg_latency_ms}ms/tok, "
#           f"{results.tokens_per_sec} tok/s, {results.peak_ram_mb}MB RAM")
#     return {"status": "received"}


# # Called by YOUR GRAPH to block until Pi results arrive
# @app.get("/wait-results")
# async def wait_results(timeout: int = 300):
#     """Graph calls this to block until Pi has submitted results."""
#     try:
#         await asyncio.wait_for(_state["result_event"].wait(), timeout=timeout)
#         return {"status": "ready", "results": _state["pi_results"]}
#     except asyncio.TimeoutError:
#         raise HTTPException(
#             status_code=408,
#             detail=f"Pi did not respond within {timeout}s. Check Pi connection."
#         )


# @app.get("/health")
# async def health():
#     return {
#         "status": "ok",
#         "pending_model":  _state["pending_model"] is not None,
#         "results_ready":  _state["pi_results"] is not None,
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)