import os
import subprocess
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_file
from langchain.tools import tool

from logging_ import adding_logs


MODELS_HF_DIR   = Path(__file__).parent.parent / "models" / "hf"
MODELS_GGUF_DIR = Path(__file__).parent.parent / "models" / "gguf"

logger = adding_logs("tools")

# TOOL 1: Convert HuggingFace model → GGUF F16

@tool
def convert_hf_to_gguf(hf_model_path: str, output_path: str) -> dict:
    """
    Converts a HuggingFace model to GGUF F16 format using llama.cpp's convert script.
    Must be run before quantization. Output is a float16 GGUF baseline file.

    Args:
        hf_model_path: Path to the local HuggingFace model directory (must contain config.json)
        output_path: Desired path for the output .gguf file (e.g. ./models/model-f16.gguf)

    Returns:
        dict with keys: success (bool), output_model (str), stdout (str), stderr (str), error (str|None)
    """
    logger.info(f"Starting conversion of HF model from {hf_model_path} to GGUF at {output_path}")
    
    model_dir = Path(hf_model_path)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {hf_model_path}")
        return {
            "success": False, "output_model": None,
            "stdout": "", "stderr": "",
            "error": f"Model path does not exist: {hf_model_path}"
        }
    if not (model_dir / "config.json").exists():
        logger.error(f"No config.json found in {hf_model_path}. Is this a valid HF model dir?")
        return {
            "success": False, "output_model": None,
            "stdout": "", "stderr": "",
            "error": f"No config.json found in {hf_model_path}. Is this a valid HF model dir?"
        }

    convert_script = Path("external/llama.cpp/convert_hf_to_gguf_update.py")
    if not convert_script.exists():
        logger.error("GGUF converter script not found: external/llama.cpp/convert_hf_to_gguf_update.py")
        return {
            "success": False, "output_model": None,
            "stdout": "", "stderr": "",
            "error": "convert_hf_to_gguf.py not found. Clone llama.cpp and run from its root."
        }

    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Executing conversion command")
    cmd = [
        "python", str(convert_script),
        str(hf_model_path),
        "--outfile", str(output_path),
        "--outtype", "f16",        # explicit: always produce F16 baseline
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,           # 10 min cap — large models can be slow
        )
        logger.info(f"Return code: {result.returncode}")
        logger.debug(result.stdout)

    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out after 600 seconds")
        return {
            "success": False, "output_model": None,
            "stdout": "", "stderr": "",
            "error": "Conversion timed out after 600 seconds."
        }
    except FileNotFoundError:
        logger.error("Python interpreter not found. Check your environment.")
        return {
            "success": False, "output_model": None,
            "stdout": "", "stderr": "",
            "error": "Python interpreter not found. Check your environment."
        }

    if result.returncode == 0 and not output_file.exists():
        logger.warning("Process exited 0 but output file was not created")
        return {
            "success": False, "output_model": None,
            "stdout": result.stdout, "stderr": result.stderr,
            "error": "Process exited 0 but output file was not created."
        }

    if result.returncode == 0:
        logger.info(f"Conversion completed successfully, output: {output_path}")

    return {
        "success": result.returncode == 0,
        "output_model": str(output_path) if result.returncode == 0 else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.stderr if result.returncode != 0 else None,
    }



# TOOL 2: Quantize GGUF F16 → target quant type

VALID_QUANT_TYPES = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S",
                     "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16"}

@tool
def quantize_gguf(input_model: str, output_model: str, quant_type: str) -> dict:
    """
    Quantizes a GGUF F16 model to a smaller quant type using llama-quantize.
    llama-quantize must be compiled and present in the current directory or PATH.

    Args:
        input_model: Path to the source F16 .gguf file
        output_model: Path for the quantized output .gguf file
        quant_type: One of Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0

    Returns:
        dict with keys: success (bool), output_model (str), quant_type (str),
                        model_size_mb (float|None), stdout (str), stderr (str), error (str|None)
    """
    if quant_type.upper() not in VALID_QUANT_TYPES:
        return {
            "success": False, "output_model": None, "quant_type": quant_type,
            "model_size_mb": None, "stdout": "", "stderr": "",
            "error": f"Invalid quant_type '{quant_type}'. Valid options: {sorted(VALID_QUANT_TYPES)}"
        }

    
    input_file = Path(input_model)
    if not input_file.exists():
        logger.error(f"Input model not found: {input_model}")
        return {
            "success": False, "output_model": None, "quant_type": quant_type,
            "model_size_mb": None, "stdout": "", "stderr": "",
            "error": f"Input model not found: {input_model}"
        }
    if input_file.suffix != ".gguf":
        logger.error(f"Input must be a .gguf file, got: {input_file.suffix}")
        return {
            "success": False, "output_model": None, "quant_type": quant_type,
            "model_size_mb": None, "stdout": "", "stderr": "",
            "error": f"Input must be a .gguf file, got: {input_file.suffix}"
        }

    quantize_bin = Path("external/llama.cpp/build/bin/llama-quantize")
    if not quantize_bin.exists():
        # fallback: check PATH
        import shutil
        if not shutil.which("llama-quantize"):
            logger.error("llama-quantize binary not found. Build llama.cpp first: `make llama-quantize`")
            return {
                "success": False, "output_model": None, "quant_type": quant_type,
                "model_size_mb": None, "stdout": "", "stderr": "",
                "error": "llama-quantize binary not found. Build llama.cpp first: `make llama-quantize`"
            }
        quantize_bin = "llama-quantize"

    
    output_file = Path(output_model)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Executing quantization command")
    cmd = [str(quantize_bin), str(input_model), str(output_model), quant_type.upper()]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,           # 15 min cap
        )
    except subprocess.TimeoutExpired:
        logger.error("Quantization timed out after 900 seconds")
        return {
            "success": False, "output_model": None, "quant_type": quant_type,
            "model_size_mb": None, "stdout": "", "stderr": "",
            "error": "Quantization timed out after 900 seconds."
        }

    model_size_mb = None
    if result.returncode == 0:
        if not output_file.exists():
            logger.warning("Process exited 0 but output file was not created")
            return {
                "success": False, "output_model": None, "quant_type": quant_type,
                "model_size_mb": None,
                "stdout": result.stdout, "stderr": result.stderr,
                "error": "Process exited 0 but output file was not created."
            }
        model_size_mb = round(output_file.stat().st_size / (1024 ** 2), 2)
        logger.info(f"Quantization completed successfully, output: {output_model}, size: {model_size_mb} MB")

    return {
        "success": result.returncode == 0,
        "output_model": str(output_model) if result.returncode == 0 else None,
        "quant_type": quant_type.upper(),
        "model_size_mb": model_size_mb,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.stderr.strip() if result.returncode != 0 else None,
    }


# TOOL 3: Upload quantized GGUF to HuggingFace
@tool
def upload_model_hf(
    model_path: str,
    repo_id: str,
    quant_type: str,
    private: bool = False,
    hf_token: Optional[str] = None,
) -> dict:
    """
    Uploads a quantized GGUF model file to a HuggingFace repository.
    Creates the repo if it does not exist. Requires HF_TOKEN env var or hf_token arg.

    Args:
        model_path: Local path to the quantized .gguf file
        repo_id: HuggingFace repo in format 'username/repo-name'
        quant_type: Quant type string used in the file name (e.g. Q4_K_M)
        private: Whether to create the repo as private (default False)
        hf_token: Optional HF token (falls back to HF_TOKEN env var)

    Returns:
        dict with keys: success (bool), hf_repo_url (str|None), filename (str), error (str|None)
    """
    logger.info(f"Starting upload of {model_path} to HF repo {repo_id}")
    
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("No HuggingFace token found. Set HF_TOKEN env var or pass hf_token.")
        return {
            "success": False, "hf_repo_url": None, "filename": None,
            "error": "No HuggingFace token found. Set HF_TOKEN env var or pass hf_token."
        }

    local_file = Path(model_path)
    if not local_file.exists():
        logger.error(f"Model file not found: {model_path}")
        return {
            "success": False, "hf_repo_url": None, "filename": None,
            "error": f"Model file not found: {model_path}"
        }
    if local_file.suffix != ".gguf":
        logger.error(f"Expected a .gguf file, got: {local_file.suffix}")
        return {
            "success": False, "hf_repo_url": None, "filename": None,
            "error": f"Expected a .gguf file, got: {local_file.suffix}"
        }

    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        logger.error(f"repo_id must be 'username/repo-name', got: '{repo_id}'")
        return {
            "success": False, "hf_repo_url": None, "filename": None,
            "error": f"repo_id must be 'username/repo-name', got: '{repo_id}'"
        }

    api = HfApi(token=token)

    logger.info(f"Creating/accessing HF repo: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,        
            token=token,
        )
    except Exception as e:
        logger.error(f"Failed to create/access repo '{repo_id}': {e}")
        return {
            "success": False, "hf_repo_url": None, "filename": None,
            "error": f"Failed to create/access repo '{repo_id}': {e}"
        }

    # Upload file 
    # Use a descriptive filename in the repo: <original_stem>-<quant>.gguf
    remote_filename = f"{local_file.stem}-{quant_type.upper()}.gguf"

    logger.info(f"Uploading file to HF as: {remote_filename}")
    try:
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {quant_type.upper()} quantized GGUF via agentic pipeline",
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {
            "success": False, "hf_repo_url": None, "filename": remote_filename,
            "error": f"Upload failed: {e}"
        }

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Upload successful to {repo_url}")
    return {
        "success": True,
        "hf_repo_url": repo_url,
        "filename": remote_filename,
        "error": None,
    }

@tool
def download_hf_model(repo_id: str, local_dir: str) -> dict:
    """
    Downloads a HuggingFace model to a local directory using snapshot_download.

    Args:
        repo_id: HuggingFace repo ID e.g. 'Qwen/Qwen3-0.6B'
        local_dir: Local path to save the model

    Returns:
        dict with keys: success (bool), model_path (str), error (str|None)
    """
    logger.info(f"Starting download of HF model {repo_id} to {local_dir}")
    
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        logger.info(f"Download completed successfully to {path}")
        return {"success": True, "model_path": path, "error": None}
        
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return {"success": False, "model_path": None, "error": str(e)}