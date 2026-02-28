"""
Model Cache + Optional Preload
"""
import os
import sys
import logging
import subprocess

logger = logging.getLogger(__name__)


def configure_model_cache():
    cache_dir = os.getenv("MODEL_CACHE_DIR")
    if not cache_dir:
        return

    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    logger.info(f"[MODEL CACHE] Using cache directory: {cache_dir}")


def preload_models():
    if os.getenv("PRELOAD_MODELS", "false").lower() != "true":
        return

    logger.info("[MODEL PRELOAD] Preloading local models (this may take time).")
    try:
        from app.retrieval.embeddings import EmbeddingModel
        _ = EmbeddingModel().model
    except Exception as e:
        logger.error(f"[MODEL PRELOAD] Embeddings preload failed: {e}")
        raise RuntimeError(f"Failed to preload Embeddings: {e}")

    try:
        from app.retrieval.reranker import SemanticReranker
        _ = SemanticReranker().model
    except Exception as e:
        logger.error(f"[MODEL PRELOAD] Reranker preload failed: {e}")
        raise RuntimeError(f"Failed to preload Reranker: {e}")

    try:
        from app.multimodal.vision import VisionModel
        vision = VisionModel()
        _ = vision.processor
        _ = vision.model
    except Exception as e:
        logger.error(f"[MODEL PRELOAD] Vision preload failed: {e}")
        raise RuntimeError(f"Failed to preload Vision: {e}")

    try:
        from app.multimodal.tts import TextToSpeech
        _ = TextToSpeech().tts
    except Exception as e:
        logger.error(f"[MODEL PRELOAD] TTS preload failed: {e}")
        raise RuntimeError(f"Failed to preload TTS: {e}")


def ensure_paddle_runtime():
    """
    Optionally installs paddle runtime based on detected hardware.
    Controlled via env:
      PADDLE_AUTO_INSTALL=true
      PADDLE_PIP_SPEC_GPU or PADDLE_PIP_SPEC_CPU
      PADDLE_PIP_EXTRA_INDEX_URL (optional)
    """
    if os.getenv("PADDLE_AUTO_INSTALL", "false").lower() != "true":
        return

    try:
        import paddle  # noqa: F401
        return
    except Exception as e:
        last_error = str(e)

    def _detect_cuda_version() -> str | None:
        # Prefer torch if available
        try:
            import torch
            if torch.version.cuda:
                return str(torch.version.cuda)
        except Exception:
            pass
        # Fallback to nvidia-smi
        try:
            import subprocess as sp
            out = sp.check_output(["nvidia-smi"], stderr=sp.STDOUT, text=True)
            for line in out.splitlines():
                if "CUDA Version:" in line:
                    return line.split("CUDA Version:")[1].strip().split(" ")[0]
        except Exception:
            return None
        return None

    try:
        from app.infra.hardware import HardwareProbe
        profile = HardwareProbe.get_profile()
        use_gpu = profile.get("primary_device") == "cuda"
    except Exception:
        use_gpu = False

    pip_spec = os.getenv("PADDLE_PIP_SPEC_GPU" if use_gpu else "PADDLE_PIP_SPEC_CPU")
    extra_index = os.getenv("PADDLE_PIP_EXTRA_INDEX_URL")

    # If GPU runtime fails to import due to missing CUDA DLLs, fall back to CPU spec.
    if use_gpu and pip_spec and last_error:
        cuda_fail_signals = ["cusparse", "cudnn", "cufft", "cublas", "cudart", "nvrtc", "cufftw", "cuda"]
        if any(sig in last_error.lower() for sig in cuda_fail_signals):
            logger.warning(f"[PADDLE] GPU runtime import failed ({last_error}). Falling back to CPU runtime.")
            use_gpu = False
            pip_spec = os.getenv("PADDLE_PIP_SPEC_CPU")
            extra_index = None

    if use_gpu and not extra_index:
        # Auto-select a Paddle wheel index by detected CUDA version.
        # PaddleOCR docs publish cu118 and cu123 wheel indexes.
        cuda_ver = _detect_cuda_version()
        if cuda_ver and cuda_ver.startswith("11.8"):
            extra_index = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
        elif cuda_ver and cuda_ver.startswith("12.3"):
            extra_index = "https://www.paddlepaddle.org.cn/packages/stable/cu123/"
        elif cuda_ver and cuda_ver.startswith("12."):
            # Best-effort: use cu123 for CUDA 12.x unless overridden
            extra_index = "https://www.paddlepaddle.org.cn/packages/stable/cu123/"
            logger.warning(f"[PADDLE] CUDA {cuda_ver} detected. Using cu123 wheels by default. Override with PADDLE_PIP_EXTRA_INDEX_URL if needed.")

    if not pip_spec:
        logger.warning("[PADDLE] Auto-install enabled but PADDLE_PIP_SPEC_* not set.")
        return

    cmd = [sys.executable, "-m", "pip", "install", pip_spec]
    if extra_index:
        cmd.extend(["-i", extra_index])

    logger.info(f"[PADDLE] Auto-installing runtime: {pip_spec}")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        logger.warning(f"[PADDLE] Auto-install failed: {e}")
