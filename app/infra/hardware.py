"""
Hardware Optimization & Diagnostic Service

This module is called on startup to actively map the local machine's architecture.
It ensures the Agentic backend doesn't trigger 50 concurrent crawler threads on a 2-core laptop, 
crashing the OS memory violently.
"""
import os
import logging
import multiprocessing
import sys

logger = logging.getLogger(__name__)

class HardwareProbe:
    """
    Examines the physical execution hardware (CPU vs CUDA vs Apple MPS).
    Determines logical scaling multipliers dynamically so the enterprise architecture 
    works natively whether deployed on a 16-Core AWS Server or a cheap student Laptop.
    """
    
    @staticmethod
    def detect_environment() -> dict:
        """
        Pipes the computational status of the Python Host environment.
        
        Returns:
            dict: The normalized mappings determining exact Queue worker limits and Inference targets.
        """
        try:
            # CPU Extraction
            cpu_cores = multiprocessing.cpu_count() or 4
            recommended_async_workers = (cpu_cores * 2) + 1

            # GPU Extraction (PyTorch hooks)
            has_gpu = False
            device = "cpu"
            gpu_name = "None"
            gpu_mem_gb = 0.0

            force_device = os.getenv("FORCE_DEVICE")
            if force_device:
                device = force_device
                has_gpu = device in ["cuda", "mps"]
                gpu_name = "Forced"
            else:
                try:
                    # Ensure torch DLLs are discoverable on Windows
                    if sys.platform == "win32":
                        venv_root = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))
                        torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
                        if os.path.isdir(torch_lib):
                            try:
                                os.add_dll_directory(torch_lib)
                            except Exception:
                                pass
                        # Also add NVIDIA runtime DLLs (used by Paddle/CUDA stacks)
                        nvidia_root = os.path.join(venv_root, "Lib", "site-packages", "nvidia")
                        if os.path.isdir(nvidia_root):
                            try:
                                for entry in os.listdir(nvidia_root):
                                    dll_dir = os.path.join(nvidia_root, entry, "bin")
                                    if os.path.isdir(dll_dir):
                                        try:
                                            os.add_dll_directory(dll_dir)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    import torch
                    if torch.cuda.is_available():
                        has_gpu = True
                        device = "cuda"
                        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown CUDA"
                        try:
                            props = torch.cuda.get_device_properties(0)
                            gpu_mem_gb = round(props.total_memory / (1024 ** 3), 2)
                        except Exception:
                            gpu_mem_gb = 0.0
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        has_gpu = True
                        device = "mps"
                        gpu_name = "Apple Silicon"
                except ImportError:
                    logger.warning("PyTorch not installed. Falling back strictly to CPU mapping.")
                except OSError as e:
                    logger.error(f"PyTorch failed to load native libs: {e}. Falling back to CPU mapping.")

            # Heuristic batch sizes
            if device == "cuda":
                embedding_batch_size = 128 if gpu_mem_gb >= 8 else 64
                rerank_batch_size = 16 if gpu_mem_gb < 8 else 32
            elif device == "mps":
                embedding_batch_size = 64
                rerank_batch_size = 8
            else:
                embedding_batch_size = 32
                rerank_batch_size = 4

            # Overrides
            if os.getenv("EMBEDDING_BATCH_SIZE"):
                embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE"))
            if os.getenv("RERANK_BATCH_SIZE"):
                rerank_batch_size = int(os.getenv("RERANK_BATCH_SIZE"))

            crawler_workers = min(max(2, cpu_cores), 8)
            if os.getenv("CRAWLER_WORKERS"):
                crawler_workers = int(os.getenv("CRAWLER_WORKERS"))

            config = {
                "cpu_cores": cpu_cores,
                "recommended_async_workers": recommended_async_workers,
                "has_gpu": has_gpu,
                "primary_device": device,
                "gpu_name": gpu_name,
                "gpu_mem_gb": gpu_mem_gb,
                "embedding_batch_size": embedding_batch_size,
                "rerank_batch_size": rerank_batch_size,
                "crawler_workers": crawler_workers
            }

            logger.info(f"[HARDWARE PROBE] Initialized scaling config: {config}")
            return config

        except Exception as e:
            logger.error(f"[HARDWARE PROBE] Catastrophic failure during environment scan: {e}")
            return {
                "cpu_cores": 2,
                "recommended_async_workers": 4,
                "has_gpu": False,
                "primary_device": "cpu",
                "gpu_name": "None",
                "gpu_mem_gb": 0.0,
                "embedding_batch_size": 32,
                "rerank_batch_size": 4,
                "crawler_workers": 2
            }

    _cached_profile = None

    @classmethod
    def get_profile(cls) -> dict:
        if cls._cached_profile is None:
            cls._cached_profile = cls.detect_environment()
        return cls._cached_profile.copy()
