"""
Vision Model (LLaVA or BLIP)

Semantic image understanding with configurable backend.
Default is BLIP for low-VRAM environments; LLaVA can be enabled via env.
"""
import os
import io
import logging
import base64
import requests

from PIL import Image

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from app.infra.hardware import HardwareProbe

# Reduce HF/transformers console noise
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger(__name__)


class VisionModel:
    def __init__(self, model_name: str = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.vision_cloud_model = os.getenv("VISION_CLOUD_MODEL", "gemini-2.5-flash")
        self.backend = (os.getenv("VISION_BACKEND", "auto")).lower()
        requested_backend = self.backend
        self.fallback_model = os.getenv(
            "VISION_FALLBACK_MODEL", "Salesforce/blip-image-captioning-base"
        )
        self.allow_fallback = os.getenv("VISION_ALLOW_FALLBACK", "true").lower() == "true"
        profile = HardwareProbe.get_profile()
        self.device = profile.get("primary_device", "cpu")
        gpu_mem_gb = profile.get("gpu_mem_gb", 0.0) or 0.0
        min_vram = float(os.getenv("VISION_LLAVA_MIN_VRAM_GB", "8"))

        # Auto-select backend based on VRAM availability
        if self.backend == "auto":
            if self.device == "cuda" and gpu_mem_gb >= min_vram:
                self.backend = "llava"
            else:
                self.backend = "blip"

        if self.backend == "llava" and (self.device != "cuda" or gpu_mem_gb < min_vram):
            logger.warning(
                f"[VISION] Requested LLaVA but GPU VRAM {gpu_mem_gb}GB < {min_vram}GB. "
                "Falling back to BLIP."
            )
            self.backend = "blip"

        self.model_name = model_name or os.getenv(
            "VISION_MODEL_NAME",
            "Salesforce/blip-image-captioning-base" if self.backend == "blip" else "llava-hf/llava-1.5-7b-hf",
        )
        self._processor = None
        self._model = None
        self._fallback = None
        logger.info(
            f"[VISION] Configured backend={self.backend} (requested={requested_backend}) "
            f"model={self.model_name} device={self.device}"
        )
        if self.gemini_key:
            logger.info(f"[VISION] Gemini vision enabled model_id={self.vision_cloud_model}")

    @property
    def processor(self):
        if self._processor is None:
            try:
                from transformers import AutoProcessor, BlipProcessor
            except Exception as e:
                raise RuntimeError(f"transformers not installed or failed to import: {e}")
            if self.backend == "blip":
                self._processor = BlipProcessor.from_pretrained(self.model_name)
            else:
                self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            try:
                import torch
                from transformers import LlavaForConditionalGeneration, BlipForConditionalGeneration
            except Exception as e:
                raise RuntimeError(f"transformers/torch not installed or failed to import: {e}")

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                if self.backend == "blip":
                    self._model = BlipForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                    )
                    self._model.to(self.device)
                else:
                    if self.device == "cuda":
                        self._model = LlavaForConditionalGeneration.from_pretrained(
                            self.model_name,
                            torch_dtype=dtype,
                            device_map="auto"
                        )
                    else:
                        self._model = LlavaForConditionalGeneration.from_pretrained(
                            self.model_name,
                            torch_dtype=dtype
                        )
                        self._model.to(self.device)
            except Exception as e:
                if not self.allow_fallback:
                    raise
                logger.warning(f"[VISION] Primary model load failed, falling back: {e}")
                self._model = None
        return self._model

    def answer(self, image_bytes: bytes, question: str = None, max_new_tokens: int = 256) -> str:
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty.")

        prompt = question or "Describe the image."

                # Prefer Gemini vision when key exists (fallback to local models if it fails).
        if self.gemini_key:
            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = "image/jpeg"
                if image_bytes.startswith(b"\x89PNG"):
                    mime_type = "image/png"
                elif image_bytes.startswith(b"RIFF"):
                    mime_type = "image/webp"

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": b64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": max_new_tokens,
                        "temperature": 0.1
                    }
                }
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.vision_cloud_model}:generateContent?key={self.gemini_key}",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                try:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info("[VISION] Gemini vision success; returning remote response.")
                    return content.strip()
                except Exception as parse_e:
                    logger.warning(f"[VISION] Gemini vision parse error: {parse_e} raw={result}")
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"Gemini vision failed: {e}")
                logger.warning(f"[VISION] Gemini vision failed, falling back to local: {e}")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        # Primary LLaVA path
        if self.model is not None:
            if self.backend == "blip":
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
            try:
                import torch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                result = self.processor.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"Vision inference failed: {e}")
                logger.warning(f"[VISION] Primary inference failed, falling back: {e}")
                result = None
        else:
            result = None

        # Fallback model
        if result is None and self.allow_fallback:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                if self._fallback is None:
                    self._fallback = {
                        "processor": BlipProcessor.from_pretrained(self.fallback_model),
                        "model": BlipForConditionalGeneration.from_pretrained(self.fallback_model)
                    }
                inputs = self._fallback["processor"](image, return_tensors="pt")
                output = self._fallback["model"].generate(**inputs, max_new_tokens=max_new_tokens)
                result = self._fallback["processor"].decode(output[0], skip_special_tokens=True)
            except Exception as e:
                raise RuntimeError(f"Vision inference failed (fallback): {e}")

        # Return only the assistant portion if present
        if "ASSISTANT:" in result:
            return result.split("ASSISTANT:")[-1].strip()
        return result.strip()
