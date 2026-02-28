"""
Vision Model (LLaVA or BLIP)

Semantic image understanding with configurable backend.
Default is BLIP for low-VRAM environments; LLaVA can be enabled via env.
"""
import os
import io
import logging

from PIL import Image

from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)


class VisionModel:
    def __init__(self, model_name: str = None):
        self.backend = (os.getenv("VISION_BACKEND", "blip")).lower()
        self.model_name = model_name or os.getenv(
            "VISION_MODEL_NAME",
            "Salesforce/blip-image-captioning-base" if self.backend == "blip" else "llava-hf/llava-1.5-7b-hf",
        )
        self.fallback_model = os.getenv(
            "VISION_FALLBACK_MODEL", "Salesforce/blip-image-captioning-base"
        )
        self.allow_fallback = os.getenv("VISION_ALLOW_FALLBACK", "true").lower() == "true"
        profile = HardwareProbe.get_profile()
        self.device = profile.get("primary_device", "cpu")
        self._processor = None
        self._model = None
        self._fallback = None
        logger.info(f"[VISION] Configured backend={self.backend} model={self.model_name} device={self.device}")

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

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = question or "Describe the image."
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
