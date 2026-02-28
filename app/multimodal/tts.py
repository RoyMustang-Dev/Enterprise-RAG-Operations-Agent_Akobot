"""
Text-to-Speech (Coqui TTS)

Free, local TTS using Coqui models. Auto-selects GPU if available.
"""
import os
import tempfile
import logging

from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC"
        )
        profile = HardwareProbe.get_profile()
        self.use_gpu = profile.get("primary_device") == "cuda"
        self._tts = None
        logger.info(f"[TTS] Configured model={self.model_name} gpu={self.use_gpu}")

    @property
    def tts(self):
        if self._tts is None:
            try:
                # Monkeypatch transformers for Coqui TTS compatibility
                import transformers
                if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
                    import torch
                    def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor, assume_unique: bool = False, invert: bool = False) -> torch.Tensor:
                        return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
                    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly

                from TTS.api import TTS
            except Exception as e:
                raise RuntimeError(f"Coqui TTS not installed or failed to import: {e}")

            device = "cuda" if self.use_gpu else "cpu"
            self._tts = TTS(model_name=self.model_name).to(device)
        return self._tts

    def generate_audio(self, text: str, file_path: str = None) -> str:
        if not text:
            raise ValueError("TTS text cannot be empty.")

        if file_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            file_path = tmp.name
            tmp.close()

        self.tts.tts_to_file(text=text, file_path=file_path)
        return file_path
