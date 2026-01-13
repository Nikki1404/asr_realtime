import time
import tempfile
import logging
import numpy as np
import soundfile as sf
import torch

log = logging.getLogger("asr_server")

class NemoASR:
    """
    Loads: nvidia/nemotron-speech-streaming-en-0.6b
    via NeMo ASRModel.from_pretrained(), which pulls the .nemo from Hugging Face into HF cache.
    """
    def __init__(self, model_name: str, device: str, sample_rate: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.model = None

    def load(self):
        from nemo.collections.asr.models import ASRModel

        t0 = time.time()
        self.model = ASRModel.from_pretrained(model_name=self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        log.info(f"Loaded model={self.model_name} in {(time.time()-t0):.2f}s on {self.device}")

        # warmup (reduces first-run latency)
        self._warmup()

    @torch.no_grad()
    def _warmup(self):
        x = np.zeros(int(self.sr * 0.6), dtype=np.float32)
        _ = self.transcribe_float32(x)

    def pcm16_to_float32(self, pcm16: bytes) -> np.ndarray:
        return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    @torch.no_grad()
    def transcribe_float32(self, audio_f32: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        audio_f32 = np.asarray(audio_f32, dtype=np.float32).reshape(-1)

        # write temp wav (most compatible with NeMo models)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_f32, self.sr, subtype="PCM_16")
            out = self.model.transcribe([tmp.name])
        if not out:
            return ""
        return (out[0] or "").strip()
