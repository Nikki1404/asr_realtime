import numpy as np, tempfile, soundfile as sf, torch
from nemo.collections.asr.models import ASRModel

class NemoASR:
    def __init__(self, model_name, device, sr):
        self.sr = sr
        self.model = ASRModel.from_pretrained(model_name).to(device).eval()
        self.warmup()

    def warmup(self):
        x = np.zeros(int(self.sr * 0.5), dtype="float32")
        self.transcribe(x)

    def transcribe(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, self.sr, subtype="PCM_16")
            out = self.model.transcribe([f.name])
        return (out[0] or "").strip()

    def pcm16_to_f32(self, pcm):
        return np.frombuffer(pcm, np.int16).astype("float32") / 32768
