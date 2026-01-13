from collections import deque
import numpy as np

class AdaptiveEnergyVAD:
    def __init__(self, sample_rate: int, frame_ms: int, start_margin: float, min_noise_rms: float, pre_speech_ms: int):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.start_margin = start_margin
        self.min_noise_rms = min_noise_rms

        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

        self.pre_frames = max(1, int(pre_speech_ms / frame_ms))
        self.ring = deque(maxlen=self.pre_frames)

        self.in_speech = False
        self.noise_rms = min_noise_rms

    def reset(self):
        self.ring.clear()
        self.in_speech = False

    def _rms(self, pcm16: bytes) -> float:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)

        if not self.in_speech:
            alpha = 0.95
            self.noise_rms = max(self.min_noise_rms, alpha * self.noise_rms + (1 - alpha) * e)

        th = max(self.min_noise_rms, self.noise_rms) * self.start_margin
        is_speech = e >= th

        self.ring.append(frame_pcm16)

        pre_roll = None
        if (not self.in_speech) and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)

        if self.in_speech and (not is_speech):
            # we do not flip off here; endpointing decides finalization
            pass

        return is_speech, pre_roll
