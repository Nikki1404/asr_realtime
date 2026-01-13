import numpy as np
from collections import deque

class AdaptiveEnergyVAD:
    def __init__(self, sr, frame_ms, margin, min_rms, pre_ms):
        self.sr = sr
        self.frame_ms = frame_ms
        self.margin = margin
        self.min_rms = min_rms
        self.frame_samples = int(sr * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.pre_frames = int(pre_ms / frame_ms)
        self.ring = deque(maxlen=self.pre_frames)
        self.noise = min_rms
        self.in_speech = False

    def rms(self, pcm):
        x = np.frombuffer(pcm, np.int16).astype("float32") / 32768
        return float(np.sqrt((x*x).mean() + 1e-12))

    def reset(self):
        self.ring.clear()
        self.in_speech = False

    def push(self, frame):
        e = self.rms(frame)
        if not self.in_speech:
            self.noise = max(self.min_rms, 0.95*self.noise + 0.05*e)
        speech = e > self.noise * self.margin
        self.ring.append(frame)
        pre = None
        if speech and not self.in_speech:
            self.in_speech = True
            pre = b"".join(self.ring)
        return speech, pre
