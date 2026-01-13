from dataclasses import dataclass
import os

@dataclass
class Config:
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    device: str = "cuda"
    sample_rate: int = 16000

    vad_frame_ms: int = 20
    vad_start_margin: float = 2.5
    vad_min_noise_rms: float = 0.003
    pre_speech_ms: int = 300

    end_silence_ms: int = 1400
    min_utt_ms: int = 900
    max_utt_ms: int = 30000
    post_speech_pad_ms: int = 400

    partial_every_ms: int = 700
    partial_window_ms: int = 7000

    max_concurrent_inferences: int = 1

    enable_gpu_metrics: bool = True
    gpu_index: int = 0

    log_level: str = "INFO"

def load_config():
    return Config()
