from dataclasses import dataclass
import os

@dataclass
class Config:
    # Model
    model_name: str = os.getenv("MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
    device: str = os.getenv("DEVICE", "cuda")  # cuda/cpu
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # VAD (energy-based, adaptive noise floor)
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))  # speech if rms > noise * margin
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    # Endpointing
    end_silence_ms: int = int(os.getenv("END_SILENCE_MS", "1400"))
    min_utt_ms: int = int(os.getenv("MIN_UTT_MS", "900"))
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
    post_speech_pad_ms: int = int(os.getenv("POST_SPEECH_PAD_MS", "400"))

    # Partial updates (sliding window)
    partial_every_ms: int = int(os.getenv("PARTIAL_EVERY_MS", "700"))
    partial_window_ms: int = int(os.getenv("PARTIAL_WINDOW_MS", "7000"))

    # Concurrency
    max_concurrent_inferences: int = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))

    # GPU metrics
    enable_gpu_metrics: bool = os.getenv("ENABLE_GPU_METRICS", "1") == "1"
    gpu_index: int = int(os.getenv("GPU_INDEX", "0"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

def load_config() -> Config:
    return Config()
