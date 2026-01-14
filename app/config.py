from dataclasses import dataclass
import os

@dataclass
class Config:
    # Model
    model_name: str = os.getenv("MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
    device: str = os.getenv("DEVICE", "cuda")  # cuda/cpu
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # Streaming latency/accuracy knob (right context)
    # 0 = ultra low latency, 1 = recommended, 6/13 = higher accuracy but more latency
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))

    # VAD + endpointing
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    # Endpointing (pause triggers final)
    end_silence_ms: int = int(os.getenv("END_SILENCE_MS", "900"))
    min_utt_ms: int = int(os.getenv("MIN_UTT_MS", "250"))
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    # When finalizing, we add extra zero padding (ms) to flush last words
    finalize_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))

    # Keep ring-buffer bounded (avoid slowdowns on long sessions)
    max_buffer_ms: int = int(os.getenv("MAX_BUFFER_MS", "12000"))

    # Concurrency
    max_concurrent_inferences: int = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))

    # GPU metrics (optional)
    enable_gpu_metrics: bool = os.getenv("ENABLE_GPU_METRICS", "0") == "1"
    gpu_index: int = int(os.getenv("GPU_INDEX", "0"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

def load_config() -> Config:
    return Config()
