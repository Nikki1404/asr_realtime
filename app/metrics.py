from prometheus_client import Counter, Histogram, Gauge

ACTIVE_STREAMS = Gauge("asr_active_streams", "Active streams")
PARTIALS_TOTAL = Counter("asr_partials_total", "Partials")
FINALS_TOTAL = Counter("asr_finals_total", "Finals")
UTTERANCES_TOTAL = Counter("asr_utterances_total", "Utterances")

TTFT_WALL = Histogram("asr_ttft_wall_sec", "TTFT")
TTF_WALL = Histogram("asr_ttf_wall_sec", "TTF")
INFER_SEC = Histogram("asr_infer_sec", "Infer time")
QUEUE_WAIT = Histogram("asr_queue_wait_sec", "GPU wait")

AUDIO_SEC = Histogram("asr_audio_sec", "Audio seconds")
RTF = Histogram("asr_rtf", "RTF")

BACKLOG_MS = Gauge("asr_backlog_ms", "Backlog ms")

GPU_UTIL = Gauge("asr_gpu_util", "GPU util")
GPU_MEM_USED_MB = Gauge("asr_gpu_mem_used_mb", "GPU mem used")
GPU_MEM_TOTAL_MB = Gauge("asr_gpu_mem_total_mb", "GPU mem total")
