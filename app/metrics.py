from prometheus_client import Counter, Histogram, Gauge

ACTIVE_STREAMS = Gauge("asr_active_streams", "Active websocket streams")

PARTIALS_TOTAL = Counter("asr_partials_total", "Partial messages sent")
FINALS_TOTAL = Counter("asr_finals_total", "Final messages sent")
UTTERANCES_TOTAL = Counter("asr_utterances_total", "Utterances finalized")

TTFT_WALL = Histogram("asr_ttft_wall_sec", "Wall TTFT seconds")
TTF_WALL  = Histogram("asr_ttf_wall_sec", "Wall TTF seconds")

QUEUE_WAIT = Histogram("asr_queue_wait_sec", "GPU semaphore wait seconds")
PREPROC_SEC = Histogram("asr_preproc_sec", "Preprocessor seconds")
INFER_SEC   = Histogram("asr_infer_sec", "Streaming infer seconds")
FLUSH_SEC   = Histogram("asr_flush_sec", "Final flush seconds")

AUDIO_SEC = Histogram("asr_audio_sec", "Audio seconds per utterance")
RTF       = Histogram("asr_rtf", "Real-time factor")

BACKLOG_MS = Gauge("asr_backlog_ms", "Buffered audio backlog (ms)")

GPU_UTIL = Gauge("asr_gpu_util", "GPU utilization percent")
GPU_MEM_USED_MB = Gauge("asr_gpu_mem_used_mb", "GPU memory used MB")
GPU_MEM_TOTAL_MB = Gauge("asr_gpu_mem_total_mb", "GPU memory total MB")
