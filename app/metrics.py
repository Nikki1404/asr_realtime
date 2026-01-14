from prometheus_client import Counter, Histogram, Gauge

ACTIVE_STREAMS = Gauge("asr_active_streams", "Active websocket streams")

PARTIALS_TOTAL = Counter("asr_partials_total", "Partial messages sent")
FINALS_TOTAL = Counter("asr_finals_total", "Final messages sent")
UTTERANCES_TOTAL = Counter("asr_utterances_total", "Utterances finalized")

TTFT_WALL = Histogram("asr_ttft_wall_sec", "Wall TTFT seconds")
TTF_WALL  = Histogram("asr_ttf_wall_sec", "Wall TTF seconds")

INFER_SEC = Histogram("asr_infer_sec", "Model inference seconds")
PREPROC_SEC = Histogram("asr_preproc_sec", "Model preproc seconds")
FLUSH_SEC = Histogram("asr_flush_sec", "Model flush seconds")

AUDIO_SEC = Histogram("asr_audio_sec", "Audio seconds per utterance")
RTF = Histogram("asr_rtf", "Real-time factor")

BACKLOG_MS = Gauge("asr_backlog_ms", "Buffered audio backlog (ms)")
