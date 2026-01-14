# Realtime ASR with NVIDIA Nemotron (Streaming)

This project implements **true realtime speech-to-text** using **NVIDIA NeMo Nemotron streaming ASR** with:

- Live microphone input 
- Low-latency partial transcriptions
- Accurate final transcripts on pause
- Built-in **latency & performance metrics**
- WebSocket-based streaming API

---

##  Features

- Realtime partial transcripts
- Automatic finalization on silence
- Metrics per utterance:
  - TTFT (time to first token)
  - TTF (time to final)
  - RTF (real-time factor)
  - Preprocess / inference / flush time
  - Chunk count
- Prometheus metrics endpoint
- GPU-accelerated (CUDA)

---

##  Model Used

- `nvidia/nemotron-speech-streaming-en-0.6b`
- Loaded via **Hugging Face** using NeMo

---

##  How to Run

### Start server (Docker)

```bash
docker build -t cx_asr_realtime .
docker run --gpus all -p 8003:8003 cx_asr_realtime
```

Endpoints:
- WebSocket ASR: `ws://localhost:8003/ws/asr`
- Metrics: `http://localhost:8003/metrics`

---

### Run client (microphone)

```bash
python scripts/ws_client.py --mic --url ws://127.0.0.1:8003/ws/asr
```
### Client WAV

```bash
python scripts/ws_client.py --wav sample.wav --url ws://127.0.0.1:8002/ws/asr
```

You will see:
- `[PARTIAL]` updates while speaking
- `[FINAL]` transcript after pause
- `[SERVER_METRICS]` printed immediately after finalization

---

##  Example Metrics Output

```
[FINAL] Hello, this is a realtime ASR test
[SERVER_METRICS]
reason=pause
ttft_ms=517
ttf_ms=7635
audio_ms=7600
rtf=0.25
chunks=26
preproc_ms=466
infer_ms=1920
flush_ms=40
```

### Interpretation

- First text appears in ~0.5s
- Final transcript emitted ~35ms after you stop speaking
- Model runs ~4× faster than real time

---

##  Key Configuration

```bash
CONTEXT_RIGHT=1        # latency vs accuracy tradeoff
END_SILENCE_MS=900    # pause duration to finalize
FINALIZE_PAD_MS=400   # padding to flush last words
MAX_BUFFER_MS=12000
```

---

This setup provides **production-grade realtime ASR with accurate metrics**.
