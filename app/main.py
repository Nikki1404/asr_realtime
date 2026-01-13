import asyncio
import json
import logging
import time
from collections import deque

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config
from app.metrics import (
    ACTIVE_STREAMS, PARTIALS_TOTAL, FINALS_TOTAL, UTTERANCES_TOTAL,
    TTFT_WALL, TTF_WALL, QUEUE_WAIT, INFER_SEC, AUDIO_SEC, RTF,
    BACKLOG_MS
)
from app.gpu_monitor import start_gpu_monitor
from app.vad import AdaptiveEnergyVAD
from app.nemo_asr import NemoASR

cfg = load_config()
logging.basicConfig(level=getattr(logging, cfg.log_level))
log = logging.getLogger("asr_server")

app = FastAPI()

asr: NemoASR | None = None
gpu_sem: asyncio.Semaphore | None = None


@app.on_event("startup")
async def startup():
    global asr, gpu_sem
    asr = NemoASR(cfg.model_name, cfg.device, cfg.sample_rate)
    asr.load()
    gpu_sem = asyncio.Semaphore(cfg.max_concurrent_inferences)
    start_gpu_monitor(cfg.enable_gpu_metrics, cfg.gpu_index)


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.model_name}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    """
    Protocol:
      - client sends first TEXT: {"mode":"mic"}
      - then binary frames: PCM16, 16kHz, mono
      - server emits:
          {"type":"partial","text": "<committed + current>"}   while speaking
          {"type":"final","text":"<segment>","segment_index":N, ...} on silence
          {"type":"final_all","text":"<full transcript>"} on stream end
      - client ends by sending b""
    """
    global asr, gpu_sem
    assert asr is not None and gpu_sem is not None

    await ws.accept()
    ACTIVE_STREAMS.inc()

    client = f"{ws.client.host}:{ws.client.port}"
    log.info(f"WS connect {client}")

    # handshake
    try:
        first = await ws.receive()
        if not first.get("text"):
            raise ValueError("No handshake")
        mode = json.loads(first["text"]).get("mode", "mic")
    except Exception:
        mode = "mic"

    if mode != "mic":
        await ws.send_text(json.dumps({"type": "error", "message": "Send handshake: {'mode':'mic'}"}))
        await ws.close()
        ACTIVE_STREAMS.dec()
        return

    vad = AdaptiveEnergyVAD(
        sample_rate=cfg.sample_rate,
        frame_ms=cfg.vad_frame_ms,
        start_margin=cfg.vad_start_margin,
        min_noise_rms=cfg.vad_min_noise_rms,
        pre_speech_ms=cfg.pre_speech_ms,
    )

    frame_bytes = vad.frame_bytes
    raw_buf = bytearray()

    # utterance tracking
    utt_started = False
    utt_pcm = bytearray()
    utt_ms = 0
    silence_ms = 0

    t_utt_start = None
    t_first_partial = None
    last_partial_emit = 0.0

    # Partial sliding window
    win_frames = max(1, int(cfg.partial_window_ms / cfg.vad_frame_ms))
    partial_ring = deque(maxlen=win_frames)

    # Transcript commit
    full_committed = ""
    segment_index = 0

    async def infer_pcm16(pcm16: bytes):
        q0 = time.perf_counter()
        await gpu_sem.acquire()
        qwait = time.perf_counter() - q0
        if qwait > 0:
            QUEUE_WAIT.observe(qwait)
        try:
            t0 = time.perf_counter()
            audio = asr.pcm16_to_float32(pcm16)
            text = await asyncio.to_thread(asr.transcribe_float32, audio)
            dt = time.perf_counter() - t0
            INFER_SEC.observe(dt)
            return (text or "").strip(), dt
        finally:
            gpu_sem.release()

    def compose_partial(current: str) -> str:
        cur = (current or "").strip()
        if not cur:
            return full_committed.strip()
        if not full_committed.strip():
            return cur
        return f"{full_committed.strip()} {cur}"

    async def emit_partial_if_due():
        nonlocal t_first_partial, last_partial_emit
        now = time.time()
        if (now - last_partial_emit) * 1000 < cfg.partial_every_ms:
            return
        if not utt_started:
            return
        if utt_ms < 300:
            return

        window_pcm = b"".join(partial_ring)
        if len(window_pcm) < frame_bytes * 5:
            return

        text, _dt = await infer_pcm16(window_pcm)
        if text:
            if t_first_partial is None and t_utt_start is not None:
                t_first_partial = time.time()
                TTFT_WALL.observe(t_first_partial - t_utt_start)

            PARTIALS_TOTAL.inc()
            await ws.send_text(json.dumps({
                "type": "partial",
                "text": compose_partial(text)
            }))

        last_partial_emit = now

    async def finalize_segment(reason: str):
        nonlocal full_committed, segment_index, utt_started, utt_pcm, utt_ms, silence_ms, t_utt_start, t_first_partial

        # small pad helps not cut last phoneme
        if cfg.post_speech_pad_ms > 0:
            pad_bytes = int(cfg.sample_rate * (cfg.post_speech_pad_ms / 1000.0) * 2)
            utt_pcm.extend(b"\x00" * pad_bytes)
            utt_ms += cfg.post_speech_pad_ms

        t0 = time.time()
        text, infer_dt = await infer_pcm16(bytes(utt_pcm))
        ttf = time.time() - (t_utt_start or t0)

        FINALS_TOTAL.inc()
        UTTERANCES_TOTAL.inc()
        TTF_WALL.observe(ttf)

        audio_sec = utt_ms / 1000.0
        AUDIO_SEC.observe(audio_sec)
        if audio_sec > 0:
            RTF.observe(infer_dt / audio_sec)

        if text:
            full_committed = (full_committed + " " + text).strip()

        payload = {
            "type": "final",
            "text": text,
            "segment_index": segment_index,
            "reason": reason,
            "audio_ms": utt_ms,
            "ttft_wall_ms": int(1000 * ((t_first_partial - t_utt_start) if (t_first_partial and t_utt_start) else 0)) if t_first_partial else None,
            "ttf_wall_ms": int(1000 * ttf),
        }
        await ws.send_text(json.dumps(payload))
        segment_index += 1

        # reset for next utterance
        utt_started = False
        utt_pcm = bytearray()
        partial_ring.clear()
        utt_ms = 0
        silence_ms = 0
        t_utt_start = None
        t_first_partial = None
        vad.reset()
        BACKLOG_MS.set(0)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                continue

            # end stream
            if data == b"":
                if utt_started and utt_ms >= cfg.min_utt_ms:
                    await finalize_segment("eos")
                await ws.send_text(json.dumps({"type": "final_all", "text": full_committed.strip()}))
                break

            raw_buf.extend(data)
            BACKLOG_MS.set(max(0.0, (len(raw_buf) / (cfg.sample_rate * 2)) * 1000.0))

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre_roll = vad.push_frame(frame)

                if (not utt_started) and pre_roll:
                    utt_started = True
                    utt_pcm = bytearray()
                    partial_ring.clear()

                    utt_pcm.extend(pre_roll)
                    partial_ring.append(pre_roll)

                    utt_ms = int(1000 * (len(pre_roll) / 2) / cfg.sample_rate)
                    silence_ms = 0
                    t_utt_start = time.time()
                    t_first_partial = None
                    last_partial_emit = 0.0

                if utt_started:
                    utt_pcm.extend(frame)
                    partial_ring.append(frame)
                    utt_ms += cfg.vad_frame_ms

                    if is_speech:
                        silence_ms = 0
                    else:
                        silence_ms += cfg.vad_frame_ms

                    await emit_partial_if_due()

                    # endpoint
                    if silence_ms >= cfg.end_silence_ms and utt_ms >= cfg.min_utt_ms:
                        await finalize_segment("silence_endpoint")

                    # hard cap
                    if utt_ms >= cfg.max_utt_ms:
                        await finalize_segment("max_utt")

    finally:
        ACTIVE_STREAMS.dec()
        try:
            await ws.close()
        except Exception:
            pass
        log.info(f"WS disconnect {client}")
