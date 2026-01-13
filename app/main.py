import asyncio
import json
import logging
import time

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config
from app.metrics import (
    ACTIVE_STREAMS, PARTIALS_TOTAL, FINALS_TOTAL, UTTERANCES_TOTAL,
    TTFT_WALL, TTF_WALL, QUEUE_WAIT, PREPROC_SEC, INFER_SEC, FLUSH_SEC,
    AUDIO_SEC, RTF, BACKLOG_MS,
)
from app.gpu_monitor import start_gpu_monitor
from app.vad import AdaptiveEnergyVAD
from app.endpointing import Endpointing
from app.nemotron_streaming import NemotronStreamingASR

cfg = load_config()
logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
log = logging.getLogger("asr_server")

app = FastAPI()

engine: NemotronStreamingASR | None = None
gpu_sem: asyncio.Semaphore | None = None


@app.on_event("startup")
async def startup():
    global engine, gpu_sem
    engine = NemotronStreamingASR(
        model_name=cfg.model_name,
        device=cfg.device,
        sample_rate=cfg.sample_rate,
        context_right=cfg.context_right,
    )
    load_sec = engine.load()
    gpu_sem = asyncio.Semaphore(int(cfg.max_concurrent_inferences))
    start_gpu_monitor(cfg.enable_gpu_metrics, cfg.gpu_index)
    log.info(f"Loaded model={cfg.model_name} in {load_sec:.2f}s on {cfg.device}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.model_name}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None and gpu_sem is not None

    await ws.accept()
    ACTIVE_STREAMS.inc()

    client = f"{ws.client.host}:{ws.client.port}"
    log.info(f"WS connect {client}")

    session = engine.new_session(max_buffer_ms=cfg.max_buffer_ms)
    vad = AdaptiveEnergyVAD(cfg.sample_rate, cfg.vad_frame_ms, cfg.vad_start_margin, cfg.vad_min_noise_rms, cfg.pre_speech_ms)
    ep = Endpointing(cfg.end_silence_ms, cfg.min_utt_ms, cfg.max_utt_ms)

    frame_bytes = int(cfg.sample_rate * (cfg.vad_frame_ms / 1000.0) * 2)
    raw_buf = bytearray()

    utt_started = False
    utt_audio_ms = 0
    t_utt_start = 0.0
    t_first_partial = None

    async def send_partial(text: str):
        PARTIALS_TOTAL.inc()
        await ws.send_text(json.dumps({"type": "partial", "text": text}))

    async def finalize(reason: str):
        nonlocal utt_started, utt_audio_ms, t_utt_start, t_first_partial

        if not utt_started:
            return

        # GPU semaphore lock (concurrency)
        q0 = time.perf_counter()
        await gpu_sem.acquire()
        qwait = time.perf_counter() - q0
        if qwait > 0:
            QUEUE_WAIT.observe(qwait)

        try:
            final_text = session.finalize(pad_ms=cfg.finalize_pad_ms)
        finally:
            gpu_sem.release()

        # Emit even if empty? (usually keep it gated)
        if final_text.strip():
            UTTERANCES_TOTAL.inc()
            FINALS_TOTAL.inc()

            ttf = time.time() - t_utt_start
            TTF_WALL.observe(ttf)

            audio_sec = max(0.001, utt_audio_ms / 1000.0)
            AUDIO_SEC.observe(audio_sec)

            # "compute" = preproc + infer + flush
            compute = session.utt_preproc + session.utt_infer + session.utt_flush
            rtf = compute / audio_sec
            RTF.observe(rtf)

            PREPROC_SEC.observe(session.utt_preproc)
            INFER_SEC.observe(session.utt_infer)
            FLUSH_SEC.observe(session.utt_flush)

            payload = {
                "type": "final",
                "text": final_text.strip(),
                "reason": reason,
                "ttft_ms": int(1000 * (t_first_partial - t_utt_start)) if t_first_partial else None,
                "ttf_ms": int(1000 * ttf),
                "audio_ms": utt_audio_ms,
                "rtf": rtf,
                "chunks": session.chunks,
                "model_preproc_ms": int(1000 * session.utt_preproc),
                "model_infer_ms": int(1000 * session.utt_infer),
                "model_flush_ms": int(1000 * session.utt_flush),
                "backlog_ms_end": session.backlog_ms(),
            }
            await ws.send_text(json.dumps(payload))

        # reset utterance tracking
        utt_started = False
        utt_audio_ms = 0
        t_utt_start = 0.0
        t_first_partial = None
        ep.reset()
        vad.reset()
        BACKLOG_MS.set(0)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                # optional control message
                txt = msg.get("text")
                if txt:
                    try:
                        obj = json.loads(txt)
                        if obj.get("type") == "end":
                            await finalize("client_end")
                    except Exception:
                        pass
                continue

            # client EOS
            if data == b"":
                await finalize("eos")
                break

            raw_buf.extend(data)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre_roll = vad.push_frame(frame)

                if (not utt_started) and pre_roll is not None:
                    utt_started = True
                    utt_audio_ms = 0
                    t_utt_start = time.time()
                    t_first_partial = None
                    ep.reset()
                    session.reset_stream_state()

                    # feed pre-roll + count
                    session.accept_pcm16(pre_roll)
                    utt_audio_ms += int(1000 * (len(pre_roll) / 2) / cfg.sample_rate)

                if utt_started:
                    session.accept_pcm16(frame)
                    utt_audio_ms += cfg.vad_frame_ms
                    BACKLOG_MS.set(session.backlog_ms())

                    # streaming step loop (might emit multiple partial updates)
                    while True:
                        q0 = time.perf_counter()
                        await gpu_sem.acquire()
                        qwait = time.perf_counter() - q0
                        if qwait > 0:
                            QUEUE_WAIT.observe(qwait)

                        try:
                            text = session.step_if_ready()
                        finally:
                            gpu_sem.release()

                        if text is None:
                            break

                        if t_first_partial is None:
                            t_first_partial = time.time()
                            TTFT_WALL.observe(t_first_partial - t_utt_start)

                        await send_partial(text)

                    # endpointing on silence/pause
                    if ep.update(is_speech, cfg.vad_frame_ms, utt_audio_ms):
                        await finalize("pause")

    finally:
        ACTIVE_STREAMS.dec()
        try:
            await ws.close()
        except Exception:
            pass
        log.info(f"WS disconnect {client}")
