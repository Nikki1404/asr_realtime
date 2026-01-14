import json
import time
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.nemotron_streaming import NemotronStreamingASR

cfg = load_config()
logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("asr_server")

app = FastAPI()
engine: NemotronStreamingASR | None = None


@app.on_event("startup")
async def startup():
    global engine
    engine = NemotronStreamingASR(
        cfg.model_name,
        cfg.device,
        cfg.sample_rate,
        cfg.context_right,
    )
    engine.load()


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None
    await ws.accept()
    ACTIVE_STREAMS.inc()

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    session = engine.new_session(cfg.max_buffer_ms)

    utt_started = False
    utt_audio_ms = 0
    silence_ms = 0
    t_start = None
    t_first_partial = None

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            pcm = msg.get("bytes")
            if pcm == b"":
                break

            is_speech, pre = vad.push_frame(pcm)
            silence_ms = 0 if is_speech else silence_ms + cfg.vad_frame_ms

            if pre and not utt_started:
                utt_started = True
                utt_audio_ms = 0
                t_start = time.time()
                t_first_partial = None
                session.accept_pcm16(pre)

            if not utt_started:
                continue

            session.accept_pcm16(pcm)
            utt_audio_ms += cfg.vad_frame_ms

            text = session.step_if_ready()
            if text:
                if t_first_partial is None:
                    t_first_partial = time.time()
                    TTFT_WALL.observe(t_first_partial - t_start)
                PARTIALS_TOTAL.inc()
                await ws.send_text(json.dumps({"type": "partial", "text": text}))

            if silence_ms >= cfg.end_silence_ms and utt_audio_ms >= cfg.min_utt_ms:
                final = session.finalize(cfg.finalize_pad_ms)

                UTTERANCES_TOTAL.inc()
                FINALS_TOTAL.inc()

                audio_sec = utt_audio_ms / 1000.0
                rtf = session.utt_infer / audio_sec if audio_sec > 0 else None

                payload = {
                    "type": "final",
                    "text": final,
                    "reason": "pause",
                    "audio_ms": utt_audio_ms,
                    "chunks": session.chunks,
                    "rtf": rtf,
                    "preproc_ms": int(session.utt_preproc * 1000),
                    "infer_ms": int(session.utt_infer * 1000),
                    "flush_ms": int(session.utt_flush * 1000),
                }

                await ws.send_text(json.dumps(payload))

                session.reset_stream_state()
                vad.reset()
                utt_started = False
                utt_audio_ms = 0
                silence_ms = 0

    finally:
        ACTIVE_STREAMS.dec()
        await ws.close()
