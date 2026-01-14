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
        model_name=cfg.model_name,
        device=cfg.device,
        sample_rate=cfg.sample_rate,
        context_right=cfg.context_right,
    )
    load_sec = engine.load()
    log.info(f"Loaded model={cfg.model_name} in {load_sec:.2f}s on {cfg.device}")


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None

    await ws.accept()
    ACTIVE_STREAMS.inc()
    log.info(f"WS connected: {ws.client}")

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    session = engine.new_session(max_buffer_ms=cfg.max_buffer_ms)

    utt_started = False
    utt_audio_ms = 0
    t_start = None
    t_first_partial = None
    silence_ms = 0

    # 🔴 FIX: server-side framing (20 ms)
    frame_bytes = int(cfg.sample_rate * (cfg.vad_frame_ms / 1000) * 2)
    pcm_buffer = bytearray()

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            pcm = msg.get("bytes")

            # EOS from client
            if pcm == b"":
                if utt_started:
                    final = session.finalize(cfg.finalize_pad_ms).strip()
                    UTTERANCES_TOTAL.inc()
                    FINALS_TOTAL.inc()
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final,
                        "reason": "eos",
                    }))
                break

            pcm_buffer.extend(pcm)

            # 🔴 FIX: always process exact VAD frames
            while len(pcm_buffer) >= frame_bytes:
                frame = bytes(pcm_buffer[:frame_bytes])
                del pcm_buffer[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)

                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += cfg.vad_frame_ms

                # start utterance
                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_start = time.time()
                    t_first_partial = None
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                # streaming partials
                text = session.step_if_ready()
                if text:
                    if t_first_partial is None:
                        t_first_partial = time.time()
                        TTFT_WALL.observe(t_first_partial - t_start)

                    PARTIALS_TOTAL.inc()
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": text
                    }))

                # pause endpoint
                should_finalize = (
                    silence_ms >= cfg.end_silence_ms
                    and utt_audio_ms >= cfg.min_utt_ms
                )

                # hard cap
                if utt_audio_ms >= cfg.max_utt_ms:
                    should_finalize = True

                if should_finalize:
                    final = session.finalize(cfg.finalize_pad_ms).strip()

                    UTTERANCES_TOTAL.inc()
                    FINALS_TOTAL.inc()

                    ttf = time.time() - t_start
                    TTF_WALL.observe(ttf)

                    AUDIO_SEC.observe(utt_audio_ms / 1000.0)
                    PREPROC_SEC.observe(session.utt_preproc)
                    INFER_SEC.observe(session.utt_infer)
                    FLUSH_SEC.observe(session.utt_flush)

                    rtf = None
                    if utt_audio_ms > 0:
                        rtf = session.utt_infer / (utt_audio_ms / 1000.0)
                        RTF.observe(rtf)

                    payload = {
                        "type": "final",
                        "text": final,
                        "reason": "pause" if silence_ms >= cfg.end_silence_ms else "max_utt",
                        "audio_ms": utt_audio_ms,
                        "ttft_ms": int((t_first_partial - t_start) * 1000) if t_first_partial else None,
                        "ttf_ms": int(ttf * 1000),
                        "rtf": rtf,
                        "chunks": session.chunks,
                        "preproc_ms": int(session.utt_preproc * 1000),
                        "infer_ms": int(session.utt_infer * 1000),
                        "flush_ms": int(session.utt_flush * 1000),
                    }

                    await ws.send_text(json.dumps(payload))

                    # reset
                    session.reset_stream_state()
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0
                    t_start = None
                    t_first_partial = None

    finally:
        ACTIVE_STREAMS.dec()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS disconnected")
