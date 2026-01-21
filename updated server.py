import json
import time
import logging
from typing import Optional
import os
import wave
import uuid
import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.nemotron_streaming import NemotronStreamingASR

cfg = load_config()
logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("asr_server")

app = FastAPI()
engine: Optional[NemotronStreamingASR] = None


@app.on_event("startup")
async def startup():
    global engine
    engine = NemotronStreamingASR(
        model_name=cfg.model_name,
        device=cfg.device,
        sample_rate=cfg.sample_rate,   # 16k
        context_right=cfg.context_right,
    )
    load_sec = engine.load()
    log.info(f"Loaded model={cfg.model_name} in {load_sec:.2f}s on {cfg.device}")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None

    session_id = uuid.uuid4().hex[:8]

    # 🔑 Input SR from client (NodeJS = 8000, Python = 16000)
    INPUT_SAMPLE_RATE = int(os.getenv("INPUT_SAMPLE_RATE", "16000"))
    TARGET_SAMPLE_RATE = cfg.sample_rate  # 16000

    if INPUT_SAMPLE_RATE not in (8000, 16000):
        log.warning(f"[ASR][{session_id}] Unsupported INPUT_SAMPLE_RATE={INPUT_SAMPLE_RATE}")

    ws_open = False

    try:
        await ws.accept()
        ws_open = True
    except Exception as e:
        log.error(f"WS accept failed: {e}")
        return

    ACTIVE_STREAMS.inc()
    log.info(
        f"WS connected: {ws.client} session={session_id} "
        f"input_sr={INPUT_SAMPLE_RATE} target_sr={TARGET_SAMPLE_RATE}"
    )

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

    frame_bytes = int(cfg.sample_rate * (cfg.vad_frame_ms / 1000.0) * 2)
    pcm_buffer = bytearray()
    received_pcm = bytearray()

    def reset_utt_state():
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms
        vad.reset()
        utt_started = False
        utt_audio_ms = 0
        t_start = None
        t_first_partial = None
        silence_ms = 0

    async def safe_send_text(payload: str) -> bool:
        nonlocal ws_open
        if not ws_open:
            return False
        try:
            await ws.send_text(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            ws_open = False
            return False
        except Exception:
            ws_open = False
            return False

    def dump_received_audio():
        if not received_pcm:
            return

        debug_dir = getattr(cfg, "debug_audio_dir", "./debug_audio")
        os.makedirs(debug_dir, exist_ok=True)

        wav_path = os.path.join(
            debug_dir,
            f"ws_{session_id}_{int(time.time() * 1000)}.wav"
        )

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(cfg.sample_rate)
            wf.writeframes(received_pcm)

        log.info(f"[ASR][DEBUG][{session_id}] WAV dumped → {wav_path}")
        received_pcm.clear()

    async def finalize_and_emit(reason: str):
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms

        if not utt_started:
            return

        final = session.finalize(cfg.finalize_pad_ms).strip()
        dump_received_audio()

        payload = {
            "type": "final",
            "text": final,
            "reason": reason,
            "audio_ms": utt_audio_ms,
        }

        log.info(f"[ASR][FINAL][{session_id}] {final}")
        await safe_send_text(json.dumps(payload))
        reset_utt_state()

    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                ws_open = False
                await finalize_and_emit("disconnect")
                break

            pcm = msg.get("bytes")

            if pcm is None:
                continue

            # ===============================
            # 🔑 UPSAMPLING LOGIC (SAFE)
            # ===============================
            if INPUT_SAMPLE_RATE == 8000 and pcm:
                # int16 → duplicate samples → int16
                x = np.frombuffer(pcm, dtype=np.int16)
                x = np.repeat(x, 2)
                pcm = x.tobytes()
            # If 16k → untouched

            if pcm:
                received_pcm.extend(pcm)

            if pcm == b"":
                ws_open = False
                await finalize_and_emit("eos")
                break

            pcm_buffer.extend(pcm)

            while len(pcm_buffer) >= frame_bytes:
                frame = bytes(pcm_buffer[:frame_bytes])
                del pcm_buffer[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)

                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += cfg.vad_frame_ms

                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_start = time.time()
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                text = session.step_if_ready()
                if text:
                    await safe_send_text(json.dumps({"type": "partial", "text": text}))

                should_finalize = (
                    silence_ms >= cfg.end_silence_ms
                    and utt_audio_ms >= cfg.min_utt_ms
                )

                if utt_audio_ms >= cfg.max_utt_ms:
                    should_finalize = True

                if should_finalize:
                    await finalize_and_emit("pause")

            if not ws_open:
                break

    finally:
        ACTIVE_STREAMS.dec()
        if ws_open:
            try:
                await ws.close()
            except Exception:
                pass
        log.info(f"WS disconnected session={session_id}")
