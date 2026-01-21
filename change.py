# app/main.py  ✅ COMPLETE FIXED VERSION (ASGI send-after-close + TS debug logs + server-side transcript logs)

import json
import time
import logging
from typing import Optional

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
        sample_rate=cfg.sample_rate,
        context_right=cfg.context_right,
    )
    load_sec = engine.load()
    log.info(f"Loaded model={cfg.model_name} in {load_sec:.2f}s on {cfg.device}")


@app.get("/health")
async def health():
    # Useful for browser/TS debugging (network/port/proxy check)
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    """
    ✅ FIXES:
    - Never ws.send_text after disconnect/close (prevents ASGI RuntimeError)
    - Guard sends with ws_open flag + safe_send_text()
    - Catch WebSocketDisconnect / RuntimeError on send
    - Close only if still open
    - Try/catch around accept + receive for TS client debugging
    - Server-side transcript logging (FINAL at INFO, PARTIAL at DEBUG)
    - Logs incoming frame shape: bytes vs text (critical for TS)
    """
    assert engine is not None

    ws_open = False

    # --- Accept (helps debug TS handshake failures) ---
    try:
        await ws.accept()
        ws_open = True
    except Exception as e:
        log.error(f"WS accept failed: {e}")
        return

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

    # server-side framing to 20ms
    frame_bytes = int(cfg.sample_rate * (cfg.vad_frame_ms / 1000.0) * 2)
    pcm_buffer = bytearray()

    def reset_utt_state():
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms
        vad.reset()
        utt_started = False
        utt_audio_ms = 0
        t_start = None
        t_first_partial = None
        silence_ms = 0

    async def safe_send_text(payload: str) -> bool:
        """
        Send only if ws is open. If it fails (closed), return False.
        """
        nonlocal ws_open
        if not ws_open:
            return False

        try:
            await ws.send_text(payload)
            return True
        except (WebSocketDisconnect, RuntimeError) as e:
            # This includes: "Unexpected ASGI message 'websocket.send' after websocket.close"
            ws_open = False
            log.warning(f"WS send failed (closed): {e}")
            return False
        except Exception as e:
            ws_open = False
            log.warning(f"WS send failed (unknown): {e}")
            return False

    async def finalize_and_emit(reason: str):
        """
        ✅ IMPORTANT FIX:
        - Snapshot stats BEFORE finalize() resets them
        - Never crash if WS already closed (safe_send_text)
        - Server-side FINAL transcript log
        """
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms

        if not utt_started:
            return

        # snapshot metrics BEFORE finalize() resets them
        snap_chunks = int(getattr(session, "chunks", 0))
        snap_preproc = float(getattr(session, "utt_preproc", 0.0))
        snap_infer = float(getattr(session, "utt_infer", 0.0))

        flush_t0 = time.perf_counter()
        final = session.finalize(cfg.finalize_pad_ms).strip()
        flush_wall = time.perf_counter() - flush_t0

        UTTERANCES_TOTAL.inc()
        FINALS_TOTAL.inc()

        # wall timings
        now = time.time()
        ttf_sec = (now - t_start) if t_start else 0.0
        TTF_WALL.observe(ttf_sec)

        if t_first_partial and t_start:
            TTFT_WALL.observe(t_first_partial - t_start)

        audio_sec = utt_audio_ms / 1000.0
        AUDIO_SEC.observe(audio_sec)

        PREPROC_SEC.observe(snap_preproc)
        INFER_SEC.observe(snap_infer)
        FLUSH_SEC.observe(flush_wall)

        rtf = None
        if audio_sec > 0:
            rtf = snap_infer / audio_sec
            RTF.observe(rtf)

        payload = {
            "type": "final",
            "text": final,
            "reason": reason,
            "audio_ms": utt_audio_ms,
            "ttft_ms": int((t_first_partial - t_start) * 1000) if (t_first_partial and t_start) else None,
            "ttf_ms": int(ttf_sec * 1000),
            "rtf": rtf,
            "chunks": snap_chunks,
            "preproc_ms": int(snap_preproc * 1000),
            "infer_ms": int(snap_infer * 1000),
            "flush_ms": int(flush_wall * 1000),
        }

        # ✅ server-side FINAL transcript log
        log.info(
            f"[ASR][FINAL] reason={reason} audio_ms={utt_audio_ms} "
            f"ttft_ms={payload['ttft_ms']} ttf_ms={payload['ttf_ms']} rtf={rtf} "
            f"text='{final}'"
        )

        await safe_send_text(json.dumps(payload))

        reset_utt_state()

    try:
        while True:
            # --- receive (helps debug TS protocol mismatches) ---
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                ws_open = False
                await finalize_and_emit("disconnect")
                break
            except Exception as e:
                log.error(f"WS receive failed: {e}")
                ws_open = False
                await finalize_and_emit("receive_error")
                break

            # ✅ client frame shape log (bytes vs text)
            log.info(
                f"WS frame type={msg.get('type')} "
                f"has_bytes={msg.get('bytes') is not None} "
                f"has_text={msg.get('text') is not None}"
            )

            mtype = msg.get("type")

            if mtype == "websocket.disconnect":
                ws_open = False
                await finalize_and_emit("disconnect")
                break

            pcm = msg.get("bytes")

            # Some TS clients might send text frames accidentally
            if pcm is None:
                txt = msg.get("text")
                if txt is not None:
                    log.warning(f"Received text frame (expected bytes): {txt[:200]}")
                continue

            # EOS from client
            if pcm == b"":
                ws_open = False  # client intends to close; avoid post-close sends
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

                # start utterance
                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_start = time.time()
                    t_first_partial = None
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                # accept audio
                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                # streaming partial
                text = session.step_if_ready()
                if text:
                    if t_first_partial is None:
                        t_first_partial = time.time()

                    PARTIALS_TOTAL.inc()

                    # ✅ server-side PARTIAL transcript log (DEBUG only; can be noisy)
                    log.debug(f"[ASR][PARTIAL] {text}")

                    ok = await safe_send_text(json.dumps({"type": "partial", "text": text}))
                    if not ok:
                        ws_open = False
                        break

                # pause endpoint
                should_finalize = (
                    silence_ms >= cfg.end_silence_ms
                    and utt_audio_ms >= cfg.min_utt_ms
                )

                # hard cap
                if utt_audio_ms >= cfg.max_utt_ms:
                    should_finalize = True

                if should_finalize:
                    reason = "pause" if silence_ms >= cfg.end_silence_ms else "max_utt"
                    await finalize_and_emit(reason)

            if not ws_open:
                break

    finally:
        ACTIVE_STREAMS.dec()

        # Close only if still open
        if ws_open:
            try:
                await ws.close()
            except Exception:
                pass

        log.info("WS disconnected")
