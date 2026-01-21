import json
import time
import logging
from typing import Optional

#  ADDED
import os
import wave
import uuid

#  ADDED (server-side upsampling)
import numpy as np
import resampy

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
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None

    #  ADDED: unique session id
    session_id = uuid.uuid4().hex[:8]

    #  ADDED (assumed NodeJS input SR)
    INPUT_SAMPLE_RATE = int(os.getenv("INPUT_SAMPLE_RATE", "8000"))
    TARGET_SAMPLE_RATE = int(cfg.sample_rate)

    #  ADDED: one-time log so you know server is resampling
    if INPUT_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        log.info(f"[ASR][{session_id}] Resampling enabled: {INPUT_SAMPLE_RATE}Hz → {TARGET_SAMPLE_RATE}Hz")

    ws_open = False

    try:
        await ws.accept()
        ws_open = True
    except Exception as e:
        log.error(f"WS accept failed: {e}")
        return

    ACTIVE_STREAMS.inc()
    log.info(f"WS connected: {ws.client} session={session_id}")

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

    #  ADDED: raw PCM accumulator
    received_pcm = bytearray()

    #  ADDED: 8k→16k resampler state (handles arbitrary chunk sizes cleanly)
    _rs_in = np.array([], dtype=np.float32)
    _rs_pos = 0.0
    _rs_ratio = (TARGET_SAMPLE_RATE / float(INPUT_SAMPLE_RATE)) if INPUT_SAMPLE_RATE > 0 else 2.0

    def _resample_pcm16_stream(pcm_bytes: bytes) -> bytes:
        """
        Streaming-ish resample:
        - accumulates float32 input
        - emits int16 output
        - preserves phase across packets using _rs_pos
        """
        nonlocal _rs_in, _rs_pos, _rs_ratio

        if not pcm_bytes:
            return b""

        if INPUT_SAMPLE_RATE == TARGET_SAMPLE_RATE:
            return pcm_bytes

        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return b""

        # append new samples
        _rs_in = np.concatenate([_rs_in, x])

        # determine how many output samples we can safely emit
        out_count = int((len(_rs_in) - _rs_pos) * _rs_ratio)
        if out_count <= 0:
            return b""

        # positions in input for each output sample (linear interpolation)
        idx = _rs_pos + (np.arange(out_count, dtype=np.float32) / _rs_ratio)
        i0 = np.floor(idx).astype(np.int64)
        frac = idx - i0
        i1 = np.minimum(i0 + 1, len(_rs_in) - 1)

        y = (1.0 - frac) * _rs_in[i0] + frac * _rs_in[i1]

        # advance position; drop consumed input to keep buffer bounded
        _rs_pos = idx[-1] + (1.0 / _rs_ratio)
        drop = int(_rs_pos) - 8  # keep a tiny tail for interpolation safety
        if drop > 0:
            _rs_in = _rs_in[drop:]
            _rs_pos -= drop

        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    def reset_utt_state():
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms
        vad.reset()
        utt_started = False
        utt_audio_ms = 0
        t_start = None
        t_first_partial = None
        silence_ms = 0

        #  ADDED: also reset resampler buffers at utterance boundary
        nonlocal _rs_in, _rs_pos
        _rs_in = np.array([], dtype=np.float32)
        _rs_pos = 0.0

    async def safe_send_text(payload: str) -> bool:
        nonlocal ws_open
        if not ws_open:
            return False
        try:
            await ws.send_text(payload)
            return True
        except (WebSocketDisconnect, RuntimeError) as e:
            ws_open = False
            log.warning(f"WS send failed (closed): {e}")
            return False
        except Exception as e:
            ws_open = False
            log.warning(f"WS send failed (unknown): {e}")
            return False

    #  ADDED: WAV dump helper
    def dump_received_audio():
        if not received_pcm:
            return

        # gate by config if present; default OFF unless cfg.enable_audio_dump exists and is True
        if hasattr(cfg, "enable_audio_dump") and not getattr(cfg, "enable_audio_dump", False):
            received_pcm.clear()
            return

        debug_dir = getattr(cfg, "debug_audio_dir", "./debug_audio")
        os.makedirs(debug_dir, exist_ok=True)

        wav_path = os.path.join(
            debug_dir,
            f"ws_{session_id}_{int(time.time() * 1000)}.wav"
        )

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)               # PCM16
            wf.setframerate(cfg.sample_rate)
            wf.writeframes(received_pcm)

        log.info(f"[ASR][DEBUG][{session_id}] WAV dumped → {wav_path}")
        received_pcm.clear()

    async def finalize_and_emit(reason: str):
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms

        if not utt_started:
            return

        snap_chunks = int(getattr(session, "chunks", 0))
        snap_preproc = float(getattr(session, "utt_preproc", 0.0))
        snap_infer = float(getattr(session, "utt_infer", 0.0))

        flush_t0 = time.perf_counter()
        final = session.finalize(cfg.finalize_pad_ms).strip()
        flush_wall = time.perf_counter() - flush_t0

        #  ADDED: dump received PCM as WAV (this is 16k after resampling)
        dump_received_audio()

        UTTERANCES_TOTAL.inc()
        FINALS_TOTAL.inc()

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

        log.info(
            f"[ASR][FINAL][{session_id}] reason={reason} audio_ms={utt_audio_ms} "
            f"text='{final}'"
        )

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
            except Exception as e:
                log.error(f"WS receive failed: {e}")
                ws_open = False
                await finalize_and_emit("receive_error")
                break

            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                ws_open = False
                await finalize_and_emit("disconnect")
                break

            pcm = msg.get("bytes")

            if pcm is None:
                txt = msg.get("text")
                if txt is not None:
                    log.warning(f"Received text frame (expected bytes): {txt[:200]}")
                continue

            #  ADDED: upsample 8k PCM → 16k PCM BEFORE any downstream processing
            if pcm:
                pcm = _resample_pcm16_stream(pcm)

            #  ADDED: accumulate exact PCM bytes (after resampling → 16k)
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
                    t_first_partial = None
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                text = session.step_if_ready()
                if text:
                    if t_first_partial is None:
                        t_first_partial = time.time()

                    PARTIALS_TOTAL.inc()
                    log.debug(f"[ASR][PARTIAL][{session_id}] {text}")

                    ok = await safe_send_text(json.dumps({"type": "partial", "text": text}))
                    if not ok:
                        ws_open = False
                        break

                should_finalize = (
                    silence_ms >= cfg.end_silence_ms
                    and utt_audio_ms >= cfg.min_utt_ms
                )

                if utt_audio_ms >= cfg.max_utt_ms:
                    should_finalize = True

                if should_finalize:
                    reason = "pause" if silence_ms >= cfg.end_silence_ms else "max_utt"
                    await finalize_and_emit(reason)

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




docker run -d --gpus all \
  -p 4000:8003 \
  -e ENABLE_AUDIO_DUMP=1 \
  -e DEBUG_AUDIO_DIR=/srv/debug_audio \
  -v $(pwd)/debug_audio:/srv/debug_audio \
  cx_asr_realtime_nemotron

