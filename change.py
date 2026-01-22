import json
import time
import logging
from typing import Optional
import os
import wave
import uuid

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

    session_id = uuid.uuid4().hex[:8]

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
        except (WebSocketDisconnect, RuntimeError) as e:
            ws_open = False
            log.warning(f"WS send failed (closed): {e}")
            return False
        except Exception as e:
            ws_open = False
            log.warning(f"WS send failed (unknown): {e}")
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


do we need to send only this or both? Can you confirm?
for start -> ws.send(JSON.stringify({
  type: "start",
  sample_rate: 8000   // or 16000 if already 16k
}));
for config -> {
      "type": "config",
      "audio": {
        "sample_rate": 8000,
        "frame_ms": 20,
        "channels": 1,
        "format": "pcm16"
      },
      "client": {
        "name": "web-ts",
        "realtime": true
      }
    }

(trapped) error reading bcrypt version
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 620, in _load_backend_mixin
    version = _bcrypt.__about__.__version__
AttributeError: module 'bcrypt' has no attribute '__about__'
INFO:     10.90.126.154:15882 - "POST /apps/docs/api/auth/register HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/fastapi/applications.py", line 1139, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/middleware/errors.py", line 186, in __call__
    raise exc
  File "/usr/local/lib/python3.10/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "/usr/local/lib/python3.10/site-packages/starlette/middleware/cors.py", line 93, in __call__
    await self.simple_response(scope, receive, send, request_headers=headers)
  File "/usr/local/lib/python3.10/site-packages/starlette/middleware/cors.py", line 144, in simple_response
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/site-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/fastapi/routing.py", line 120, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/usr/local/lib/python3.10/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/site-packages/fastapi/routing.py", line 106, in app
    response = await f(request)
  File "/usr/local/lib/python3.10/site-packages/fastapi/routing.py", line 430, in app
    raw_response = await run_endpoint_function(
  File "/usr/local/lib/python3.10/site-packages/fastapi/routing.py", line 316, in run_endpoint_function
    return await dependant.call(**values)
  File "/app/api/users.py", line 59, in register_user
    hashed_password = AuthService.get_password_hash(user_data.password)
  File "/app/auth.py", line 40, in get_password_hash
    return pwd_context.hash(password)
  File "/usr/local/lib/python3.10/site-packages/passlib/context.py", line 2258, in hash
    return record.hash(secret, **kwds)
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 779, in hash
    self.checksum = self._calc_checksum(secret)
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 591, in _calc_checksum
    self._stub_requires_backend()
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2254, in _stub_requires_backend
    cls.set_backend()
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2156, in set_backend
    return owner.set_backend(name, dryrun=dryrun)
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2163, in set_backend
    return cls.set_backend(name, dryrun=dryrun)
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2188, in set_backend
    cls._set_backend(name, dryrun)
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2311, in _set_backend
    super(SubclassBackendMixin, cls)._set_backend(name, dryrun)
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 2224, in _set_backend
    ok = loader(**kwds)
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 626, in _load_backend_mixin
    return mixin_cls._finalize_backend_mixin(name, dryrun)
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 421, in _finalize_backend_mixin
    if detect_wrap_bug(IDENT_2A):
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 380, in detect_wrap_bug
    if verify(secret, bug_hash):
  File "/usr/local/lib/python3.10/site-packages/passlib/utils/handlers.py", line 792, in verify
    return consteq(self._calc_checksum(secret), chk)
  File "/usr/local/lib/python3.10/site-packages/passlib/handlers/bcrypt.py", line 655, in _calc_checksum
    hash = _bcrypt.hashpw(secret, config)
ValueError: password cannot be longer than 72 bytes, truncate manually if necessary (e.g. my_password[:72])
