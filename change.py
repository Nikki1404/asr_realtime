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
        context_right=getattr(cfg, "context_right", 64),
    )
    load_sec = engine.load()
    log.info(f"Loaded {cfg.model_name} in {load_sec:.2f}s")


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None

    await ws.accept()
    ACTIVE_STREAMS.inc()
    log.info(f"WS connect {ws.client.host}:{ws.client.port}")

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    # IMPORTANT: set max buffer in session so preprocessing doesn't grow forever
    session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

    frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000.0) * 2
    raw_buf = bytearray()

    utt_started = False
    utt_audio_ms = 0
    t_start = None
    t_first_partial = None
    silence_ms = 0

    async def emit_final(reason: str):
        nonlocal utt_started, utt_audio_ms, t_start, t_first_partial, silence_ms

        if not utt_started:
            return

        final = session.finalize(cfg.post_speech_pad_ms)

        # If you want metrics even for empty final, remove this guard
        if final.strip() == "":
            # reset state and return
            vad.reset()
            utt_started = False
            utt_audio_ms = 0
            t_start = None
            t_first_partial = None
            silence_ms = 0
            return

        UTTERANCES_TOTAL.inc()
        FINALS_TOTAL.inc()

        ttf = time.time() - (t_start or time.time())
        TTF_WALL.observe(ttf)

        audio_sec = utt_audio_ms / 1000.0
        AUDIO_SEC.observe(audio_sec)

        # Compute time from session (already tracked in your StreamingSession)
        compute_sec = session.utt_preproc + session.utt_infer + session.utt_flush
        rtf = (compute_sec / audio_sec) if audio_sec > 0 else None
        if rtf is not None:
            RTF.observe(rtf)

        payload = {
            "type": "final",
            "text": final,
            "reason": reason,
            "audio_ms": utt_audio_ms,
            "ttf_ms": int(ttf * 1000),
            "ttft_ms": int((t_first_partial - t_start) * 1000) if (t_first_partial and t_start) else None,
            "chunks": session.chunks,
            "model_preproc_ms": int(session.utt_preproc * 1000),
            "model_infer_ms": int(session.utt_infer * 1000),
            "model_flush_ms": int(session.utt_flush * 1000),
            "rtf": rtf,
        }

        await ws.send_text(json.dumps(payload))

        # reset for next utterance
        vad.reset()
        utt_started = False
        utt_audio_ms = 0
        t_start = None
        t_first_partial = None
        silence_ms = 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            pcm = msg.get("bytes")
            if pcm is None:
                continue

            # EOS marker from client = end-of-utterance (DO NOT CLOSE SOCKET)
            if pcm == b"":
                await emit_final("client_eou")
                continue

            raw_buf.extend(pcm)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

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
                    silence_ms = 0
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                text = session.step_if_ready()
                if text:
                    PARTIALS_TOTAL.inc()
                    if t_first_partial is None:
                        t_first_partial = time.time()
                        TTFT_WALL.observe(t_first_partial - t_start)
                    await ws.send_text(json.dumps({"type": "partial", "text": text}))

                # Finalize on pause
                if (
                    (not is_speech)
                    and utt_audio_ms >= cfg.min_utt_ms
                    and silence_ms >= cfg.end_silence_ms
                ):
                    await emit_final("server_pause")

                # Hard cap
                elif utt_audio_ms >= cfg.max_utt_ms:
                    await emit_final("server_max_utt")

    finally:
        ACTIVE_STREAMS.dec()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS disconnect")


import asyncio
import argparse
import json
import os
import sys
import time
import wave
import tempfile

import numpy as np
import soundfile as sf
import resampy
import websockets

try:
    import sounddevice as sd
    HAS_MIC = True
except Exception:
    HAS_MIC = False

TARGET_SR = 16000

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# Client-side pause detection (tune if needed)
PAUSE_MS = 1400          # should match/beat server END_SILENCE_MS
RMS_THRESHOLD = 0.008    # raise if fan noise triggers false speech
TAIL_SIL_MS = 800        # silence to help model finalize


def resample_to_16k(wav_path: str) -> str:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr, TARGET_SR)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, audio, TARGET_SR, subtype="PCM_16")
    return tmp.name


async def receiver(ws):
    finals = []
    while True:
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            return finals

        obj = json.loads(msg)
        typ = obj.get("type")

        if typ == "partial":
            txt = obj.get("text", "").replace("\n", " ")
            sys.stdout.write("\r[PARTIAL] " + txt[:160] + " " * 20)
            sys.stdout.flush()

        elif typ == "final":
            txt = (obj.get("text") or "").strip()
            print("\n[FINAL]", txt)
            finals.append(txt)

            print("[SERVER_METRICS]",
                  f"reason={obj.get('reason')}",
                  f"ttft_ms={obj.get('ttft_ms')}",
                  f"ttf_ms={obj.get('ttf_ms')}",
                  f"audio_ms={obj.get('audio_ms')}",
                  f"rtf={obj.get('rtf')}",
                  f"chunks={obj.get('chunks')}",
                  f"preproc_ms={obj.get('model_preproc_ms')}",
                  f"infer_ms={obj.get('model_infer_ms')}",
                  f"flush_ms={obj.get('model_flush_ms')}",
                  )


async def run_wav(ws, wav_path: str, realtime: bool):
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            await ws.send(data)
            if realtime:
                await asyncio.sleep(SLEEP_SEC)

    # finalize
    silence_frames = int(TARGET_SR * (TAIL_SIL_MS / 1000))
    await ws.send(b"\x00\x00" * silence_frames)
    await asyncio.sleep(TAIL_SIL_MS / 1000)
    await ws.send(b"")  # end-of-utterance marker (server will NOT close)


async def run_mic(ws):
    if not HAS_MIC:
        raise RuntimeError("sounddevice not installed. Install: pip install sounddevice")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
        callback=cb,
    )
    stream.start()

    print("🎤 Speak freely. Pause to end sentences. Ctrl+C to exit.")

    last_voice_ts = time.time()
    in_utt = False

    try:
        while True:
            blk = await q.get()
            if blk is None:
                break

            # RMS on int16 block
            x = blk.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))

            now = time.time()
            is_voice = rms >= RMS_THRESHOLD

            if is_voice:
                last_voice_ts = now
                in_utt = True
                await ws.send(blk.tobytes())
            else:
                # don't spam server with noise blocks
                # but if we are in an utterance and paused long enough -> finalize
                if in_utt and ((now - last_voice_ts) * 1000.0) >= PAUSE_MS:
                    # tail silence + EOU marker
                    silence_frames = int(TARGET_SR * (TAIL_SIL_MS / 1000))
                    await ws.send(b"\x00\x00" * silence_frames)
                    await asyncio.sleep(TAIL_SIL_MS / 1000)
                    await ws.send(b"")  # end-of-utterance marker
                    in_utt = False

            await asyncio.sleep(0)  # yield

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        await q.put(None)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")
    p.add_argument("--wav", help="Path to wav file")
    p.add_argument("--mic", action="store_true", help="Use live microphone")
    p.add_argument("--fast", action="store_true", help="Disable realtime pacing (wav only)")
    args = p.parse_args()

    start = time.time()

    async with websockets.connect(args.url, max_size=None) as ws:
        print(f"[INFO] Connected to {args.url}")

        recv_task = asyncio.create_task(receiver(ws))

        if args.mic:
            await run_mic(ws)
        else:
            if not args.wav:
                raise ValueError("--wav required unless --mic is set")

            wav = args.wav
            cleanup = None
            with wave.open(wav, "rb") as wf:
                sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            if sr != TARGET_SR or ch != 1 or sw != 2:
                print(f"[INFO] Resampling WAV → 16kHz mono PCM16 (src={sr}Hz ch={ch} sw={sw})")
                wav = resample_to_16k(wav)
                cleanup = wav

            await run_wav(ws, wav, realtime=not args.fast)
            if cleanup:
                os.unlink(cleanup)

        # user ended mic with Ctrl+C -> close socket so receiver finishes
        try:
            await ws.close()
        except Exception:
            pass

        finals = await recv_task

    total = time.time() - start
    print("\nFULL TRANSCRIPT:")
    print(" ".join([t for t in finals if t.strip()]))

    print("\nCLIENT METRICS:")
    print(f"Total wall time: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())



from dataclasses import dataclass
import os


@dataclass
class Config:
    # Model
    model_name: str = os.getenv("MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
    device: str = os.getenv("DEVICE", "cuda")  # cuda/cpu
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # Streaming latency/accuracy knob (right context)
    # 0 = ultra low latency, 1 = recommended, 6/13 = higher accuracy but more latency
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))

    # VAD + endpointing
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    # Endpointing (pause triggers final)
    end_silence_ms: int = int(os.getenv("END_SILENCE_MS", "900"))
    min_utt_ms: int = int(os.getenv("MIN_UTT_MS", "250"))
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    # When finalizing, we add extra zero padding (ms) to flush last words
    finalize_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))

    # Keep ring-buffer bounded (avoid slowdowns on long sessions)
    max_buffer_ms: int = int(os.getenv("MAX_BUFFER_MS", "12000"))

    # Concurrency
    max_concurrent_inferences: int = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))

    # GPU metrics
    enable_gpu_metrics: bool = os.getenv("ENABLE_GPU_METRICS", "1") == "1"
    gpu_index: int = int(os.getenv("GPU_INDEX", "0"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def load_config() -> Config:
    return Config()
