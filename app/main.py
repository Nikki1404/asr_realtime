import asyncio, json, time
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from collections import deque

from app.config import load_config
from app.metrics import *
from app.gpu_monitor import start_gpu_monitor
from app.vad import AdaptiveEnergyVAD
from app.nemo_asr import NemoASR

cfg = load_config()
app = FastAPI()
asr = None
gpu_sem = None

@app.on_event("startup")
async def startup():
    global asr, gpu_sem
    asr = NemoASR(cfg.model_name, cfg.device, cfg.sample_rate)
    gpu_sem = asyncio.Semaphore(cfg.max_concurrent_inferences)
    start_gpu_monitor(cfg.enable_gpu_metrics, cfg.gpu_index)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    ACTIVE_STREAMS.inc()

    await ws.receive()  # handshake {"mode":"mic"}

    vad = AdaptiveEnergyVAD(cfg.sample_rate, cfg.vad_frame_ms,
                            cfg.vad_start_margin, cfg.vad_min_noise_rms,
                            cfg.pre_speech_ms)

    raw = bytearray()
    utt = bytearray()
    partial_buf = deque(maxlen=int(cfg.partial_window_ms/cfg.vad_frame_ms))

    utt_started = False
    silence = 0
    utt_ms = 0
    full = ""

    async def infer(pcm):
        async with gpu_sem:
            return asr.transcribe(asr.pcm16_to_f32(pcm))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data == b"":
                if utt:
                    txt = await infer(utt)
                    await ws.send_text(json.dumps({"type":"final","text":txt}))
                await ws.send_text(json.dumps({"type":"final_all","text":full.strip()}))
                break

            raw.extend(data)
            while len(raw) >= vad.frame_bytes:
                frame = raw[:vad.frame_bytes]
                raw = raw[vad.frame_bytes:]

                speech, pre = vad.push(frame)
                if pre:
                    utt_started = True
                    utt.extend(pre)
                    partial_buf.append(pre)

                if utt_started:
                    utt.extend(frame)
                    partial_buf.append(frame)
                    utt_ms += cfg.vad_frame_ms

                    if speech:
                        silence = 0
                    else:
                        silence += cfg.vad_frame_ms

                    if utt_ms > 400 and silence < cfg.end_silence_ms:
                        txt = await infer(b"".join(partial_buf))
                        await ws.send_text(json.dumps({"type":"partial","text":txt}))

                    if silence >= cfg.end_silence_ms:
                        txt = await infer(utt)
                        full += " " + txt
                        await ws.send_text(json.dumps({"type":"final","text":txt}))
                        utt.clear()
                        partial_buf.clear()
                        vad.reset()
                        utt_started = False
                        silence = 0
                        utt_ms = 0
    finally:
        ACTIVE_STREAMS.dec()
        await ws.close()
