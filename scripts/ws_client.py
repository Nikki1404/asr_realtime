import asyncio
import json
import time
import sys
import numpy as np
import sounddevice as sd
import websockets

# ================= CONFIG =================
URL = "ws://127.0.0.1:8003/ws/asr"
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

END_SILENCE_MS = 1500
END_SILENCE_SAMPLES = int(SAMPLE_RATE * END_SILENCE_MS / 1000)

# ================= DEBUG =================
print("🔍 Available audio devices:")
print(sd.query_devices())
print("🎙 Default input device:", sd.default.device)

# ================= MIC STREAM =================
async def mic_sender(ws):
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("⚠️ MIC STATUS:", status, flush=True)
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    )

    stream.start()
    print("🎤 MIC STREAM STARTED — SPEAK NOW", flush=True)

    try:
        while True:
            data = await q.get()
            await ws.send(data.tobytes())
    except asyncio.CancelledError:
        print("🛑 Mic sender cancelled", flush=True)
    finally:
        stream.stop()
        stream.close()

# ================= RECEIVER =================
async def receiver(ws):
    async for msg in ws:
        data = json.loads(msg)
        if data["type"] == "partial":
            print("\r[PARTIAL]", data["text"][:120], end="", flush=True)

        elif data["type"] == "final":
            print("\n[FINAL]", data["text"], flush=True)
            print("📊 METRICS:", data, flush=True)

# ================= MAIN =================
async def main():
    print("🌐 Connecting to", URL, flush=True)

    async with websockets.connect(URL, max_size=None) as ws:
        print("✅ WebSocket connected", flush=True)

        recv_task = asyncio.create_task(receiver(ws))
        send_task = asyncio.create_task(mic_sender(ws))

        print("⏳ Press ENTER to stop recording", flush=True)
        await asyncio.to_thread(sys.stdin.readline)

        print("🧠 Sending silence to trigger finalization", flush=True)

        await ws.send(b"\x00\x00" * END_SILENCE_SAMPLES)
        await asyncio.sleep(END_SILENCE_MS / 1000)
        await ws.send(b"")

        send_task.cancel()
        await recv_task

# ================= ENTRY =================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting cleanly")
