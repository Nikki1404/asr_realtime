import asyncio
import argparse
import json
import time

import sounddevice as sd
import numpy as np
import websockets

TARGET_SR = 16000

# Bigger blocks = better RNNT stability
BLOCK_MS = 120
BLOCK_SAMPLES = int(TARGET_SR * BLOCK_MS / 1000)

def live_print(txt: str):
    txt = txt.replace("\n", " ")
    print("\r[PARTIAL] " + txt[:160] + " " * 25, end="", flush=True)

async def receiver(ws):
    committed = []
    t0 = time.time()
    first_partial = None

    async for msg in ws:
        obj = json.loads(msg)
        typ = obj.get("type")

        if typ == "partial":
            if first_partial is None:
                first_partial = time.time()
            live_print(obj.get("text", ""))

        elif typ == "final":
            seg = (obj.get("text") or "").strip()
            if seg:
                committed.append(seg)
                print("\n[FINAL] " + seg)

        elif typ == "final_all":
            print("\n\nFULL TRANSCRIPT:\n" + (obj.get("text") or "").strip())
            print("\nCLIENT METRICS:")
            print(f"Total wall: {(time.time()-t0):.2f}s")
            if first_partial:
                print(f"TTFT wall: {(first_partial-t0)*1000:.1f} ms")
            break

async def sender(ws):
    q = asyncio.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            pass
        q.put_nowait(indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SAMPLES,
        callback=cb,
    )
    stream.start()
    print("🎤 Speak now (CTRL+C to stop).")
    try:
        while True:
            block = await q.get()
            await ws.send(block.tobytes())
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop()
        stream.close()

async def main(url: str):
    async with websockets.connect(url, max_size=None) as ws:
        print(f"[INFO] Connected to {url}")
        await ws.send(json.dumps({"mode": "mic"}))

        recv = asyncio.create_task(receiver(ws))
        send = asyncio.create_task(sender(ws))

        try:
            await asyncio.gather(recv, send)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                await ws.send(b"")
            except Exception:
                pass
            send.cancel()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080/ws/asr")
    args = ap.parse_args()
    asyncio.run(main(args.url))
