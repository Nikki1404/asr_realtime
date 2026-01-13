import asyncio
import json
import sys
import time
import argparse

import numpy as np
import sounddevice as sd
import websockets

# ---------------- CONFIG ----------------
TARGET_SR = 16000

# IMPORTANT:
# Must match server VAD_FRAME_MS
FRAME_MS = 20
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)

# Extra silence to guarantee endpointing
END_SILENCE_MS = 1600
END_SILENCE_SAMPLES = int(TARGET_SR * END_SILENCE_MS / 1000)


# ---------------- UTIL ----------------
def live_partial(text: str):
    text = text.replace("\n", " ")
    sys.stdout.write("\r[PARTIAL] " + text[:160] + " " * 20)
    sys.stdout.flush()


# ---------------- RECEIVER ----------------
async def receiver(ws):
    full = []
    t0 = time.time()
    t_first_partial = None

    async for msg in ws:
        data = json.loads(msg)
        typ = data.get("type")

        if typ == "partial":
            if t_first_partial is None:
                t_first_partial = time.time()
            live_partial(data.get("text", ""))

        elif typ == "final":
            print()
            print("[FINAL]", data.get("text", ""))
            print("   reason:", data.get("reason"))
            print("   audio_ms:", data.get("audio_ms"))
            print("   ttf_ms:", data.get("ttf_ms"))
            print("   rtf:", data.get("rtf"))
            print("   chunks:", data.get("chunks"))
            full.append(data.get("text", ""))

    print("\n\nFULL TRANSCRIPT:")
    print(" ".join(full))

    print("\nCLIENT METRICS:")
    print(f"Total wall time: {(time.time()-t0):.2f}s")
    if t_first_partial:
        print(f"TTFT wall: {(t_first_partial-t0)*1000:.1f} ms")


# ---------------- SENDER (MIC) ----------------
async def sender(ws):
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    )

    stream.start()
    print("🎤 Speak normally. Pause to end sentence.")
    print("➡ Press ENTER to stop recording cleanly.\n")

    async def mic_loop():
        while True:
            block = await q.get()
            if block is None:
                return
            await ws.send(block.tobytes())

    mic_task = asyncio.create_task(mic_loop())

    # Wait for ENTER
    await asyncio.to_thread(sys.stdin.readline)

    # Stop mic
    stream.stop()
    stream.close()

    # Send silence to force endpoint
    await ws.send(b"\x00\x00" * END_SILENCE_SAMPLES)
    await asyncio.sleep(END_SILENCE_MS / 1000)

    # Proper EOS
    await ws.send(b"")

    await q.put(None)
    await mic_task


# ---------------- MAIN ----------------
async def main(url: str):
    async with websockets.connect(url, max_size=None) as ws:
        print(f"[INFO] Connected to {url}")

        recv_task = asyncio.create_task(receiver(ws))
        send_task = asyncio.create_task(sender(ws))

        await asyncio.gather(send_task, recv_task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8003/ws/asr")
    args = ap.parse_args()

    asyncio.run(main(args.url))
