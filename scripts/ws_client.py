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

# For Nemotron RNNT stability: 80–120ms chunks work well
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0


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
    """
    Prints partials live. Prints finals + server metrics after pause.
    """
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

            # Print per-utterance metrics
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
    # unreachable


async def run_wav(ws, wav_path: str, realtime: bool):
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            await ws.send(data)
            if realtime:
                await asyncio.sleep(SLEEP_SEC)

    # Let server endpoint by sending a bit of silence, then EOS
    silence_frames = int(TARGET_SR * 0.8)
    await ws.send(b"\x00\x00" * silence_frames)
    await asyncio.sleep(0.8)
    await ws.send(b"")


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

    async def sender():
        while True:
            blk = await q.get()
            if blk is None:
                return
            try:
                await ws.send(blk.tobytes())
            except websockets.exceptions.ConnectionClosed:
                return

    send_task = asyncio.create_task(sender())

    try:
        while True:
            await asyncio.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

        # Send EOS so server can finalize if something is pending
        try:
            await ws.send(b"\x00\x00" * int(TARGET_SR * 0.8))
            await asyncio.sleep(0.8)
            await ws.send(b"")
        except Exception:
            pass

        await q.put(None)
        await send_task


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")
    p.add_argument("--wav", help="Path to wav file")
    p.add_argument("--mic", action="store_true", help="Use live microphone")
    p.add_argument("--fast", action="store_true", help="Disable realtime pacing (wav only)")
    args = p.parse_args()

    start = time.time()
    t_first_final = None

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

        finals = await recv_task

    total = time.time() - start
    print("\nFULL TRANSCRIPT:")
    print(" ".join([t for t in finals if t.strip()]))

    print("\nCLIENT METRICS:")
    print(f"Total wall time: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
