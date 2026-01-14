        key_phrases_file: null
        key_phrases_list: null
        context_score: 1.0
        depth_scaling: 2.0
        unk_score: 0.0
        final_eos_score: 1.0
        score_per_phrase: 0.0
        source_lang: en
        use_triton: true
        uniform_weights: false
        use_bpe_dropout: false
        num_of_transcriptions: 5
        bpe_alpha: 0.3
      boosting_tree_alpha: 0.0
      hat_subtract_ilm: false
      hat_ilm_weight: 0.0
      max_symbols_per_step: 10
      blank_lm_score_mode: LM_WEIGHTED_FULL
      pruning_mode: LATE
      allow_cuda_graphs: true
    temperature: 1.0
    durations: []
    big_blank_durations: []

INFO:asr_server:Loaded model=nvidia/nemotron-speech-streaming-en-0.6b in 27.16s on cuda
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8003 (Press CTRL+C to quit)



and my ws_client.py is -
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
FRAME_MS = 20
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)

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
                  f"preproc_ms={obj.get('preproc_ms')}",
                  f"infer_ms={obj.get('infer_ms')}",
                  f"flush_ms={obj.get('flush_ms')}",
                  )

async def run_wav(ws, wav_path: str, realtime: bool):
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(FRAME_SAMPLES)
            if not data:
                break
            await ws.send(data)
            if realtime:
                await asyncio.sleep(FRAME_MS / 1000.0)

    # send a bit of silence to ensure endpointing, then EOS
    await ws.send(b"\x00\x00" * int(TARGET_SR * 1.0))
    await asyncio.sleep(1.0)
    await ws.send(b"")

async def run_mic(ws):
    if not HAS_MIC:
        raise RuntimeError("sounddevice not installed. pip install sounddevice")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,   # ✅ 20ms blocks (server expects this)
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

        # Send EOS so server finalizes any pending utterance
        try:
            await ws.send(b"\x00\x00" * int(TARGET_SR * 1.0))
            await asyncio.sleep(1.0)
            await ws.send(b"")
        except Exception:
            pass

        await q.put(None)
        await send_task

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8003/ws/asr")
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

        finals = await recv_task

    total = time.time() - start
    print("\nFULL TRANSCRIPT:")
    print(" ".join([t for t in finals if t.strip()]))

    print("\nCLIENT METRICS:")
    print(f"Total wall time: {total:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
then why getting this 
(asr_env) PS C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts> python .\ws_client.py --mic --url ws://127.0.0.1:8003/ws/asr
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 185, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 149, in main
    async with websockets.connect(args.url, max_size=None) as ws:
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\asr_env\Lib\site-packages\websockets\asyncio\client.py", line 590, in __aenter__
    return await self
           ^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\asr_env\Lib\site-packages\websockets\asyncio\client.py", line 544, in __await_impl__
    self.connection = await self.create_connection()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\asr_env\Lib\site-packages\websockets\asyncio\client.py", line 470, in create_connection
    _, connection = await loop.create_connection(factory, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1166, in create_connection
    raise exceptions[0]
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1141, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
        exceptions, addrinfo, laddr_infos)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1044, in _connect_sock
    await self.sock_connect(sock, address)
  File "C:\Program Files\Python313\Lib\asyncio\proactor_events.py", line 726, in sock_connect
    return await self._proactor.connect(sock, address)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 804, in _poll
    value = callback(transferred, key, ov)
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 600, in finish_connect
    ov.getresult()
    ~~~~~~~~~~~~^^
ConnectionRefusedError: [WinError 1225] The remote computer refused the network connection
(asr_env) PS C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts>

