import asyncio, json, sounddevice as sd, websockets

SR = 16000
BLOCK_MS = 120
BLOCK = int(SR * BLOCK_MS / 1000)

async def main(url):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"mode":"mic"}))
        print("🎤 Speak...")

        def cb(indata, frames, t, status):
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.create_task, ws.send(indata.tobytes())
            )

        with sd.InputStream(samplerate=SR, channels=1,
                            dtype="int16", blocksize=BLOCK, callback=cb):
            async for msg in ws:
                print(msg)

asyncio.run(main("ws://127.0.0.1:8080/ws/asr"))
