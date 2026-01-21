swapping model_name in config.py

pointing it to:

openai/whisper-large-v3-turbo

Nemotron streaming works well because it uses a native streaming architecture, not because of the model name alone.

Simply replacing the model name in the Nemotron config with Whisper Turbo or Riva 1B does not work and does not fix buffer or latency issues.

The buffering and latency problems in Whisper and Riva come from our custom batch / buffer logic, not from the ASR models themselves.

Instead of swapping model names, I am adding Whisper and Riva as separate ASR engines inside the same Nemotron realtime project.

The project will use a common realtime framework (WebSocket, VAD, endpointing, metrics) for all ASRs.

Nemotron will run as a true streaming engine, while Whisper and Riva will run as chunked engines within the same pipeline.

This allows us to test all ASRs under one clean framework, compare latency fairly, and incrementally remove buffer-related issues.

Latency reduction is the top priority; cost and accuracy comparisons will be done later once the pipeline is stable.

It cannot work, because the Nemotron code path is tightly coupled to Nemotron’s streaming encoder and decoder APIs.


Hi Kunal,

I tried the idea of directly swapping the model in config.py (Nemotron → Riva / Whisper Turbo) within the current realtime code, but it doesn’t work as a clean replacement.

What I faced when I tried

The existing realtime code is tightly coupled to Nemotron’s streaming behavior, not just the model name.

We rely on Nemotron-specific capabilities like:

incremental decoding via conformer_stream_step(),

encoder cache state across audio frames,

true partial results during streaming.

Other ASR systems (Riva non-streaming mode, Whisper Turbo, Azure, Google) don’t expose the same streaming + cache interface.

Because of this, changing only the model name breaks partial handling, TTFT behavior, and overall streaming flow.

Key realization

Nemotron is acting as a streaming engine, not just an ASR model.

Treating all ASRs as interchangeable “models” inside the same logic is what causes complexity and latency issues.

What we are doing now (better approach)

We are keeping the realtime server, VAD, endpointing, metrics, and WebSocket flow unchanged.

Inside the Nemotron project, we are introducing a pluggable ASR engine layer:

Nemotron runs as a true streaming ASR engine.

Whisper Turbo is plugged in as a chunked ASR engine (final-only).

Both engines implement the same high-level interface:

load()

new_session()

accept_pcm16()

finalize()

This lets us:

remove our custom catch/buffer/trigger logic,

reuse the same realtime pipeline,

and swap ASR backends cleanly without pretending they behave the same internally.

Why this works better

We eliminate the slow and error-prone manual buffering logic.

Nemotron handles realtime streaming properly.

Whisper Turbo still works in the same server, but in a mode that matches its actual capabilities.

The architecture stays extensible for Riva, Azure, and Google later, without rewriting core logic.

So instead of swapping models inside the same logic, we’re plugging different ASR engines into the Nemotron-based realtime framework, which keeps latency low and the codebase clean.
