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
