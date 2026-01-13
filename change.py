(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx_asr_realtime# docker run --gpus all -p 8003:8003   -v /mnt/hf_cache:/srv/hf_cache   cx_asr_realtime

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:matplotlib.font_manager:generated new fontManager
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
[NeMo W 2026-01-13 23:25:26 <frozen importlib:241] Megatron num_microbatches_calculator not found, using Apex version.
WARNING:nv_one_logger.api.config:OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
INFO:nv_one_logger.exporter.export_config_manager:Final configuration contains 0 exporter(s)
WARNING:nv_one_logger.training_telemetry.api.training_telemetry_provider:No exporters were provided. This means that no telemetry data will be collected.
[NeMo I 2026-01-13 23:26:02 mixins:69] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo W 2026-01-13 23:26:02 rnnt_models:63] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    use_lhotse: true
    skip_missing_manifest_entries: true
    input_cfg: null
    tarred_audio_filepaths: null
    manifest_filepath: null
    sample_rate: 16000
    shuffle: true
    num_workers: 2
    pin_memory: true
    max_duration: 40.0
    min_duration: 0.1
    text_field: answer
    batch_duration: null
    use_bucketing: true
    max_tps: null
    bucket_duration_bins: null
    bucket_batch_size: null
    num_buckets: null
    bucket_buffer_size: null
    shuffle_buffer_size: null
    augmentor: null

[NeMo W 2026-01-13 23:26:02 rnnt_models:63] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    use_lhotse: true
    manifest_filepath: /data/ASR/en/librispeech/test-other.json
    sample_rate: 16000
    batch_size: 32
    shuffle: false
    max_duration: 40.0
    min_duration: 0.1
    num_workers: 2
    pin_memory: true
    text_field: answer
    tarred_audio_filepaths: null

[NeMo I 2026-01-13 23:26:07 rnnt_models:83] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-01-13 23:26:07 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-01-13 23:26:07 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-01-13 23:26:13 modelPT:501] Model EncDecRNNTBPEModel was successfully restored from /srv/hf_cache/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/e6a95c855dabb83223055f513d9f9fbce96d6970/nemotron-speech-streaming-en-0.6b.nemo.
[NeMo I 2026-01-13 23:26:14 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-01-13 23:26:14 nemotron_streaming:66] Changed decoding strategy to
    model_type: rnnt
    strategy: greedy
    compute_hypothesis_token_set: false
    preserve_alignments: null
    tdt_include_token_duration: null
    confidence_cfg:
      preserve_frame_confidence: false
      preserve_token_confidence: false
      preserve_word_confidence: false
      exclude_blank: true
      aggregation: min
      tdt_include_duration: false
      method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
    fused_batch_size: null
    compute_timestamps: null
    compute_langs: false
    word_seperator: ' '
    segment_seperators:
    - .
    - '!'
    - '?'
    segment_gap_threshold: null
    rnnt_timestamp_type: all
    greedy:
      max_symbols_per_step: 10
      preserve_alignments: false
      preserve_frame_confidence: false
      tdt_include_token_duration: false
      tdt_include_duration_confidence: false
      confidence_method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
      loop_labels: false
      use_cuda_graph_decoder: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
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
      max_symbols: 10
    beam:
      beam_size: 4
      search_type: default
      score_norm: true
      return_best_hypothesis: true
      tsd_max_sym_exp_per_step: 50
      alsd_max_target_len: 1.0
      nsc_max_timesteps_expansion: 1
      nsc_prefix_alpha: 1
      maes_num_steps: 2
      maes_prefix_alpha: 1
      maes_expansion_gamma: 2.3
      maes_expansion_beta: 2
      language_model: null
      softmax_temperature: 1.0
      preserve_alignments: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
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

ERROR:    Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 571, in __aenter__
    await self._router.startup()
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 671, in startup
    await handler()
  File "/srv/app/main.py", line 40, in startup
    load_sec = engine.load()
  File "/srv/app/nemotron_streaming.py", line 95, in load
    self._warmup()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/srv/app/nemotron_streaming.py", line 103, in _warmup
    _ = self.stream_transcribe(warm, reset_state=True, force_flush=True)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
