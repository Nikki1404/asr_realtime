import time
from dataclasses import dataclass
from typing import Optional, Any, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def safe_text(h: Any) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h
    if hasattr(h, "text"):
        try:
            return h.text or ""
        except Exception:
            return ""
    return str(h)


@dataclass
class StreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class NemotronStreamingASR:
    """
    True streaming ASR using NeMo conformer_stream_step().
    No repeated model.transcribe() calls.
    """

    def __init__(self, model_name: str, device: str, sample_rate: int, context_right: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.context_right = context_right

        self.model = None

        # streaming params (set at load)
        self.shift_frames: int = 0
        self.pre_cache_frames: int = 0
        self.hop_samples: int = 0
        self.drop_extra: int = 0

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        # HF model name or local .nemo
        if self.model_name.endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(self.model_name, map_location="cpu")
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location="cpu")

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        # streaming attention context
        self.model.encoder.set_default_att_context_size([70, int(self.context_right)])

        # IMPORTANT: disable cuda-graph decoder (avoids CUDA failure 35 spam & fallback)
        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                }
            })
        )

        self.model.eval()
        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except Exception:
            pass

        # streaming cfg
        scfg = self.model.encoder.streaming_cfg
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, (list, tuple)) else scfg.shift_size
        pre_cache = scfg.pre_encode_cache_size
        self.pre_cache_frames = pre_cache[1] if isinstance(pre_cache, (list, tuple)) else pre_cache
        self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))

        # hop size in samples
        hop_sec = float(self.model.cfg.preprocessor.get("window_stride", 0.01))
        self.hop_samples = int(hop_sec * self.sr)

        # warmup streaming kernels
        self._warmup()

        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        """
        Warmup with 1s silence using the SAME stream_transcribe path.
        Must build initial cache and call stream_transcribe correctly.
        """
        warm = np.zeros(int(self.sr * 1.0), dtype=np.float32)

        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        cache_tup = (cache[0], cache[1], cache[2])

        prev_hyp = None
        prev_pred = None
        emitted_frames = 0

        # force flush once to compile kernels
        _txt, _cache_tup, _prev_hyp, _prev_pred, _emitted_frames, _tim = self.stream_transcribe(
            audio_f32=warm,
            cache=cache_tup,
            prev_hyp=prev_hyp,
            prev_pred_out=prev_pred,
            emitted_frames=emitted_frames,
            force_flush=True,
        )

    def new_session(self, max_buffer_ms: int):
        return StreamingSession(self, max_buffer_ms=max_buffer_ms)

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32: np.ndarray,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prev_hyp: Any,
        prev_pred_out: Any,
        emitted_frames: int,
        force_flush: bool = False,
    ):
        """
        Process enough frames for ONE streaming step (shift_frames),
        or flush if force_flush=True.
        """
        assert self.model is not None
        timings = StreamTimings()

        # preprocess current buffer
        t0 = time.perf_counter()
        audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0)
        if self.device == "cuda":
            audio_tensor = audio_tensor.cuda()
        audio_len = torch.tensor([len(audio_f32)], device=audio_tensor.device)
        mel, mel_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
        timings.preproc_sec += (time.perf_counter() - t0)

        available = int(mel.shape[-1]) - 1
        if available <= 0:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        need_frames = self.shift_frames
        enough = (available - emitted_frames) >= need_frames

        if (not enough) and (not force_flush):
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        if emitted_frames == 0:
            chunk_start = 0
            chunk_end = min(self.shift_frames, available)
            drop_extra = 0
        else:
            chunk_start = max(0, emitted_frames - self.pre_cache_frames)
            chunk_end = min(emitted_frames + self.shift_frames, available)
            drop_extra = self.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=chunk_mel.device)

        t1 = time.perf_counter()
        (prev_pred_out, texts, cache0, cache1, cache2, prev_hyp) = self.model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=False,
            previous_hypotheses=prev_hyp,
            previous_pred_out=prev_pred_out,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )
        timings.infer_sec += (time.perf_counter() - t1)

        new_cache = (cache0, cache1, cache2)

        if emitted_frames < available:
            emitted_frames = min(emitted_frames + self.shift_frames, available)

        text = ""
        if texts and texts[0] is not None:
            text = safe_text(texts[0]).strip()

        return text, new_cache, prev_hyp, prev_pred_out, emitted_frames, timings


class StreamingSession:
    """
    Holds per-websocket streaming state.
    Maintains ring buffer so preprocessing cost doesn't grow unbounded.
    """

    def __init__(self, engine: NemotronStreamingASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        self.audio = np.array([], dtype=np.float32)

        self.cache = None
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""

        # timings (per utterance)
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0

        self.chunks = 0

        self.reset_stream_state()

    def reset_stream_state(self):
        cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = (cache[0], cache[1], cache[2])
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""
        self.audio = np.array([], dtype=np.float32)

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])
        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]

    def backlog_ms(self) -> int:
        return int(1000 * (len(self.audio) / self.engine.sr))

    def step_if_ready(self) -> Optional[str]:
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=False,
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        if text is None or text == "" or text == self.current_text:
            return None

        self.current_text = text
        self.chunks += 1
        return text

    def finalize(self, pad_ms: int) -> str:
        pad = np.zeros(int(self.engine.sr * (pad_ms / 1000.0)), dtype=np.float32)
        self.audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=True,
        )
        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.utt_flush += (time.perf_counter() - t0)

        if text:
            self.current_text = text.strip()

        final = self.current_text.strip()

        # reset for next utterance
        self.reset_stream_state()
        return final
