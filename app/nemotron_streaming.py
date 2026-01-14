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
    """

    def __init__(self, model_name: str, device: str, sample_rate: int, context_right: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.context_right = context_right
        self.model = None

        self.shift_frames = 0
        self.pre_cache_frames = 0
        self.drop_extra = 0

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        if self.model_name.endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(self.model_name, map_location="cpu")
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location="cpu")

        self.model = self.model.cuda() if self.device == "cuda" else self.model.cpu()

        self.model.encoder.set_default_att_context_size([70, int(self.context_right)])

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

        scfg = self.model.encoder.streaming_cfg
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, (list, tuple)) else scfg.shift_size
        pre_cache = scfg.pre_encode_cache_size
        self.pre_cache_frames = pre_cache[1] if isinstance(pre_cache, (list, tuple)) else pre_cache
        self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))

        self._warmup()
        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        warm = np.zeros(int(self.sr * 1.0), dtype=np.float32)
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        cache = (cache[0], cache[1], cache[2])
        _ = self.stream_transcribe(warm, cache, None, None, 0, force_flush=True)

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32: np.ndarray,
        cache,
        prev_hyp,
        prev_pred_out,
        emitted_frames,
        force_flush: bool = False,
    ):
        timings = StreamTimings()

        t0 = time.perf_counter()
        audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0)
        if self.device == "cuda":
            audio_tensor = audio_tensor.cuda()
        audio_len = torch.tensor([len(audio_f32)], device=audio_tensor.device)
        mel, _ = self.model.preprocessor(audio_tensor, audio_len)
        timings.preproc_sec += time.perf_counter() - t0

        available = int(mel.shape[-1]) - 1
        if available <= 0:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        if (available - emitted_frames) < self.shift_frames and not force_flush:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        chunk_start = max(0, emitted_frames - self.pre_cache_frames)
        chunk_end = min(emitted_frames + self.shift_frames, available)
        chunk = mel[:, :, chunk_start:chunk_end]
        chunk_len = torch.tensor([chunk.shape[-1]], device=chunk.device)

        t1 = time.perf_counter()
        prev_pred_out, texts, c0, c1, c2, prev_hyp = self.model.conformer_stream_step(
            chunk,
            chunk_len,
            cache[0],
            cache[1],
            cache[2],
            keep_all_outputs=False,
            previous_hypotheses=prev_hyp,
            previous_pred_out=prev_pred_out,
            drop_extra_pre_encoded=self.drop_extra,
            return_transcription=True,
        )
        timings.infer_sec += time.perf_counter() - t1

        cache = (c0, c1, c2)
        emitted_frames = min(emitted_frames + self.shift_frames, available)

        text = safe_text(texts[0]) if texts else ""
        return text.strip(), cache, prev_hyp, prev_pred_out, emitted_frames, timings


class StreamingSession:
    """
    Per-utterance streaming state.
    METRICS FIXED HERE.
    """

    def __init__(self, engine: NemotronStreamingASR, max_buffer_ms: int):
        self.engine = engine
        self.max_samples = int(engine.sr * max_buffer_ms / 1000)
        self.reset_stream_state()

    def reset_stream_state(self):
        cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = (cache[0], cache[1], cache[2])
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0
        self.audio = np.array([], dtype=np.float32)
        self.current_text = ""

        # ✅ METRICS
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.compute_sec_total = 0.0
        self.chunks = 0

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])
        if len(self.audio) > self.max_samples:
            self.audio = self.audio[-self.max_samples:]

    def step_if_ready(self) -> Optional[str]:
        t0 = time.perf_counter()
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = \
            self.engine.stream_transcribe(
                self.audio, self.cache, self.prev_hyp,
                self.prev_pred, self.emitted_frames, False
            )
        wall = time.perf_counter() - t0

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.compute_sec_total += wall
        self.chunks += 1

        if not text or text == self.current_text:
            return None

        self.current_text = text
        return text

    def finalize(self, pad_ms: int) -> str:
        pad = np.zeros(int(self.engine.sr * pad_ms / 1000), dtype=np.float32)
        self.audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = \
            self.engine.stream_transcribe(
                self.audio, self.cache, self.prev_hyp,
                self.prev_pred, self.emitted_frames, True
            )
        wall = time.perf_counter() - t0

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.utt_flush += wall
        self.compute_sec_total += wall
        self.chunks += 1

        if text:
            self.current_text = text

        final = self.current_text.strip()
        self.reset_stream_state()
        return final
