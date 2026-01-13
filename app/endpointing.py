class Endpointing:
    def __init__(self, end_silence_ms: int, min_utt_ms: int, max_utt_ms: int):
        self.end_silence_ms = end_silence_ms
        self.min_utt_ms = min_utt_ms
        self.max_utt_ms = max_utt_ms
        self.reset()

    def reset(self):
        self.silence_ms = 0

    def update(self, is_speech: bool, frame_ms: int, utt_audio_ms: int) -> bool:
        if is_speech:
            self.silence_ms = 0
        else:
            self.silence_ms += frame_ms

        if utt_audio_ms >= self.max_utt_ms:
            return True

        if utt_audio_ms >= self.min_utt_ms and self.silence_ms >= self.end_silence_ms:
            return True

        return False
