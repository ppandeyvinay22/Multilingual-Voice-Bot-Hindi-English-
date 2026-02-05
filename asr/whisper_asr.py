import numpy as np
from faster_whisper import WhisperModel

class WhisperASR:
    def __init__(self, model_size="medium", device="cpu"):
        """
        model_size: tiny | base | small | medium
        medium improves accuracy but is slower on CPU
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"
        )

    def transcribe(self, audio: np.ndarray, sample_rate=16000) -> str:
        """
        audio: numpy array of shape (n_samples,) or (n_samples, 1)
        returns: transcribed text
        """
        if audio.ndim == 2:
            audio = audio[:, 0]

        segments, _ = self.model.transcribe(
            audio,
            language=None,          # auto-detect Hindi / English
            vad_filter=False        # we already did VAD ourselves
        )

        text = ""
        for seg in segments:
            text += seg.text

        return text.strip()
