import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

class VADDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = load_silero_vad()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        audio_chunk: numpy array (frames, 1) or (frames,)
        returns: True if speech detected
        """
        if audio_chunk is None:
            return False

        if audio_chunk.ndim == 2:
            audio_chunk = audio_chunk[:, 0]

        audio_tensor = torch.from_numpy(audio_chunk).float()

        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=0.3,                 # 👈 lower sensitivity threshold
            min_speech_duration_ms=100,     # 👈 detect shorter speech
            min_silence_duration_ms=100
        )


        return len(timestamps) > 0
