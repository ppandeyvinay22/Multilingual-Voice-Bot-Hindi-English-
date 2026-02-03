from mic_input import MicInput
from vad import VADDetector
import numpy as np
import time

mic = MicInput()
vad = VADDetector()

mic.start()
print("Speak something... (10 seconds)")

audio_buffer = []

start = time.time()
while time.time() - start < 10:
    chunk = mic.read()
    if chunk is not None:
        audio_buffer.append(chunk)

        # Buffer small mic chunks into a longer continuous audio segment.
        # Silero VAD needs ~0.3–0.5s of audio context; single chunks (5–100 samples)
        # are too short to reliably detect speech.
        buffered_audio = np.concatenate(audio_buffer, axis=0)

        if buffered_audio.shape[0] >= 8000:  # ~0.5 sec at 16kHz
            if vad.is_speech(buffered_audio):
                print("🗣️ Speech detected")
                audio_buffer = []  # reset after detection

mic.stop()
