import sys
import os

# add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np

from audio.mic_input import MicInput
from audio.vad import VADDetector
from asr.whisper_asr import WhisperASR

print("⏳ Loading Whisper model (this can take time)...")
asr = WhisperASR()
print("✅ Whisper loaded")

mic = MicInput()
vad = VADDetector()

mic.start()

audio_buffer = []
silence_start = None
MIN_SAMPLES = 8000        # ~0.5 sec
SILENCE_SEC = 0.5

try:
    while True:
        chunk = mic.read()
        if chunk is None:
            continue

        audio_buffer.append(chunk)
        buffered_audio = np.concatenate(audio_buffer, axis=0)

        # If speech is present, reset silence timer
        if vad.is_speech(chunk):
            silence_start = None
        else:
            if buffered_audio.shape[0] >= MIN_SAMPLES:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_SEC:
                    # extra guard: check audio energy
                    rms = np.sqrt(np.mean(buffered_audio**2))

                    if rms > 0.002:   # speech energy threshold
                        print("⏳ Transcribing utterance...")
                        audio = buffered_audio / (np.max(np.abs(buffered_audio)) + 1e-8)
                        text = asr.transcribe(audio)
                        print("📝 USER SAID:", text)
                    else:
                        print("⚠️ Ignored low-energy (silence) audio")

                    audio_buffer = []
                    silence_start = None

except KeyboardInterrupt:
    mic.stop()
    print("\n🛑 Stopped")
