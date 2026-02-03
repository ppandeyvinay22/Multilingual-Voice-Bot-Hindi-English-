import time
import numpy as np

from audio.mic_input import MicInput
from audio.vad import VADDetector
from asr.whisper_asr import WhisperASR

mic = MicInput()
vad = VADDetector()
asr = WhisperASR()

mic.start()
print("🎙️ Speak something (Hindi + English). Pause to transcribe.")

audio_buffer = []
silence_start = None

while True:
    chunk = mic.read()
    if chunk is None:
        continue

    audio_buffer.append(chunk)
    buffered_audio = np.concatenate(audio_buffer, axis=0)

    if vad.is_speech(buffered_audio):
        silence_start = None
    else:
        if silence_start is None:
            silence_start = time.time()
        elif time.time() - silence_start > 0.4:
            print("… silence detected")
            print("⏳ Transcribing...")
            text = asr.transcribe(buffered_audio)
            print("📝 USER SAID:", text)
            audio_buffer = []
            silence_start = None
