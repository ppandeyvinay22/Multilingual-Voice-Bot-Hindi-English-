import sounddevice as sd
import queue
import numpy as np

class MicInput:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            print("Mic status:", status)
        self.audio_queue.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()
        print("🎙️ Microphone started")

    def stop(self):
        self.stream.stop()
        self.stream.close()
        print("🛑 Microphone stopped")

    def read(self):
        """Get next audio chunk"""
        try:
            return self.audio_queue.get(timeout=1)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Drop stale audio chunks to keep turn alignment tight."""
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

# Example usage
if __name__ == "__main__":
    mic = MicInput()
    mic.start()

    print("Speak for 5 seconds...")
    for _ in range(50):
        chunk = mic.read()
        if chunk is not None:
            print("Audio chunk shape:", chunk.shape)

    mic.stop()
