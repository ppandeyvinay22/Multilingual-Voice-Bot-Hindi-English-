import threading
import time
import pyttsx3


class TextToSpeech:
    def __init__(self):
        print("🔊 Initializing TTS (pyttsx3)")
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)  # natural speed
        self._thread = None
        self._stop_flag = False
        self.last_start_time = None
        self._bind_callbacks()

    def _bind_callbacks(self):
        def on_start(_name):
            self.last_start_time = time.time()

        # Use engine callbacks to capture actual speech start timing.
        self.engine.connect("started-utterance", on_start)

    def speak(self, text: str):
        self._stop_flag = False
        self.last_start_time = None
        self._thread = threading.Thread(
            target=self._run,
            args=(text,),
            daemon=True
        )
        self._thread.start()

    def _run(self, text: str):
        if self._stop_flag:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        print("🛑 Stopping TTS")
        self._stop_flag = True
        self.engine.stop()
