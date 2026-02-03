import time
import numpy as np

from audio.mic_input import MicInput
from audio.vad import VADDetector
from asr.whisper_asr import WhisperASR
from logic.state_machine import ConversationStateMachine, State
from audio.tts import TextToSpeech
from logic.state_machine import match_faq, load_faq

def main():
    print("🚀 Starting Voice Bot")

    # Initialize core components
    mic = MicInput()
    vad = VADDetector()
    asr = WhisperASR()
    tts = TextToSpeech()
    sm = ConversationStateMachine()

    # Start system
    sm.on_start()
    mic.start()

    audio_buffer = []
    silence_start = None

    MIN_SAMPLES = 8000      # ~0.5 sec audio
    SILENCE_SEC = 0.7
    latency_log = {}


    try:
        while True:
            time.sleep(0.01)
            chunk = mic.read()
            if chunk is None:
                continue

            # Always collect audio while listening
            if sm.is_listening():
                audio_buffer.append(chunk)

                # Check if user is speaking RIGHT NOW
                if vad.is_speech(chunk):
                    silence_start = None
                else:
                    if len(audio_buffer) > 0:
                        buffered_audio = np.concatenate(audio_buffer, axis=0)

                        if buffered_audio.shape[0] >= MIN_SAMPLES:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start >= SILENCE_SEC:
                                # User finished speaking
                                sm.on_user_finished_speaking()

            # PROCESSING state: run ASR once
            if sm.is_processing():
                buffered_audio = np.concatenate(audio_buffer, axis=0)

                print("⏳ ASR processing...")
                text = asr.transcribe(buffered_audio)
                print("📝 USER SAID:", text)
                
                latency_log["ASR_end_time"] = time.time()

                audio_buffer = []
                silence_start = None

                # 🔒 HARD FILTER (THIS STOPS THE CHAOS)
                if not text or len(text.strip()) < 4:
                    print("⚠️ Ignoring noise / short utterance")
                    sm.transition_to(State.LISTENING)
                    continue

                # Optional: ignore common hallucinations
                noise_phrases = ["thank you", "you", "yeah", "okay", "hello"]
                if text.strip().lower() in noise_phrases:
                    print("⚠️ Ignoring hallucinated phrase")
                    sm.transition_to(State.LISTENING)
                    continue
                
                
                latency_log["LLM_start_time"] = time.time()
                sm.on_processing_done()


            # SPEAKING state (placeholder for now)
            if sm.is_speaking():
                response_text = "Hmm... let me check that for you."
                print("🗣️ Bot speaking:", response_text)

                latency_log["TTS_start_time"] = time.time()

                # Approximate: treat TTS call time as audio start (demo-acceptable)
                latency_log["Audio_first_byte_time"] = time.time()

                tts.speak(response_text)


                time.sleep(2)  # fixed demo-safe speaking window
                tts.stop()
                print("📊 Latency log:", latency_log)

                time.sleep(0.2)

                sm.on_tts_finished()



    except KeyboardInterrupt:
        print("\n🛑 Stopping Voice Bot")
        mic.stop()


if __name__ == "__main__":
    main()
