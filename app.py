import time
from collections import deque
import numpy as np

from audio.mic_input import MicInput
from audio.vad import VADDetector
from asr.whisper_asr import WhisperASR
from logic.state_machine import ConversationStateMachine, State
from audio.tts import TextToSpeech
from logic.state_machine import match_faq, load_faq
from llm.llm_client import LLMClient
from logic.verify import load_users, extract_mobile, extract_last4, extract_dob, verify_user


def main():
    print("🚀 Starting Voice Bot")

    # Initialize core components
    mic = MicInput()
    vad = VADDetector()
    asr = WhisperASR()
    tts = TextToSpeech()
    sm = ConversationStateMachine()
    faq_list = load_faq()
    users = load_users()
    llm = LLMClient()
    response_text = None
    pending_mobile = None
    verify_attempts = 0
    next_state_after_speaking = None
    last_listen_state = None

    # Start system
    sm.on_start()
    response_text = "Welcome. Please tell me your mobile number."
    next_state_after_speaking = State.VERIFY_MOBILE
    sm.transition_to(State.SPEAKING)
    mic.start()

    audio_buffer = []
    vad_buffer = deque()
    vad_buffer_samples = 0
    speech_active = False
    last_speech_time = None
    recording = False
    recording_start_time = None

    SAMPLE_RATE = 16000
    VAD_WINDOW_SEC = 0.6
    VAD_MIN_SEC = 0.4
    MIN_UTTERANCE_SEC = 0.6
    SILENCE_SEC = 0.5
    MIN_RMS = 0.002
    MAX_UTTERANCE_SEC = 6.0

    VAD_WINDOW_SAMPLES = int(SAMPLE_RATE * VAD_WINDOW_SEC)
    VAD_MIN_SAMPLES = int(SAMPLE_RATE * VAD_MIN_SEC)
    MIN_UTTERANCE_SAMPLES = int(SAMPLE_RATE * MIN_UTTERANCE_SEC)
    latency_log = {}

    try:
        while True:
            time.sleep(0.01)
            chunk = mic.read()
            if chunk is None:
                continue

            # Always collect audio while listening
            if sm.is_listening():
                # Maintain a rolling buffer for VAD (~0.6s window)
                vad_buffer.append(chunk)
                vad_buffer_samples += chunk.shape[0]
                while vad_buffer_samples > VAD_WINDOW_SAMPLES and len(vad_buffer) > 1:
                    old = vad_buffer.popleft()
                    vad_buffer_samples -= old.shape[0]

                vad_speech = False
                if vad_buffer_samples >= VAD_MIN_SAMPLES:
                    vad_audio = np.concatenate(list(vad_buffer), axis=0)
                    vad_speech = vad.is_speech(vad_audio)

                if vad_speech:
                    speech_active = True
                    last_speech_time = time.time()
                    if not recording:
                        recording = True
                        recording_start_time = time.time()
                    audio_buffer.append(chunk)
                else:
                    if recording:
                        audio_buffer.append(chunk)
                    if speech_active and last_speech_time:
                        if time.time() - last_speech_time >= SILENCE_SEC:
                            buffered_audio = np.concatenate(audio_buffer, axis=0)
                            if buffered_audio.shape[0] >= MIN_UTTERANCE_SAMPLES:
                                rms = float(np.sqrt(np.mean(buffered_audio ** 2)))
                                if rms >= MIN_RMS:
                                    last_listen_state = sm.state
                                    sm.on_user_finished_speaking()
                                else:
                                    print("⚠️ Ignored low-energy (silence) audio")
                                    audio_buffer = []
                                    vad_buffer.clear()
                                    vad_buffer_samples = 0
                                    recording = False
                                    recording_start_time = None
                            speech_active = False
                            last_speech_time = None
                            if sm.is_listening():
                                recording = False
                                recording_start_time = None
                    if recording_start_time and (time.time() - recording_start_time >= MAX_UTTERANCE_SEC):
                        print("⚠️ Max utterance length reached, processing partial audio")
                        last_listen_state = sm.state
                        sm.on_user_finished_speaking()

            # PROCESSING state: run ASR once
            if sm.is_processing():
                buffered_audio = np.concatenate(audio_buffer, axis=0)

                print("⏳ ASR processing...")
                text = asr.transcribe(buffered_audio)
                print("📝 USER SAID:", text)

                latency_log["ASR_end_time"] = time.time()

                audio_buffer = []
                vad_buffer.clear()
                vad_buffer_samples = 0
                speech_active = False
                last_speech_time = None
                recording = False
                recording_start_time = None

                # 🔒 HARD FILTER
                if not text or len(text.strip()) < 4:
                    print("⚠️ Ignoring noise / short utterance")
                    sm.transition_to(last_listen_state or State.LISTENING)
                    continue

                # Optional: ignore common hallucinations
                noise_phrases = ["thank you", "you", "yeah", "okay", "hello"]
                if text.strip().lower() in noise_phrases:
                    print("⚠️ Ignoring hallucinated phrase")
                    sm.transition_to(last_listen_state or State.LISTENING)
                    continue

                latency_log["LLM_start_time"] = time.time()

                # Verification flow (voice-only)
                if last_listen_state in {State.VERIFY_MOBILE, State.VERIFY_FAILED}:
                    mobile = extract_mobile(text)
                    if mobile:
                        pending_mobile = mobile
                        response_text = "Thanks. Please tell me the last 4 digits or your date of birth."
                        next_state_after_speaking = State.VERIFY_SECONDARY
                    else:
                        response_text = "Sorry, I didn't catch your mobile number. Please repeat it."
                        next_state_after_speaking = State.VERIFY_MOBILE
                    sm.transition_to(State.SPEAKING)

                elif last_listen_state == State.VERIFY_SECONDARY:
                    last4 = extract_last4(text)
                    dob = extract_dob(text)
                    user = verify_user(users, pending_mobile or "", last4=last4, dob=dob)
                    if user:
                        response_text = "Verified. How can I help you today?"
                        next_state_after_speaking = State.LISTENING
                        verify_attempts = 0
                    else:
                        verify_attempts += 1
                        pending_mobile = None
                        if verify_attempts >= 2:
                            response_text = "Sorry, I couldn't verify. Please try again later."
                            next_state_after_speaking = State.VERIFY_FAILED
                        else:
                            response_text = "I couldn't verify that. Please tell me your mobile number again."
                            next_state_after_speaking = State.VERIFY_MOBILE
                    sm.transition_to(State.SPEAKING)

                else:
                    faq_answer = match_faq(text, faq_list)
                    if faq_answer:
                        response_text = faq_answer
                    else:
                        response_text = llm.generate(
                            text,
                            system_text=(
                                "You are a helpful insurance support assistant. "
                                "Reply in Hinglish (Hindi + English mix) with a natural, friendly tone. "
                                "Use small fillers like 'haan', 'hmm', 'theek hai', 'acha' to sound human. "
                                "If the user's question is unclear or incomplete, ask a brief clarifying question."
                            ),
                        )
                        if not response_text:
                            response_text = "Haan, main help kar sakta hoon. Thoda aur detail share karoge?"

                    sm.on_processing_done()

            # SPEAKING state
            if sm.is_speaking():
                if not response_text:
                    response_text = "Hmm... let me check that for you."
                print("🗣️ Bot speaking:", response_text)

                latency_log["TTS_start_time"] = time.time()
                latency_log["Audio_first_byte_time"] = time.time()

                tts.speak(response_text)

                time.sleep(0.2)  # minimal wait; return to listening quickly
                print("📊 Latency log:", latency_log)

                time.sleep(0.2)

                if next_state_after_speaking is not None:
                    sm.transition_to(next_state_after_speaking)
                    next_state_after_speaking = None
                else:
                    sm.on_tts_finished()
                response_text = None

    except KeyboardInterrupt:
        print("\n🛑 Stopping Voice Bot")
        mic.stop()


if __name__ == "__main__":
    main()
