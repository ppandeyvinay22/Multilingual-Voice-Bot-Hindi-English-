from enum import Enum, auto
import time
import json


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    VERIFY_MOBILE = auto()
    VERIFY_SECONDARY = auto()
    VERIFIED = auto()
    VERIFY_FAILED = auto()


class ConversationStateMachine:
    def __init__(self):
        self.state = State.IDLE
        self.last_state_change = time.time()

    def transition_to(self, new_state: State):
        """
        Handle state transition with logging.
        """
        print(f"[STATE] {self.state.name} → {new_state.name}")
        self.state = new_state
        self.last_state_change = time.time()

    def on_start(self):
        """
        Called when the system starts.
        """
        self.transition_to(State.LISTENING)

    def on_user_finished_speaking(self):
        """
        Called when silence is detected after user speech.
        """
        if self.state in {
            State.LISTENING,
            State.VERIFY_MOBILE,
            State.VERIFY_SECONDARY,
            State.VERIFY_FAILED,
        }:
            self.transition_to(State.PROCESSING)

    def on_processing_done(self):
        """
        Called when ASR + LLM processing is finished.
        """
        if self.state == State.PROCESSING:
            self.transition_to(State.SPEAKING)

    def on_tts_finished(self):
        """
        Called when TTS playback finishes normally.
        """
        if self.state == State.SPEAKING:
            self.transition_to(State.LISTENING)

    def on_barge_in(self):
        """
        Called when user interrupts while bot is speaking.
        """
        if self.state == State.SPEAKING:
            print("[BARGE-IN] User interrupted bot speech")
            self.transition_to(State.LISTENING)

    def is_listening(self) -> bool:
        return self.state in {
            State.LISTENING,
            State.VERIFY_MOBILE,
            State.VERIFY_SECONDARY,
            State.VERIFY_FAILED,
        }

    def is_processing(self) -> bool:
        return self.state == State.PROCESSING

    def is_speaking(self) -> bool:
        return self.state == State.SPEAKING


def load_faq(path="logic/faq.json"):
    with open(path, "r") as f:
        return json.load(f)


def match_faq(text, faq_list):
    text = text.lower()
    for item in faq_list:
        for kw in item["keywords"]:
            if kw in text:
                return item["answer"]
    return None
