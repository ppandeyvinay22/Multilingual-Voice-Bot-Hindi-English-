# Multilingual Voice Bot (Hindi + English)

This project is a real-time voice bot prototype for insurance support. It supports Hindi/English code-mixed speech, user verification, FAQ lookup, and LLM fallback.

## 1. Objective
Build a working voice conversation loop where:
- user speaks naturally (Hindi/English/Hinglish)
- bot verifies identity (JSON-based verification)
- bot answers FAQs or falls back to an LLM
- latency is logged for ASR/LLM/TTS stages

## 2. Current End-to-End Flow (Exact Runtime Flow)
The main flow is implemented in `app.py`.

1. App starts and initializes:
- microphone stream (`audio/mic_input.py`)
- VAD detector (`audio/vad.py`)
- ASR model (`asr/whisper_asr.py`)
- TTS engine (`audio/tts.py`)
- state machine (`logic/state_machine.py`)
- FAQ + users (`logic/faq.json`, `logic/users.json`)
- LLM client (`llm/llm_client.py`)

2. Bot prompt at start:
- speaks: `Welcome. Please tell me your mobile number.`
- state moves to `VERIFY_MOBILE`

3. Listening and speech detection:
- audio chunks are buffered continuously
- Silero VAD decides speech/non-speech
- once silence is detected for `SILENCE_SEC`, utterance is sent to ASR

4. Verification stage:
- `VERIFY_MOBILE`: extract 10-digit mobile from ASR text
- `VERIFY_SECONDARY`: extract either last 4 digits or DOB
- match against `logic/users.json`
- success: `Verified. How can I help you today?`
- failure: retry; after retry limit, move to `VERIFY_FAILED`

5. Question answering stage:
- if user query matches FAQ keywords -> return canned FAQ answer
- else -> use LLM (`GEMINI_API_KEY`) for Hinglish response
- if LLM unavailable/fails -> fallback safe default reply
- filler words are applied only for non-sensitive responses

6. Response playback:
- TTS speaks response
- if user interrupts during bot speech, barge-in stops TTS and returns to listening immediately
- state returns to listening state for next turn

## 3. State Machine Design
Implemented in `logic/state_machine.py`.

Active states:
- `IDLE`
- `LISTENING`
- `PROCESSING`
- `SPEAKING`
- `VERIFY_MOBILE`
- `VERIFY_SECONDARY`
- `VERIFY_FAILED`
- `VERIFIED` (declared; currently conversational loop continues via `LISTENING`)

Transition model:
- `LISTENING/VERIFY_* -> PROCESSING` after speech completion
- `PROCESSING -> SPEAKING` after ASR + routing
- `SPEAKING -> next target state` based on response context

## 4. Verification Logic
Implemented in `logic/verify.py`.

Supported extraction:
- `extract_mobile(text)`
- `extract_last4(text)`
- `extract_dob(text)`
- `verify_user(users, mobile, last4, dob)`

Robust parsing features:
- numeric cleanup from text
- digit-word normalization for spoken numbers:
  - `"nine eight seven..."` -> `987...`
- DOB accepts:
  - `DD-MM-YYYY`
  - `YYYY-MM-DD`
  - `DD/MM/YYYY`
  - `YYYY/MM/DD`
  - `DDMMYYYY`

User DB currently includes 5 dummy users in `logic/users.json`.

## 5. FAQ + LLM Fallback Routing
- FAQ data: `logic/faq.json`
- keyword match function: `match_faq()` in `logic/state_machine.py`
- LLM client: `llm/llm_client.py`
- FAQ count: `11` multilingual/codemixed entries

Routing order:
1. Try FAQ keyword hit
2. If no hit, query LLM
3. If LLM not configured/fails, return deterministic fallback
4. Add controlled filler only if response is non-sensitive (no OTP/mobile/DOB/last4 prompts)

## 6. Audio Stack and Why These Modules Were Used

### 6.1 Mic Capture: `sounddevice`
- module: `audio/mic_input.py`
- reason: lightweight real-time stream callback, no cloud dependency

### 6.2 Voice Activity Detection: `silero_vad`
- module: `audio/vad.py`
- reason: better speech segmentation than raw RMS-only gating
- tuned for faster triggering (`threshold=0.3`, short min durations)

### 6.3 ASR: `faster-whisper` (Whisper)
- module: `asr/whisper_asr.py`
- current model: `medium` on CPU, `int8`
- reason: multilingual robustness and better Hinglish handling than many narrow Hindi-only models
- tradeoff: improved accuracy but higher latency on CPU

### 6.4 TTS: `pyttsx3`
- module: `audio/tts.py`
- reason for choosing `pyttsx3` over cloud TTS engines:
  - fully offline
  - zero API/network dependency
  - stable for local interview demos
- tradeoff:
  - voice naturalness is lower than cloud neural TTS
  - device-specific voice quality

## 7. Latency Metrics (What Is Logged)
In `app.py`, these timestamps are logged per utterance:
- `USER_STOP_TIME`
- `ASR_end_time`
- `LLM_start_time`
- `TTS_start_time`
- `Audio_first_byte_time`

Derived metrics (implemented in `metrics/latency.py`):
- `turn_ms = (TTS_start_time - USER_STOP_TIME) * 1000`
- `asr_to_llm_ms = (LLM_start_time - ASR_end_time) * 1000`
- `llm_to_tts_ms = (TTS_start_time - LLM_start_time) * 1000`
- `tts_startup_ms = (Audio_first_byte_time - TTS_start_time) * 1000`

Current status:
- per-turn metrics are printed in runtime logs
- aggregate metrics are printed in runtime logs:
  - `avg` turn latency
  - `p95` turn latency
- `Audio_first_byte_time` prefers actual `pyttsx3` start callback timing (`started-utterance`), with fallback to current time if callback is unavailable

## 8. Challenges Faced and Practical Decisions

1. Bot sometimes captured its own speech (echo/self-hearing)
- reason: speaker output leaked into mic stream
- mitigation used: stricter filtering, short utterance windows, quick turn handling
- stronger fix for production: WebRTC AEC/NS/AGC pipeline

2. Mobile number recognition failures
- reason: ASR may output spoken words (`nine`, `oh`) not digits
- mitigation used: digit-word normalization in `logic/verify.py`

3. Latency spikes
- reason: CPU inference cost (`medium` ASR model)
- mitigation used: reduced silence wait, integrated latency stage tracking, and aggregate monitoring (`avg`/`p95`)

4. Offline requirement vs voice quality
- chosen: `pyttsx3` for predictable local demo
- tradeoff accepted: less natural speech than cloud neural voices

5. Barge-in behavior
- requirement: stop speaking and listen immediately when user interrupts
- implementation: while in `SPEAKING`, mic is monitored with VAD; on speech, `tts.stop()` is called and state transitions back to listening target
- tradeoff: no full acoustic echo cancellation pipeline yet

## 9. Demo Script (Interview Safe)

Use this sequence:
1. Start app: `python app.py`
2. Say mobile from `logic/users.json`, for example:
- `nine zero two six one nine eight two two five`
3. Say secondary factor:
- last4: `three three four four`
- or DOB: `14-08-2000`
4. Ask code-mixed questions:
- `Hi, mera policy status check kar do`
- `premium due kab hai?`
- `claim status batao`

## 10. Environment and Config
Set API key in `.env` for LLM fallback:
- `GEMINI_API_KEY=<your_key>`
- optional: `GEMINI_MODEL`, `LLM_TIMEOUT_SEC`, `GEMINI_BASE_URL`

Without API key:
- bot still runs
- FAQ and deterministic fallback still work

## 11. Known Limitations (Current Build)
- no persistent conversation memory layer
- no AEC/NS/AGC stack yet (barge-in works, but echo-heavy environments can still degrade it)
- no streaming ASR partial transcript response
- `requirements.txt` is currently empty and should be finalized for reproducible setup

## 12. What Could Be Improved Next
1. Integrate WebRTC audio processing (AEC + NS + AGC)
2. Add stage-wise `p95` (not only overall turn `p95`)
3. Add streaming ASR partials for lower perceived latency
4. Replace offline TTS with neural TTS for better human-like output
5. Add test coverage for parser edge cases and state transitions

## 13. Repository Map
- `app.py`: main orchestration loop
- `audio/mic_input.py`: microphone stream
- `audio/vad.py`: speech detection
- `audio/tts.py`: TTS wrapper
- `asr/whisper_asr.py`: speech-to-text
- `logic/state_machine.py`: conversation state machine + FAQ matcher
- `logic/verify.py`: verification parsing and validation
- `logic/faq.json`: FAQ data
- `logic/users.json`: user DB for verification
- `llm/llm_client.py`: Gemini API client
- `metrics/latency.py`: runtime latency tracker with per-turn and aggregate (`avg`/`p95`) reporting
