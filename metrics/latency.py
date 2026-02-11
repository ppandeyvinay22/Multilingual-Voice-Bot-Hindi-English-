from __future__ import annotations

from statistics import mean


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * 0.95))
    return values[idx]


class LatencyTracker:
    def __init__(self) -> None:
        self.turn_latency_ms: list[float] = []
        self.asr_to_llm_ms: list[float] = []
        self.llm_to_tts_ms: list[float] = []
        self.tts_startup_ms: list[float] = []

    def record(self, log: dict[str, float]) -> dict[str, float] | None:
        required = {
            "USER_STOP_TIME",
            "ASR_end_time",
            "LLM_start_time",
            "TTS_start_time",
            "Audio_first_byte_time",
        }
        if not required.issubset(log.keys()):
            return None

        turn_ms = (log["TTS_start_time"] - log["USER_STOP_TIME"]) * 1000.0
        asr_to_llm_ms = (log["LLM_start_time"] - log["ASR_end_time"]) * 1000.0
        llm_to_tts_ms = (log["TTS_start_time"] - log["LLM_start_time"]) * 1000.0
        tts_startup_ms = (log["Audio_first_byte_time"] - log["TTS_start_time"]) * 1000.0

        self.turn_latency_ms.append(turn_ms)
        self.asr_to_llm_ms.append(asr_to_llm_ms)
        self.llm_to_tts_ms.append(llm_to_tts_ms)
        self.tts_startup_ms.append(tts_startup_ms)

        return {
            "turn_ms": turn_ms,
            "asr_to_llm_ms": asr_to_llm_ms,
            "llm_to_tts_ms": llm_to_tts_ms,
            "tts_startup_ms": tts_startup_ms,
        }

    def summary(self) -> dict[str, float]:
        return {
            "turn_avg_ms": mean(self.turn_latency_ms) if self.turn_latency_ms else 0.0,
            "turn_p95_ms": _p95(self.turn_latency_ms),
            "asr_to_llm_avg_ms": mean(self.asr_to_llm_ms) if self.asr_to_llm_ms else 0.0,
            "llm_to_tts_avg_ms": mean(self.llm_to_tts_ms) if self.llm_to_tts_ms else 0.0,
            "tts_startup_avg_ms": mean(self.tts_startup_ms) if self.tts_startup_ms else 0.0,
            "turn_count": float(len(self.turn_latency_ms)),
        }
