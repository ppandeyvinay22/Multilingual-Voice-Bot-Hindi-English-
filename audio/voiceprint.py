import numpy as np


def _frame_audio(audio: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio[:, 0]
    if len(audio) < frame_size:
        pad = np.zeros(frame_size - len(audio), dtype=audio.dtype)
        audio = np.concatenate([audio, pad])
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop):
        frames.append(audio[start : start + frame_size])
    if not frames:
        frames = [audio[:frame_size]]
    return np.stack(frames, axis=0)


def extract_voiceprint(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return np.zeros(6, dtype=np.float32)

    frame_size = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    frames = _frame_audio(audio, frame_size, hop)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2.0

    # Spectral centroid + bandwidth
    window = np.hanning(frame_size).astype(frames.dtype)
    fft = np.fft.rfft(frames * window, axis=1)
    mag = np.abs(fft) + 1e-8
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    centroid = np.sum(freqs * mag, axis=1) / np.sum(mag, axis=1)
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * mag, axis=1) / np.sum(mag, axis=1))

    features = np.array(
        [
            np.mean(rms),
            np.std(rms),
            np.mean(zcr),
            np.std(zcr),
            np.mean(centroid),
            np.mean(bandwidth),
        ],
        dtype=np.float32,
    )
    return features


def voiceprint_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
