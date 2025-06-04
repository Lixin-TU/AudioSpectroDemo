"""
Mel-spectrogram helper for AudioSpectroDemo.

Handles long recordings by down-sampling to TARGET_SR and converts the
power-spectrogram to an 8-bit greyscale image for display.
"""

import numpy as np
import librosa

TARGET_SR = 32_552  # resample target
MIN_FREQ = 1        # Hz – matches GUI "Min Frequency"
MAX_FREQ = 16_276   # Hz – matches GUI "Max Frequency"
TOP_DB = 80         # dynamic‑range trimming in dB
GAIN_DB = 0        # extra gain applied after power‑to‑dB

def wav_to_mel_image(
    path: str,
    window: int = 2048,
    hop: int = 512,
    n_mels: int = 256,
    padding_factor: int = 2,
) -> np.ndarray:
    """
    Load a WAV file and return an 8‑bit greyscale *mel* spectrogram.

    Parameters
    ----------
    path : str
        Input .wav file.
    window : int  (default 2048)
        STFT window size (Hann).
    hop : int  (default 512)
        Hop length between frames.
    n_mels : int  (default 256)
        Number of mel bands.
    padding_factor : int
        Zero‑padding factor (e.g. 2 → FFT size = window * 2).

    Returns
    -------
    np.ndarray
        uint8 array (n_mels × frames) ready for display.
    """
    # Load and optionally resample
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    # Apply pre‑emphasis gain (allow clipping)
    gain_factor = 10 ** (GAIN_DB / 20)
    y = y * gain_factor  # clip is acceptable per user request

    fft_size = window * padding_factor

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=fft_size,      # zero‑padded FFT
        win_length=window,   # actual window size (Hann)
        hop_length=hop,
        window="hann",
        n_mels=n_mels,
        fmin=MIN_FREQ,
        fmax=MAX_FREQ,
        power=2.0,
    )

    S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)

    # Clip to the displayable dynamic range and scale to 0‑255.
    # Values above 0 dB are clipped (white); ‑TOP_DB maps to black.
    S_db = np.clip(S_db, -TOP_DB, 0)
    S_norm = (S_db + TOP_DB) / TOP_DB         # 0→black, 1→white
    return (S_norm * 255).astype(np.uint8)