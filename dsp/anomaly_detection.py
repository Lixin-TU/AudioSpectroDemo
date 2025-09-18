"""
Anomaly Detection Processing Module

This module provides functionality to detect anomalous events in audio signals
using 2D autocorrelation analysis of mel-spectrograms.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import librosa

def process_anomaly_detection(file_path, mel_spectrogram=None):
    """
    Process anomaly detection on audio file using 2D autocorrelation analysis.
    
    Args:
        file_path: Path to the audio file
        mel_spectrogram: Optional pre-computed mel spectrogram
        
    Returns:
        dict: Results containing selected segments and visualization data
    """
    try:
        # Import shared constants for audio processing
        try:
            from .mel import TARGET_SR, MIN_FREQ, MAX_FREQ
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from dsp.mel import TARGET_SR, MIN_FREQ, MAX_FREQ

        # Always load waveform for ZCR analysis
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # Use provided spectrogram or generate from the loaded waveform
        if mel_spectrogram is not None:
            img = mel_spectrogram
            img_norm = img - img.min()
            if img_norm.max() > 0:
                img_norm = img_norm / img_norm.max()
            img = img_norm
            # Assume default parameters used elsewhere
            hop_length = 512
            n_fft = 2048
        else:
            n_mels = 256
            hop_length = 512
            n_fft = 2048

            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length, fmin=MIN_FREQ, fmax=MAX_FREQ
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = mel_spec_db
            img_norm = img - img.min()
            if img_norm.max() > 0:
                img_norm = img_norm / img_norm.max()
            img = img_norm

        # SEGMENT INTO 30 CLIPS WITH 50% OVERLAP, COMPUTE 2D AUTOCORRELATION
        segments = 30
        n_mels, n_frames = img.shape

        # Calculate segment width and step size for 50% overlap
        segment_width = int(n_frames / (segments / 2 + 0.5))
        step_size = segment_width // 2  # 50% overlap

        # Compute Zero-Crossing Rate (ZCR) over frames aligned with spectrogram
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]

        # Align ZCR length to spectrogram frame count
        if len(zcr) < n_frames:
            pad_len = n_frames - len(zcr)
            if len(zcr) > 0:
                zcr = np.pad(zcr, (0, pad_len), mode='edge')
            else:
                zcr = np.zeros(n_frames, dtype=float)
        elif len(zcr) > n_frames:
            zcr = zcr[:n_frames]

        scores = []
        segment_starts = []
        for i in range(segments):
            start = i * step_size
            if start + segment_width > n_frames:
                start = n_frames - segment_width
            segment_starts.append(start)

            seg_vals = zcr[start:start + segment_width]
            # Use mean ZCR within the segment as the anomaly score
            scores.append(float(np.mean(seg_vals)))

        # Selection logic: select top 12% segments by ZCR
        top_percent = 12
        percentile_threshold = np.percentile(scores, 100 - top_percent)

        selected_indices = [i for i, score in enumerate(scores) if score >= percentile_threshold]

        return {
            'status': 'success',
            'selected_indices': selected_indices,
            'scores': scores,
            'segment_starts': segment_starts,
            'segment_width': segment_width,
            'step_size': step_size,
            'percentile_threshold': percentile_threshold,
            'top_percent': top_percent,
            'n_mels': n_mels,
            'n_frames': n_frames,
            'spectrogram': img
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }