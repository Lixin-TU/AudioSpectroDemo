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
        # Use provided spectrogram or generate from file
        if mel_spectrogram is not None:
            img = mel_spectrogram
            # Normalize to 0-1 range for processing
            img_norm = img - img.min()
            if img_norm.max() > 0:
                img_norm = img_norm / img_norm.max()
            img = img_norm
        else:
            # Load and process audio file - fix the import issue
            try:
                from .mel import TARGET_SR, MIN_FREQ, MAX_FREQ
            except ImportError:
                # Fallback for direct execution
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from dsp.mel import TARGET_SR, MIN_FREQ, MAX_FREQ
            
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            
            n_mels = 256
            hop_length = 512
            n_fft = 2048
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, 
                hop_length=hop_length, fmin=MIN_FREQ, fmax=MAX_FREQ
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = mel_spec_db
            # Normalize to 0-1 range
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

        # Parameters for central peak handling
        disable_center = False
        center_size = 1

        scores = []
        segment_starts = []
        for i in range(segments):
            start = i * step_size
            # Ensure we don't go beyond the image bounds
            if start + segment_width > n_frames:
                start = n_frames - segment_width
            
            segment_starts.append(start)
            clip = img[:, start:start + segment_width]
            
            # 2D autocorrelation via FFT
            F = np.fft.fft2(clip)
            ac = np.fft.ifft2(F * np.conj(F)).real
            ac = np.fft.fftshift(ac)
            
            ac_scored = ac.copy()
            
            # Optionally ignore the central peak region
            if disable_center:
                cy, cx = ac_scored.shape[0]//2, ac_scored.shape[1]//2
                half_size = center_size // 2
                ac_scored[cy-half_size:cy+half_size+1, cx-half_size:cx+half_size+1] = 0

            scores.append(ac_scored.max())

        # Selection logic: show top 12% only from segments with score >= 328
        top_percent = 12
        percentile_threshold = np.percentile(scores, 100 - top_percent)

        selected_indices = []
        for i, score in enumerate(scores):
            if score >= 328 and score > percentile_threshold:
                selected_indices.append(i)

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