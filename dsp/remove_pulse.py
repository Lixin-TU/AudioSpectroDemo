"""
Remove Pulse Processing Module

This module provides functionality to extract harmonic components
from audio signals using librosa, effectively removing percussive artifacts.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


def process_remove_pulse(audio_file_path: str) -> dict:
    """
    Main function to process audio file and extract harmonic components.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: Processing results including processed audio
    """
    print(f"Processing Harmonic Extraction for: {audio_file_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        
        # Extract harmonic components (removes percussive/pulse artifacts)
        y_harmonic = librosa.effects.harmonic(y)
        
        result = {
            'status': 'success',
            'processed_audio': y_harmonic,
            'original_audio': y,
            'sample_rate': sr,
            'original_length': len(y),
            'processed_length': len(y_harmonic),
            'processing_method': 'librosa_harmonic'
        }
        
        print(f"Harmonic extraction completed")
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def calculate_improvement(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate improvement ratio between original and processed audio.
    
    Args:
        original: Original audio signal
        processed: Processed audio signal
        
    Returns:
        float: Improvement ratio (higher is better)
    """
    # Calculate signal-to-noise ratio improvement
    original_variance = np.var(original)
    processed_variance = np.var(processed)
    
    if processed_variance > 0:
        improvement = original_variance / processed_variance
    else:
        improvement = 1.0
    
    return improvement


def calculate_energy_reduction(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate the percentage of energy reduced by harmonic extraction.
    
    Args:
        original: Original audio signal
        processed: Processed audio signal
        
    Returns:
        float: Percentage of energy reduced
    """
    original_energy = np.sum(original ** 2)
    processed_energy = np.sum(processed ** 2)
    
    if original_energy > 0:
        reduction = ((original_energy - processed_energy) / original_energy) * 100
        return max(0, reduction)  # Ensure non-negative
    else:
        return 0.0


def apply_harmonic_extraction_with_params(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """
    Apply harmonic extraction with custom parameters.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        **kwargs: Additional parameters for harmonic extraction
        
    Returns:
        np.ndarray: Processed audio with harmonic components
    """
    # Extract parameters with defaults
    margin = kwargs.get('margin', 1.0)
    
    # Apply harmonic extraction with custom margin if specified
    try:
        if margin != 1.0:
            # Use librosa's harmonic-percussive separation with custom margin
            y_harmonic, _ = librosa.effects.hpss(audio, margin=margin)
        else:
            # Use default harmonic extraction
            y_harmonic = librosa.effects.harmonic(audio)
        
        return y_harmonic
    except Exception:
        # Fallback to default method if custom parameters fail
        return librosa.effects.harmonic(audio)


# Configuration parameters that can be adjusted
HARMONIC_EXTRACTION_CONFIG = {
    'margin': 1.0,                    # Margin for harmonic/percussive separation
    'kernel_size': 31,                # Kernel size for median filtering
    'power': 2.0,                     # Power for spectrogram computation
}

PULSE_DETECTION_CONFIG = {
    'threshold_factor': 3.0,        # Sensitivity of pulse detection
    'frame_length_ms': 10,          # Analysis frame length in milliseconds
    'pulse_duration_ms': 50,        # Expected pulse duration in milliseconds
    'interpolation_method': 'linear' # Method for replacing pulses
}


if __name__ == "__main__":
    # Test the module
    test_file = "test_audio.wav"  # Replace with actual test file
    result = process_remove_pulse(test_file)
    print("Test result:", result)
