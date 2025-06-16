"""
Remove Pulse Processing Module

This module provides functionality to detect and remove pulse artifacts
from audio signals, particularly useful for underwater or environmental recordings.
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Tuple, Optional


def process_remove_pulse(audio_file_path: str) -> dict:
    """
    Main function to process audio file and remove pulse artifacts.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: Processing results and statistics
    """
    print(f"Processing Remove Pulse for: {audio_file_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        
        # Detect pulse artifacts
        pulse_locations = detect_pulse_artifacts(y, sr)
        
        # Remove detected pulses
        cleaned_audio = remove_pulses(y, pulse_locations)
        
        # Calculate statistics
        pulses_removed = len(pulse_locations)
        improvement_ratio = calculate_improvement(y, cleaned_audio)
        
        result = {
            'status': 'success',
            'pulses_detected': pulses_removed,
            'improvement_ratio': improvement_ratio,
            'original_length': len(y),
            'processed_length': len(cleaned_audio),
            'sample_rate': sr
        }
        
        print(f"Remove Pulse completed: {pulses_removed} pulses removed")
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def detect_pulse_artifacts(audio: np.ndarray, sample_rate: int, 
                          threshold_factor: float = 3.0) -> list:
    """
    Detect pulse artifacts in audio signal.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of audio
        threshold_factor: Multiplier for detection threshold
        
    Returns:
        list: List of (start, end) tuples for detected pulses
    """
    # Calculate short-time energy
    frame_length = int(0.01 * sample_rate)  # 10ms frames
    hop_length = frame_length // 2
    
    # Compute energy in overlapping windows
    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy.append(np.sum(frame ** 2))
    
    energy = np.array(energy)
    
    # Detect sudden energy spikes
    energy_diff = np.diff(energy)
    threshold = np.mean(energy_diff) + threshold_factor * np.std(energy_diff)
    
    pulse_starts = np.where(energy_diff > threshold)[0]
    
    # Convert frame indices back to sample indices
    pulse_locations = []
    for start_frame in pulse_starts:
        start_sample = start_frame * hop_length
        end_sample = min(start_sample + frame_length * 5, len(audio))  # 50ms pulse duration
        pulse_locations.append((start_sample, end_sample))
    
    return pulse_locations


def remove_pulses(audio: np.ndarray, pulse_locations: list) -> np.ndarray:
    """
    Remove detected pulse artifacts from audio.
    
    Args:
        audio: Original audio signal
        pulse_locations: List of (start, end) pulse locations
        
    Returns:
        np.ndarray: Cleaned audio signal
    """
    cleaned_audio = audio.copy()
    
    for start, end in pulse_locations:
        # Replace pulse with interpolated values
        if start > 0 and end < len(audio):
            # Linear interpolation between points before and after pulse
            before_val = cleaned_audio[start - 1]
            after_val = cleaned_audio[end]
            interpolated = np.linspace(before_val, after_val, end - start)
            cleaned_audio[start:end] = interpolated
        else:
            # If pulse is at beginning or end, zero it out
            cleaned_audio[start:end] = 0
    
    return cleaned_audio


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


# Configuration parameters that can be adjusted
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
