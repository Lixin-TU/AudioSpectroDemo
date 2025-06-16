"""
Anomaly Detection Processing Module

This module provides functionality to detect anomalous events in audio signals
using traditional signal processing techniques.
"""

import numpy as np
import librosa
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional


def process_anomaly_detection(audio_file_path: str) -> dict:
    """
    Main function to process audio file and detect anomalies.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: Detection results and statistics
    """
    print(f"Processing Anomaly Detection for: {audio_file_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        
        # Extract features for anomaly detection
        features = extract_anomaly_features(y, sr)
        
        # Detect anomalies using multiple methods
        spectral_anomalies = detect_spectral_anomalies(y, sr)
        temporal_anomalies = detect_temporal_anomalies(y, sr)
        energy_anomalies = detect_energy_anomalies(y, sr)
        
        # Combine results
        all_anomalies = combine_anomaly_results(
            spectral_anomalies, temporal_anomalies, energy_anomalies, len(y), sr
        )
        
        # Calculate statistics
        total_anomalies = len(all_anomalies)
        anomaly_density = total_anomalies / (len(y) / sr)  # anomalies per second
        
        result = {
            'status': 'success',
            'total_anomalies': total_anomalies,
            'anomaly_density': anomaly_density,
            'spectral_anomalies': len(spectral_anomalies),
            'temporal_anomalies': len(temporal_anomalies),
            'energy_anomalies': len(energy_anomalies),
            'anomaly_locations': all_anomalies,
            'audio_duration': len(y) / sr,
            'sample_rate': sr,
            'features': features
        }
        
        print(f"Anomaly Detection completed: {total_anomalies} anomalies found")
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def extract_anomaly_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract features relevant for anomaly detection.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of audio
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Spectral features
    stft = librosa.stft(audio)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    
    # Energy features
    rms_energy = librosa.feature.rms(y=audio)[0]
    
    features = {
        'spectral_centroid_mean': np.mean(spectral_centroids),
        'spectral_centroid_std': np.std(spectral_centroids),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff),
        'zcr_mean': np.mean(zero_crossing_rate),
        'zcr_std': np.std(zero_crossing_rate),
        'rms_energy_mean': np.mean(rms_energy),
        'rms_energy_std': np.std(rms_energy),
        'mfcc_means': np.mean(mfccs, axis=1).tolist(),
        'mfcc_stds': np.std(mfccs, axis=1).tolist()
    }
    
    return features


def detect_spectral_anomalies(audio: np.ndarray, sample_rate: int, 
                             threshold_factor: float = 2.5) -> List[Tuple[float, float]]:
    """
    Detect anomalies based on spectral characteristics.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        threshold_factor: Standard deviations for anomaly threshold
        
    Returns:
        List of (start_time, end_time) tuples for anomalies
    """
    # Compute spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    
    # Detect outliers using z-score
    z_scores = np.abs(stats.zscore(spectral_centroids))
    anomaly_frames = np.where(z_scores > threshold_factor)[0]
    
    # Convert frames to time segments
    hop_length = 512
    frame_duration = hop_length / sample_rate
    
    anomalies = []
    if len(anomaly_frames) > 0:
        # Group consecutive frames
        groups = []
        current_group = [anomaly_frames[0]]
        
        for i in range(1, len(anomaly_frames)):
            if anomaly_frames[i] - anomaly_frames[i-1] <= 2:  # Allow 1 frame gap
                current_group.append(anomaly_frames[i])
            else:
                groups.append(current_group)
                current_group = [anomaly_frames[i]]
        groups.append(current_group)
        
        # Convert to time segments
        for group in groups:
            start_time = group[0] * frame_duration
            end_time = (group[-1] + 1) * frame_duration
            anomalies.append((start_time, end_time))
    
    return anomalies


def detect_temporal_anomalies(audio: np.ndarray, sample_rate: int,
                             window_size: float = 1.0) -> List[Tuple[float, float]]:
    """
    Detect anomalies based on temporal patterns.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        window_size: Analysis window size in seconds
        
    Returns:
        List of (start_time, end_time) tuples for anomalies
    """
    window_samples = int(window_size * sample_rate)
    hop_samples = window_samples // 2
    
    # Calculate local statistics
    local_stats = []
    times = []
    
    for i in range(0, len(audio) - window_samples, hop_samples):
        window = audio[i:i + window_samples]
        local_stats.append({
            'mean': np.mean(np.abs(window)),
            'std': np.std(window),
            'max': np.max(np.abs(window)),
            'energy': np.sum(window ** 2)
        })
        times.append(i / sample_rate)
    
    # Detect anomalies in energy patterns
    energies = [stat['energy'] for stat in local_stats]
    energy_z_scores = np.abs(stats.zscore(energies))
    
    anomalies = []
    for i, z_score in enumerate(energy_z_scores):
        if z_score > 2.0:  # Threshold for temporal anomalies
            start_time = times[i]
            end_time = start_time + window_size
            anomalies.append((start_time, end_time))
    
    return anomalies


def detect_energy_anomalies(audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
    """
    Detect anomalies based on sudden energy changes.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        
    Returns:
        List of (start_time, end_time) tuples for anomalies
    """
    # Compute RMS energy
    frame_length = 2048
    hop_length = 512
    rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                    hop_length=hop_length)[0]
    
    # Detect sudden changes
    energy_diff = np.diff(rms_energy)
    threshold = np.mean(np.abs(energy_diff)) + 3 * np.std(energy_diff)
    
    anomaly_frames = np.where(np.abs(energy_diff) > threshold)[0]
    
    # Convert to time segments
    frame_duration = hop_length / sample_rate
    anomalies = []
    
    for frame in anomaly_frames:
        start_time = frame * frame_duration
        end_time = start_time + frame_duration * 3  # Extend duration
        anomalies.append((start_time, end_time))
    
    return anomalies


def combine_anomaly_results(spectral: List, temporal: List, energy: List, 
                          audio_length: int, sample_rate: int) -> List[Dict]:
    """
    Combine results from different anomaly detection methods.
    
    Args:
        spectral: Spectral anomalies
        temporal: Temporal anomalies  
        energy: Energy anomalies
        audio_length: Length of audio in samples
        sample_rate: Sample rate
        
    Returns:
        List of combined anomaly events
    """
    all_events = []
    
    # Add spectral anomalies
    for start, end in spectral:
        all_events.append({
            'start_time': start,
            'end_time': end,
            'type': 'spectral',
            'confidence': 0.8
        })
    
    # Add temporal anomalies
    for start, end in temporal:
        all_events.append({
            'start_time': start,
            'end_time': end,
            'type': 'temporal',
            'confidence': 0.7
        })
    
    # Add energy anomalies
    for start, end in energy:
        all_events.append({
            'start_time': start,
            'end_time': end,
            'type': 'energy',
            'confidence': 0.9
        })
    
    # Sort by start time and remove overlaps
    all_events.sort(key=lambda x: x['start_time'])
    
    return all_events


# Configuration parameters
ANOMALY_DETECTION_CONFIG = {
    'spectral_threshold': 2.5,      # Z-score threshold for spectral anomalies
    'temporal_window': 1.0,         # Window size for temporal analysis (seconds)
    'energy_threshold': 3.0,        # Threshold for energy change detection
    'min_anomaly_duration': 0.1,    # Minimum anomaly duration (seconds)
    'max_anomaly_duration': 5.0     # Maximum anomaly duration (seconds)
}


if __name__ == "__main__":
    # Test the module
    test_file = "test_audio.wav"  # Replace with actual test file
    result = process_anomaly_detection(test_file)
    print("Test result:", result)
