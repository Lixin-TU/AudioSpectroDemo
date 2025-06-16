"""
AI-based Anomaly Detection Processing Module

This module provides AI/ML-based functionality to detect anomalous events 
in audio signals using machine learning techniques.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def process_ai_anomaly_detection(audio_file_path: str) -> dict:
    """
    Main function to process audio file using AI-based anomaly detection.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: AI detection results and statistics
    """
    print(f"Processing AI Anomaly Detection for: {audio_file_path}")
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        
        # Extract comprehensive feature set
        features = extract_ml_features(y, sr)
        
        # Apply AI anomaly detection
        anomaly_scores, anomaly_labels = detect_anomalies_ml(features)
        
        # Post-process results
        anomaly_events = post_process_ai_results(anomaly_labels, anomaly_scores, sr)
        
        # Calculate AI-specific metrics
        anomaly_confidence = calculate_ai_confidence(anomaly_scores, anomaly_labels)
        feature_importance = analyze_feature_importance(features, anomaly_labels)
        
        result = {
            'status': 'success',
            'ai_anomalies_detected': len(anomaly_events),
            'average_confidence': np.mean(anomaly_confidence),
            'max_confidence': np.max(anomaly_confidence) if len(anomaly_confidence) > 0 else 0,
            'anomaly_events': anomaly_events,
            'feature_importance': feature_importance,
            'model_type': 'isolation_forest',
            'total_windows_analyzed': len(features),
            'anomaly_percentage': np.mean(anomaly_labels == -1) * 100,
            'audio_duration': len(y) / sr,
            'sample_rate': sr
        }
        
        print(f"AI Anomaly Detection completed: {len(anomaly_events)} events found")
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def extract_ml_features(audio: np.ndarray, sample_rate: int, 
                       window_size: float = 2.0, hop_size: float = 1.0) -> np.ndarray:
    """
    Extract comprehensive feature set suitable for ML anomaly detection.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        window_size: Analysis window size in seconds
        hop_size: Hop size between windows in seconds
        
    Returns:
        np.ndarray: Feature matrix (n_windows, n_features)
    """
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    features_list = []
    
    for i in range(0, len(audio) - window_samples, hop_samples):
        window = audio[i:i + window_samples]
        
        # Time-domain features
        time_features = extract_time_domain_features(window)
        
        # Frequency-domain features
        freq_features = extract_frequency_domain_features(window, sample_rate)
        
        # Spectral features
        spectral_features = extract_spectral_features(window, sample_rate)
        
        # Combine all features
        window_features = np.concatenate([time_features, freq_features, spectral_features])
        features_list.append(window_features)
    
    return np.array(features_list)


def extract_time_domain_features(window: np.ndarray) -> np.ndarray:
    """Extract time-domain features from audio window."""
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(window),
        np.std(window),
        np.var(window),
        np.max(np.abs(window)),
        np.min(window),
        np.max(window)
    ])
    
    # Energy features
    features.extend([
        np.sum(window ** 2),  # Total energy
        np.mean(window ** 2), # Average power
        np.sqrt(np.mean(window ** 2))  # RMS
    ])
    
    # Higher-order statistics
    features.extend([
        float(np.abs(np.mean(window ** 3))),  # Skewness approximation
        float(np.mean(window ** 4)),          # Kurtosis approximation
    ])
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(window)))
    features.append(zero_crossings / len(window))
    
    return np.array(features)


def extract_frequency_domain_features(window: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract frequency-domain features from audio window."""
    # Compute FFT
    fft = np.fft.fft(window)
    magnitude = np.abs(fft[:len(fft)//2])
    frequencies = np.fft.fftfreq(len(window), 1/sample_rate)[:len(fft)//2]
    
    features = []
    
    # Spectral statistics
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.sum(magnitude)
    ])
    
    # Spectral centroid
    if np.sum(magnitude) > 0:
        spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
    else:
        spectral_centroid = 0
    features.append(spectral_centroid)
    
    # Spectral spread
    if np.sum(magnitude) > 0:
        spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
    else:
        spectral_spread = 0
    features.append(spectral_spread)
    
    # Frequency band energies
    freq_bands = [(0, 1000), (1000, 5000), (5000, 10000), (10000, sample_rate//2)]
    for low, high in freq_bands:
        band_mask = (frequencies >= low) & (frequencies < high)
        band_energy = np.sum(magnitude[band_mask] ** 2)
        features.append(band_energy)
    
    return np.array(features)


def extract_spectral_features(window: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract advanced spectral features using librosa."""
    features = []
    
    try:
        # Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=window, sr=sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=window, sr=sample_rate)
        features.extend(np.mean(chroma, axis=1))
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=window, sr=sample_rate)
        features.extend(np.mean(contrast, axis=1))
        
    except Exception as e:
        # If librosa features fail, add zeros
        print(f"Warning: Could not extract some spectral features: {e}")
        features.extend([0] * 40)  # Approximate number of expected features
    
    return np.array(features)


def detect_anomalies_ml(features: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply machine learning anomaly detection to features.
    
    Args:
        features: Feature matrix
        contamination: Expected fraction of anomalies
        
    Returns:
        Tuple of (anomaly_scores, anomaly_labels)
    """
    # Handle edge cases
    if len(features) == 0:
        return np.array([]), np.array([])
    
    if len(features) < 10:
        # Too few samples for ML, mark all as normal
        return np.zeros(len(features)), np.ones(len(features))
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA for dimensionality reduction if needed
    if features_scaled.shape[1] > 20:
        pca = PCA(n_components=min(20, features_scaled.shape[0] - 1))
        features_scaled = pca.fit_transform(features_scaled)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = iso_forest.fit_predict(features_scaled)
    anomaly_scores = iso_forest.score_samples(features_scaled)
    
    return anomaly_scores, anomaly_labels


def post_process_ai_results(labels: np.ndarray, scores: np.ndarray, 
                           sample_rate: int, window_hop: float = 1.0) -> List[Dict]:
    """
    Post-process AI anomaly detection results into events.
    
    Args:
        labels: Anomaly labels (-1 for anomaly, 1 for normal)
        scores: Anomaly scores (lower is more anomalous)
        sample_rate: Audio sample rate
        window_hop: Time between analysis windows
        
    Returns:
        List of anomaly event dictionaries
    """
    if len(labels) == 0:
        return []
    
    events = []
    in_anomaly = False
    current_event = None
    
    for i, (label, score) in enumerate(zip(labels, scores)):
        time_stamp = i * window_hop
        
        if label == -1:  # Anomaly detected
            if not in_anomaly:
                # Start new anomaly event
                current_event = {
                    'start_time': time_stamp,
                    'end_time': time_stamp + window_hop,
                    'type': 'ai_anomaly',
                    'confidence': 1 - min(max((score + 0.5) / 0.5, 0), 1),  # Normalize score to 0-1
                    'severity': 'low' if score > -0.2 else 'medium' if score > -0.4 else 'high',
                    'ai_score': float(score)
                }
                in_anomaly = True
            else:
                # Extend current event
                current_event['end_time'] = time_stamp + window_hop
                # Update confidence to average
                current_confidence = current_event['confidence']
                new_confidence = 1 - min(max((score + 0.5) / 0.5, 0), 1)
                current_event['confidence'] = (current_confidence + new_confidence) / 2
        else:
            if in_anomaly:
                # End current anomaly event
                events.append(current_event)
                in_anomaly = False
                current_event = None
    
    # Handle case where anomaly extends to end of audio
    if in_anomaly and current_event:
        events.append(current_event)
    
    return events


def calculate_ai_confidence(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculate confidence scores for AI predictions."""
    if len(scores) == 0:
        return np.array([])
    
    # Convert anomaly scores to confidence (lower score = higher confidence for anomalies)
    confidence = np.zeros_like(scores)
    anomaly_mask = labels == -1
    
    if np.any(anomaly_mask):
        # For anomalies, confidence based on how negative the score is
        confidence[anomaly_mask] = 1 - np.clip((scores[anomaly_mask] + 0.5) / 0.5, 0, 1)
    
    # For normal samples, confidence based on how positive the score is
    normal_mask = labels == 1
    if np.any(normal_mask):
        confidence[normal_mask] = np.clip((scores[normal_mask] + 0.5) / 0.5, 0, 1)
    
    return confidence


def analyze_feature_importance(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze which features are most important for anomaly detection.
    
    Args:
        features: Feature matrix
        labels: Anomaly labels
        
    Returns:
        Dict with feature importance analysis
    """
    if len(features) == 0 or len(np.unique(labels)) < 2:
        return {'status': 'insufficient_data'}
    
    try:
        # Simple feature importance based on variance between normal and anomalous samples
        normal_features = features[labels == 1]
        anomaly_features = features[labels == -1]
        
        if len(normal_features) == 0 or len(anomaly_features) == 0:
            return {'status': 'insufficient_data'}
        
        # Calculate mean difference for each feature
        normal_means = np.mean(normal_features, axis=0)
        anomaly_means = np.mean(anomaly_features, axis=0)
        
        feature_differences = np.abs(normal_means - anomaly_means)
        feature_importance = feature_differences / (np.std(features, axis=0) + 1e-8)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        
        return {
            'status': 'success',
            'top_feature_indices': top_indices.tolist(),
            'importance_scores': feature_importance[top_indices].tolist(),
            'total_features': len(feature_importance)
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# Configuration parameters
AI_ANOMALY_CONFIG = {
    'window_size': 2.0,             # Analysis window size (seconds)
    'hop_size': 1.0,                # Hop between windows (seconds)
    'contamination': 0.1,           # Expected anomaly fraction
    'n_estimators': 100,            # Number of trees in Isolation Forest
    'confidence_threshold': 0.7,    # Minimum confidence for reporting
    'min_event_duration': 0.5,      # Minimum event duration (seconds)
    'feature_sets': ['time', 'frequency', 'spectral']  # Feature types to use
}


if __name__ == "__main__":
    # Test the module
    test_file = "test_audio.wav"  # Replace with actual test file
    result = process_ai_anomaly_detection(test_file)
    print("Test result:", result)
