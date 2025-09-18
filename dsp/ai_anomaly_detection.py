"""
AI-based Anomaly Detection (Supervised)

This module integrates the pre-trained CNN+LSTM classifier directly for
leak/pocket detection without relying on external helper files.
"""

import os
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import librosa

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    F = None


class AudioCNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=3, input_height=128, input_width=128, dropout_rate=0.3, num_frames=5):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.num_frames = num_frames

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.frame_feature_size = 256 * 8 * 8
        self.lstm_input_size = self.frame_feature_size
        self.lstm_hidden_size = 128
        self.lstm_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout_rate if self.lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        frame_features = []
        for i in range(x.size(1)):
            frame = x[:, i, :, :, :]
            if frame.shape[-2:] != (self.input_height, self.input_width):
                frame = F.interpolate(frame, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
            f = self.pool1(F.relu(self.bn1(self.conv1(frame))))
            f = self.pool2(F.relu(self.bn2(self.conv2(f))))
            f = self.pool3(F.relu(self.bn3(self.conv3(f))))
            f = self.pool4(F.relu(self.bn4(self.conv4(f))))
            f = self.adaptive_pool(f)
            f = f.view(batch_size, -1)
            frame_features.append(f)
        lstm_input = torch.stack(frame_features, dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        x = lstm_out[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# In-module configuration (embedded, no YAML dependency)
MODEL_CFG = {
    'num_classes': 3,
    'num_frames': 5,
    'input_size': (128, 128),
    'dropout': 0.3,
    'sample_rate': 32552,
    'n_mels': 128,
    'n_fft': 2048,
    'window_length': 1.0,  # seconds per frame
    'frame_hop': 0.2,      # seconds between frames within a sequence
    'overlap': 0.50,       # sequence stride overlap
    'threshold': 0.395,    # decision threshold for abnormal classes
}


def _resize_2d_linear(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = arr.shape
    if h == new_h and w == new_w:
        return arr
    # Resize along axis 0
    row_idx = np.linspace(0, h - 1, new_h)
    temp = np.empty((new_h, w), dtype=arr.dtype)
    base_rows = np.arange(h)
    for j in range(w):
        temp[:, j] = np.interp(row_idx, base_rows, arr[:, j])
    # Resize along axis 1
    col_idx = np.linspace(0, w - 1, new_w)
    out = np.empty((new_h, new_w), dtype=arr.dtype)
    base_cols = np.arange(w)
    for i in range(new_h):
        out[i, :] = np.interp(col_idx, base_cols, temp[i, :])
    return out


def _extract_mel_window(y: np.ndarray, sr: int, start_sec: float, window_sec: float, n_mels: int, n_fft: int, target_hw: Tuple[int, int]) -> np.ndarray:
    start = int(start_sec * sr)
    end = start + int(window_sec * sr)
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0
    if end > len(y):
        pad_right = end - len(y)
        end = len(y)
    else:
        pad_right = 0
    segment = y[start:end]
    if pad_left or pad_right:
        segment = np.pad(segment, (pad_left, pad_right), mode='constant')

    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    # Transpose to (time, freq) before resize to (128,128)
    feat = mel_db.T.astype(np.float32)
    feat = _resize_2d_linear(feat, target_hw[0], target_hw[1])  # (time, freq)
    # Back to (1, H, W) channel-first
    return feat[np.newaxis, :, :]


def _load_model(device: str, checkpoints_dir: str) -> Tuple[AudioCNNLSTMClassifier, str]:
    model = AudioCNNLSTMClassifier(num_classes=MODEL_CFG['num_classes'], dropout_rate=MODEL_CFG['dropout'], num_frames=MODEL_CFG['num_frames'])
    ckpt_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        # Try project root checkpoints
        alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'best_model.pth')
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model, ckpt_path


def process_ai_anomaly_detection(audio_file_path: str, progress_cb: Optional[Callable[[int, str], None]] = None) -> dict:
    """Run supervised AI anomaly detection using the embedded model.

    progress_cb: optional callback receiving (percent:int, message:str)
    """
    if torch is None:
        return {'status': 'error', 'error_message': 'PyTorch is not available. Please install torch.'}

    try:
        if progress_cb:
            progress_cb(5, 'Loading audio...')
        y, sr_in = librosa.load(audio_file_path, sr=None, mono=True)
        target_sr = MODEL_CFG['sample_rate']
        if sr_in != target_sr:
            if progress_cb:
                progress_cb(8, 'Resampling audio...')
            y = librosa.resample(y, orig_sr=sr_in, target_sr=target_sr)
        sr = target_sr

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        if progress_cb:
            progress_cb(12, f'Loading model on {device}...')
        model, model_path = _load_model(device, checkpoints_dir)
        if progress_cb:
            progress_cb(25, 'Model loaded. Preparing windows...')

        window_len = MODEL_CFG['window_length']
        step = window_len * (1.0 - MODEL_CFG['overlap'])  # e.g. 0.5s
        frame_hop = MODEL_CFG['frame_hop']               # 0.2s
        num_frames = MODEL_CFG['num_frames']             # 5
        H, W = MODEL_CFG['input_size']

        total_duration = len(y) / sr
        times = []
        results = []  # per-sequence predictions
        t = 0.0
        class_names = {0: 'Normal', 1: 'Leak', 2: 'Pocket'}

        # Pre-compute number of sequences for progress
        total_sequences = int(max(0, np.floor((total_duration - window_len) / step) + 1))
        processed_sequences = 0

        if progress_cb:
            progress_cb(35, 'Running inference...')
        while t + window_len <= total_duration + 1e-6:
            frames = []
            for k in range(num_frames):
                start_sec = t + k * frame_hop
                frames.append(_extract_mel_window(y, sr, start_sec, window_len, MODEL_CFG['n_mels'], MODEL_CFG['n_fft'], (H, W)))
            # Stack to (num_frames, 1, H, W)
            seq = np.stack(frames, axis=0)
            tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)  # (1, num_frames, 1, H, W)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            leak_prob, pocket_prob = float(probs[1]), float(probs[2])
            results.append({
                'start_time': t,
                'end_time': t + window_len,
                'predicted_class': pred_idx,
                'normal_prob': float(probs[0]),
                'leak_prob': leak_prob,
                'pocket_prob': pocket_prob,
                'confidence': float(max(leak_prob, pocket_prob)),
            })
            times.append(t)
            t += step
            processed_sequences += 1
            if progress_cb and total_sequences > 0 and processed_sequences % max(1, total_sequences // 20) == 0:
                # Scale 35→85 for inference progress
                pct = 35 + int(50 * processed_sequences / total_sequences)
                progress_cb(min(85, pct), f'Inference {processed_sequences}/{total_sequences}...')

        # Collect anomaly segments using threshold
        if progress_cb:
            progress_cb(86, 'Post-processing events...')
        threshold = MODEL_CFG['threshold']
        raw_segments = []
        for r in results:
            if r['predicted_class'] != 0 and r['confidence'] >= threshold:
                label = class_names[r['predicted_class']]
                raw_segments.append({'start_time': r['start_time'], 'end_time': r['end_time'], 'label': label, 'probability': r['confidence']})

        # Merge consecutive/overlapping segments per class
        def merge_segments(segments: List[Dict], max_gap: float = step + 1e-6) -> List[Dict]:
            if not segments:
                return []
            segments = sorted(segments, key=lambda x: (x['label'], x['start_time']))
            merged: List[Dict] = []
            cur = segments[0].copy()
            for s in segments[1:]:
                if s['label'] == cur['label'] and s['start_time'] <= cur['end_time'] + max_gap:
                    cur['end_time'] = max(cur['end_time'], s['end_time'])
                    cur['probability'] = max(cur['probability'], s.get('probability', 0.0))
                else:
                    merged.append(cur)
                    cur = s.copy()
            merged.append(cur)
            return merged

        merged_segments = merge_segments(raw_segments)
        if progress_cb:
            progress_cb(92, 'Finalizing results...')

        result = {
            'status': 'success',
            'anomaly_events': merged_segments,
            'total_sequences': len(results),
            'audio_duration': total_duration,
            'sample_rate': sr,
            'model_info': {
                'path': model_path,
                'device': device,
                'classes': ['Normal', 'Leak', 'Pocket'],
                'window_length': window_len,
                'overlap': MODEL_CFG['overlap'],
                'num_frames': num_frames,
            },
        }
        if progress_cb:
            progress_cb(98, 'Done.')
        return result

    except Exception as e:  # pragma: no cover
        return {'status': 'error', 'error_message': str(e)}

