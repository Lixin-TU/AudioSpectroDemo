import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=3, input_height=128, input_width=128, dropout_rate=0.3, num_frames=5):
        super(AudioCNNLSTMClassifier, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_frames = num_frames  # Number of consecutive frames
        
        # CNN feature extraction layers (for each frame)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Adaptive pooling to ensure consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Temporal feature extraction with LSTM for multiple frames
        # Each frame after CNN produces 256*8*8 features, flattened to 256*64
        self.frame_feature_size = 256 * 8 * 8  # Features per frame
        self.lstm_input_size = self.frame_feature_size
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        
        # LSTM to process sequence of frames
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout_rate if self.lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 256)  # *2 for bidirectional
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, num_frames, 1, height, width)
        batch_size = x.size(0)
        
        # Process each frame through CNN
        frame_features = []
        for i in range(self.num_frames):
            frame = x[:, i, :, :, :]  # (batch_size, 1, height, width)
            
            # Ensure input has the expected shape
            if frame.shape[-2:] != (self.input_height, self.input_width):
                frame = F.interpolate(frame, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
            
            # CNN feature extraction for this frame
            frame_out = self.pool1(F.relu(self.bn1(self.conv1(frame))))
            frame_out = self.pool2(F.relu(self.bn2(self.conv2(frame_out))))
            frame_out = self.pool3(F.relu(self.bn3(self.conv3(frame_out))))
            frame_out = self.pool4(F.relu(self.bn4(self.conv4(frame_out))))
            
            # Adaptive pooling to ensure consistent size
            frame_out = self.adaptive_pool(frame_out)  # (batch, 256, 8, 8)
            
            # Flatten frame features
            frame_out = frame_out.view(batch_size, -1)  # (batch_size, 256*8*8)
            frame_features.append(frame_out)
        
        # Stack frame features for LSTM
        # Shape: (batch_size, num_frames, frame_feature_size)
        lstm_input = torch.stack(frame_features, dim=1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Use the last output from LSTM
        # For bidirectional LSTM, use the last output
        x = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size * 2)
        
        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Keep the old class for backward compatibility but make it an alias
AudioClassifier = AudioCNNLSTMClassifier

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.ones(3) * alpha
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
