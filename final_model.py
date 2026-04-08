import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamModel(nn.Module):
    """
    Final optimized model for Nodule Detection, Size, and Depth Estimation.
    
    Architecture:
    - Stream 1 (Shape): 3D CNN taking normalized sequence (B, 1, Seq, 12, 8).
      Captures Spatio-Temporal shape features (sharpness, spread, evolution).
    - Stream 2 (Intensity): MLP taking average intensity (scalar).
      Captures absolute signal strength.
    - Fusion: Concatenates streams and uses specialized heads for Prob, Size, Depth.
    """
    def __init__(self, seq_len=10):
        super(DualStreamModel, self).__init__()
        
        # Branch 1: Shape Stream (3D CNN)
        # Input: (B, 1, Seq, 12, 8)
        # Checkpoint: conv1 weight [32, 1, 3, 3, 3]
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) # -> (32, 5, 6, 4)
        
        # Checkpoint: conv2 weight [64, 32, 3, 3, 3]
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        
        # Spatial Conv2D (Restored from checkpoint)
        # Checkpoint: spatial_conv.0.weight [32, 128, 3, 3]
        # This implies:
        # Input channels: 128. Output channels: 32. Kernel: 3x3.
        # How to get 128 channels?
        # Conv2 output is (B, 64, T, H, W).
        # If we flatten time, and T=2? 64*2 = 128.
        # Pool1 reduces T from 10 to 5.
        # How to get T=2? Maybe another pool? Or stride?
        # Or maybe pool1 was (2,1,1)?
        
        # Let's check memory 03flgdtda7wvngtclt2zq9td7 again.
        # "Pool1: 2x1x1 (仅压缩时间，保留12x8空间分辨率)"
        # "Pool2: 2x1x1"
        # "Flatten Time-to-Channel (128ch)"
        
        # If T=10. Pool1(2,1,1) -> T=5.
        # Pool2(2,1,1) -> T=2 (floor of 2.5).
        # 64ch * 2 = 128ch. Matches!
        
        # So I need to restore Pool1 and Pool2 as per memory.
        
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)) 
        
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # Spatial Conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), # 3x3 padding 1 preserves size
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        # Global Pool
        # Input to global pool is (B, 32, 12, 8) if we preserved spatial dims.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Branch 2: Intensity Stream (MLP)
        # Checkpoint: intensity_mlp.0.weight [16, 3] -> Input 3.
        self.intensity_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Shared Fusion Layer
        # Checkpoint: fusion_fc.0.weight [64, 64].
        # 32 (Shape) + 32 (Intensity) = 64. Matches.
        self.fusion_fc = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task Heads
        
        # 1. Detection Head (Probability)
        self.prob_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 2. Size Head (Regression)
        self.size_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Depth Head (Regression)
        self.depth_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, avg_intensity):
        # x shape: (B, Seq, 1, 12, 8) -> Permute to (B, 1, Seq, 12, 8) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Branch 1: Shape
        x_shape = F.relu(self.bn1(self.conv1(x)))
        x_shape = self.pool1(x_shape) # (B, 32, 5, 12, 8)
        
        x_shape = F.relu(self.bn2(self.conv2(x_shape))) # (B, 64, 5, 12, 8)
        x_shape = self.pool2(x_shape) # (B, 64, 2, 12, 8)
        
        # Flatten Time into Channels
        B, C, T, H, W = x_shape.shape
        x_shape = x_shape.permute(0, 1, 2, 3, 4).reshape(B, C*T, H, W) # (B, 128, 12, 8)
        
        x_shape = self.spatial_conv(x_shape) # (B, 32, 12, 8)
        x_shape = self.global_pool(x_shape) # (B, 32, 1, 1)
        x_shape = x_shape.view(x_shape.size(0), -1) # (B, 32)
        
        # Branch 2: Intensity
        # Assuming avg_intensity is (B, 3)
        x_intensity = self.intensity_mlp(avg_intensity) # (B, 32)
        
        # Fusion
        combined = torch.cat((x_shape, x_intensity), dim=1) # (B, 64)
        features = self.fusion_fc(combined)
        
        prob = self.prob_head(features)
        size = self.size_head(features)
        depth = self.depth_head(features)
        
        return prob, size, depth
