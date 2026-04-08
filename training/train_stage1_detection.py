"""
Stage 1 Detection Training Script
==================================

This script trains the DualStreamMSTCNDetector for nodule detection.

Usage:
    python train_stage1_detection.py --data_path <path> --output_dir <path>

Example:
    python train_stage1_detection.py \
        --data_path data/training_data \
        --output_dir experiments/outputs_stage1
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dual_stream_mstcn_detection import DualStreamMSTCNDetector
from models.input_normalization_v1 import normalize_raw_frames_window_minmax


class TactileSequenceDataset(Dataset):
    """Dataset for tactile sensor sequences."""
    
    def __init__(self, data_dir: str, labels_path: str, window_size: int = 10, stride: int = 2):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        # Build sample index
        self.samples = self._build_samples()
    
    def _build_samples(self):
        """Build list of (file_path, start_frame, label) tuples."""
        samples = []
        # Implementation depends on your data format
        # This is a placeholder
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, start_frame, label = self.samples[idx]
        
        # Load and preprocess data
        # This is a placeholder - implement based on your data format
        data = np.zeros((self.window_size, 1, 12, 8), dtype=np.float32)
        
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            
            logits = model(data)
            loss = criterion(logits.squeeze(), labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    # Compute metrics
    binary_preds = (preds > 0.5).astype(int)
    metrics = {
        'loss': avg_loss,
        'f1': f1_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds),
        'recall': recall_score(labels, binary_preds),
        'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 Detection Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to labels JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DualStreamMSTCNDetector(
        seq_len=10,
        frame_feature_dim=32,
        temporal_channels=64,
        temporal_blocks=3,
        dropout=0.35,
        use_delta_branch=True,
    )
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Note: In actual training, you would create DataLoaders here
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Training script ready.")
    print("Please implement data loading based on your dataset format.")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
