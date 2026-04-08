"""
Model Evaluation Script
=======================

Evaluate trained models on test data.

Usage:
    python evaluate.py --model_path <path> --data_path <path> --stage <1/2/3>
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, accuracy_score
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dual_stream_mstcn_detection import DualStreamMSTCNDetector


def evaluate_detection(model, data, labels, threshold=0.5):
    """Evaluate Stage 1 detection model."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
    
    labels = labels.cpu().numpy()
    binary_preds = (probs > threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(labels, binary_preds),
        'f1': f1_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds),
        'recall': recall_score(labels, binary_preds),
        'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, binary_preds)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    return metrics


def evaluate_size(model, data, labels, size_classes):
    """Evaluate Stage 2 size model."""
    model.eval()
    
    with torch.no_grad():
        size_logits, size_ord_logits, size_reg_norm, size_probs = model(data)
    
    pred_classes = torch.argmax(size_probs, dim=1).cpu().numpy()
    true_classes = labels.cpu().numpy()
    
    # Top-1 and Top-2 accuracy
    top1_acc = accuracy_score(true_classes, pred_classes)
    
    top2_correct = 0
    for i, (pred, true) in enumerate(zip(pred_classes, true_classes)):
        top2_preds = torch.topk(size_probs[i], k=2).indices.cpu().numpy()
        if true in top2_preds:
            top2_correct += 1
    top2_acc = top2_correct / len(true_classes)
    
    # MAE in cm
    pred_cm = np.array([size_classes[p] for p in pred_classes])
    true_cm = np.array([size_classes[t] for t in true_classes])
    mae = np.mean(np.abs(pred_cm - true_cm))
    
    metrics = {
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'mae_cm': mae,
        'confusion_matrix': confusion_matrix(true_classes, pred_classes).tolist(),
        'classification_report': classification_report(
            true_classes, pred_classes,
            target_names=[f'{s}cm' for s in size_classes],
            output_dict=True
        ),
    }
    
    return metrics


def evaluate_depth(model, data, labels, depth_classes):
    """Evaluate Stage 3 depth model."""
    model.eval()
    
    with torch.no_grad():
        # Assuming model outputs depth logits and probs
        depth_logits, depth_probs = model(data)
    
    pred_classes = torch.argmax(depth_probs, dim=1).cpu().numpy()
    true_classes = labels.cpu().numpy()
    
    # Balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(true_classes, pred_classes)
    
    # Regular accuracy
    acc = accuracy_score(true_classes, pred_classes)
    
    # Top-2 accuracy
    top2_correct = 0
    for i, (pred, true) in enumerate(zip(pred_classes, true_classes)):
        top2_preds = torch.topk(depth_probs[i], k=min(2, len(depth_classes))).indices.cpu().numpy()
        if true in top2_preds:
            top2_correct += 1
    top2_acc = top2_correct / len(true_classes)
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'top2_accuracy': top2_acc,
        'confusion_matrix': confusion_matrix(true_classes, pred_classes).tolist(),
        'classification_report': classification_report(
            true_classes, pred_classes,
            target_names=depth_classes,
            output_dict=True
        ),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help='Model stage')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.stage == 1:
        model = DualStreamMSTCNDetector()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        print("Loaded Stage 1 Detection model")
    else:
        print(f"Stage {args.stage} evaluation not fully implemented in this example")
        return
    
    # Note: Load your test data here
    # test_data, test_labels = load_test_data(args.data_path)
    
    print("Evaluation script ready.")
    print("Please implement data loading based on your dataset format.")


if __name__ == '__main__':
    main()
