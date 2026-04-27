import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from fusion_real_time_detection import EnhancedNoduleDetectionSystem


@dataclass
class TrainingSample:
    values: list
    is_nodule: bool
    area: float | None = None
    diameter: float | None = None
    depth: float | None = None
    position: tuple | None = None


class EnhancedStressNoduleDetectionSystem:
    """Compatibility layer for the legacy GUI.

    The original project later replaced this logic with stronger residual/score
    based detectors. For the old UI we only need a stable, editable interface.
    """

    def __init__(self):
        self.base_detector = EnhancedNoduleDetectionSystem()
        self.is_trained = False
        self.training_samples: list[TrainingSample] = []
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "enhanced_stress_detector.json")

    def process_frame(self, frame_flat, timestamp):
        arr = np.asarray(frame_flat, dtype=float).reshape(12, 8)
        normalized, _, _ = self.base_detector.advanced_nodule_detection(arr, timestamp)
        combined_probability = float(np.clip(normalized.max() * 0.9 + normalized.mean() * 0.1, 0.0, 1.0))
        return {
            "matrix_data": {
                "normalized_matrix": normalized,
                "raw_matrix": arr,
            },
            "combined_probability": combined_probability,
        }

    def add_training_data(self, current_data, is_nodule=False, area=None, diameter=None, depth=None, position=None):
        sample = TrainingSample(
            values=np.asarray(current_data, dtype=float).flatten().tolist(),
            is_nodule=bool(is_nodule),
            area=area,
            diameter=diameter,
            depth=depth,
            position=position,
        )
        self.training_samples.append(sample)
        return True

    def train_system(self):
        self.is_trained = len(self.training_samples) > 0
        if self.is_trained:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            payload = {"samples": [asdict(s) for s in self.training_samples]}
            with open(self.model_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return self.is_trained

