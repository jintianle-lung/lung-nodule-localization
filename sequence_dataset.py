import json
import os

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore
        pass


class NoduleSequenceDataset(Dataset):
    """Minimal dataset wrapper kept for GUI compatibility."""

    def __init__(self, root_dir, json_label_path, mode="train", seq_len=10, transform=None, norm_method="frame", active_samples=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.norm_method = norm_method
        self.active_samples = set(active_samples) if active_samples else set()
        self.file_data = []
        self.indices = []

        with open(json_label_path, "r", encoding="utf-8") as f:
            self.labels_map = json.load(f)

        self._prepare_data(mode)

    def _prepare_data(self, mode):
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                if not fn.lower().endswith(".csv"):
                    continue
                full_path = os.path.join(dirpath, fn)
                try:
                    df = pd.read_csv(full_path)
                    values = df.iloc[:, -96:].values.astype(np.float32)
                except Exception:
                    continue
                labels = np.zeros((len(values), 3), dtype=np.float32)
                self.file_data.append((values, labels, full_path))
                for end_idx in range(self.seq_len - 1, len(values)):
                    self.indices.append((len(self.file_data) - 1, end_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_idx = self.indices[idx]
        data, labels, _ = self.file_data[file_idx]
        seq = data[end_idx - self.seq_len + 1 : end_idx + 1]
        if self.transform:
            seq = self.transform(seq)
        return seq, labels[end_idx]

