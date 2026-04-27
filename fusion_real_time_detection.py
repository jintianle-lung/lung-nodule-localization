import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class FastProtocolParser:
    """Parse the legacy 104-byte tactile packet format into 12x8 frames."""

    def __init__(self):
        self.buffer = bytearray()
        self.frame_header = b"\xA5\x5A"
        self.expected_frame_size = 104
        self.latest_frame = None
        self.last_parse_time = 0.0
        self.parse_interval = 0.005

    def add_data(self, data: bytes):
        self.buffer.extend(data)
        if len(self.buffer) > 4096:
            self.buffer = self.buffer[-1024:]

        now = time.time()
        if now - self.last_parse_time < self.parse_interval:
            return
        self.last_parse_time = now
        self._parse()

    def _parse(self):
        while len(self.buffer) >= self.expected_frame_size:
            idx = self.buffer.find(self.frame_header)
            if idx == -1:
                self.buffer = self.buffer[-10:]
                return
            if idx > 0:
                self.buffer = self.buffer[idx:]
            if len(self.buffer) < self.expected_frame_size:
                return

            frame_data = self.buffer[: self.expected_frame_size]
            matrix = np.frombuffer(frame_data[6:102], dtype=np.uint8).reshape(12, 8)
            self.latest_frame = {"timestamp": time.time(), "matrix": matrix}
            self.buffer = self.buffer[self.expected_frame_size :]

    def get_latest(self):
        return self.latest_frame


class EnhancedNoduleDetectionSystem:
    """Small deterministic visual detector used by the legacy GUI."""

    def __init__(self):
        try:
            self.medical_cmap = plt.get_cmap("turbo")
        except Exception:
            self.medical_cmap = plt.get_cmap("viridis")

    def advanced_nodule_detection(self, stress_grid, timestamp):
        stress_grid = np.asarray(stress_grid, dtype=float)
        mx = float(stress_grid.max()) if stress_grid.size else 0.0
        normalized = stress_grid / mx if mx > 0 else stress_grid.copy()
        mask = normalized > 0.3

        nodules = []
        if np.any(mask):
            area = int(mask.sum())
            if area > 2:
                yy, xx = np.where(mask)
                centroid = (float(np.mean(yy)), float(np.mean(xx)))
                intensity = float(normalized[mask].mean())
                nodules.append(
                    {
                        "area": area,
                        "centroid": centroid,
                        "intensity": intensity,
                        "risk_score": min(intensity * 1.5, 1.0),
                    }
                )

        return normalized, mask, nodules


class _QueuedFrameBuffer:
    def __init__(self, maxlen=32):
        self.frames = deque(maxlen=maxlen)

    def append(self, frame):
        self.frames.append(np.asarray(frame, dtype=float))

