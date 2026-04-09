
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
from scipy import ndimage
from final_model import DualStreamModel

FRAME_HEIGHT = 12
FRAME_WIDTH = 8
FRAME_SIZE = FRAME_HEIGHT * FRAME_WIDTH

class KeyframeInference:
    def __init__(self, model_path, data_dir, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 10
        self.model = DualStreamModel(seq_len=self.sequence_length).to(self.device)
        payload = torch.load(model_path, map_location=self.device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.interp_factor = 5
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_file(self, file_path):
        try:
            # 1. Load Data
            df = pd.read_csv(file_path)
            mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
            if mat_cols:
                mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
                data = df[mat_cols].values
            else:
                data = df.iloc[:, -FRAME_SIZE:].values
                
            if len(data) < self.sequence_length:
                return []

            # 2. Extract Metadata (Size, Depth) from path
            # Assuming structure: .../Size/Depth/filename.CSV
            dir_path = os.path.dirname(file_path)
            depth_str = os.path.basename(dir_path)
            size_str = os.path.basename(os.path.dirname(dir_path))
            
            # 3. Sliding Window Inference
            probs = []
            frame_indices = []
            
            # Prepare batches for speed
            batch_frames = []
            batch_indices = []
            BATCH_SIZE = 32
            
            # Pre-calculate interpolated frames? No, too much memory.
            # Do it on the fly per batch.
            
            # Calculate steps first
            steps = range(0, len(data) - self.sequence_length, 2)
            
            for i in steps: # Stride 2 for speed
                window = data[i : i+self.sequence_length]
                batch_frames.append(window)
                batch_indices.append(i + self.sequence_length // 2)
                
                if len(batch_frames) >= BATCH_SIZE:
                    self._predict_batch(batch_frames, batch_indices, probs, frame_indices)
                    batch_frames = []
                    batch_indices = []
            
            if batch_frames:
                self._predict_batch(batch_frames, batch_indices, probs, frame_indices)
            
            # 4. Find Peaks
            probs = np.array(probs)
            frame_indices = np.array(frame_indices)
            
            # Filter low probability
            mask = probs > 0.5
            if not np.any(mask):
                return []
                
            high_prob_indices = frame_indices[mask]
            high_probs = probs[mask]
            
            # Simple non-maximum suppression (NMS) in time
            selected = []
            sorted_idx = np.argsort(high_probs)[::-1]
            
            for idx in sorted_idx:
                frame_idx = high_prob_indices[idx]
                score = high_probs[idx]
                
                is_distinct = True
                for s in selected:
                    if abs(s['frame_index'] - frame_idx) < 30: # 30 frames distance
                        is_distinct = False
                        break
                
                if is_distinct:
                    selected.append({
                        'frame_index': frame_idx,
                        'score': float(score),
                        'size_label': size_str,
                        'depth_label': depth_str
                    })
                    if len(selected) >= 3:
                        break
            
            # 5. Save Visualizations
            for item in selected:
                self.save_visualization(item, data[item['frame_index']], file_path)
                
            return selected
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _predict_batch(self, windows, indices, probs_list, indices_list):
        # Align preprocessing with main_gui.py:
        # - reshape to 12x8
        # - per-frame min-max normalization
        # - intensity stats: mean/max/std over raw sequence
        processed_batch = []
        intensity_batch = []
        for window in windows:
            seq_raw_reshaped = []
            for frame in window:
                if frame.size > FRAME_SIZE:
                    frame = frame[-FRAME_SIZE:]
                mat = frame.reshape(FRAME_HEIGHT, FRAME_WIDTH)
                seq_raw_reshaped.append(mat)
            seq_raw_reshaped = np.array(seq_raw_reshaped, dtype=np.float32)

            avg_intensity = float(np.mean(seq_raw_reshaped))
            max_intensity = float(np.max(seq_raw_reshaped))
            std_intensity = float(np.std(seq_raw_reshaped))
            intensity_batch.append([avg_intensity, max_intensity, std_intensity])

            seq_norm = np.zeros((self.sequence_length, 1, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
            for i in range(self.sequence_length):
                mat = seq_raw_reshaped[i]
                mn, mx = mat.min(), mat.max()
                if mx - mn > 1e-6:
                    norm = (mat - mn) / (mx - mn)
                else:
                    norm = mat - mn
                seq_norm[i, 0] = norm
            processed_batch.append(seq_norm)
            
        batch_tensor = torch.tensor(np.array(processed_batch), dtype=torch.float32).to(self.device)
        intensity_tensor = torch.tensor(np.array(intensity_batch), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prob, _, _ = self.model(batch_tensor, intensity_tensor)
            batch_probs = prob.cpu().numpy().flatten()
            
        probs_list.extend(batch_probs)
        indices_list.extend(indices)

    def save_visualization(self, item, raw_frame, file_path):
        mat = raw_frame.reshape(FRAME_HEIGHT, FRAME_WIDTH)
        mn, mx = mat.min(), mat.max()
        norm = (mat - mn) / (mx - mn) if mx > mn else mat - mn
        img = ndimage.zoom(norm, self.interp_factor, order=3) # High quality for save
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        title = (f"AI Detected Keyframe: {item['frame_index']}\n"
                 f"Prob: {item['score']:.4f}\n"
                 f"Label: {item['size_label']} / {item['depth_label']}")
        
        ax.set_title(title)
        ax.axis('off')
        
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path).replace('.CSV', '')
        out_name = f"{base_name}_frame{item['frame_index']}_prob{int(item['score']*100)}.png"
        save_path = os.path.join(dir_name, out_name)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

    def run(self):
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.CSV"), recursive=True)
        print(f"Starting inference on {len(all_files)} files...")
        
        results = []
        for i, f in enumerate(all_files):
            try:
                if i % 10 == 0:
                    print(f"Processing {i}/{len(all_files)}: {os.path.basename(f)}")
                
                res = self.process_file(f)
                for r in res:
                    r['file_path'] = f
                    results.append(r)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
        # Save summary
        if results:
            df = pd.DataFrame(results)
            out_csv = os.path.join(self.output_dir, "deep_learning_keyframes.csv")
            df.to_csv(out_csv, index=False)
            print(f"Inference complete. Saved {len(results)} keyframes to {out_csv}")
        else:
            print("No keyframes detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run keyframe inference with DualStreamModel.")
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "models", "best_model.pth"),
        help="Path to model checkpoint.",
    )
    parser.add_argument("--data-dir", required=True, help="Root directory containing CSV files.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "outputs"),
        help="Directory for inference summary output.",
    )
    args = parser.parse_args()

    inference = KeyframeInference(args.model_path, args.data_dir, args.output_dir)
    inference.run()
