from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

from PIL import ImageGrab


ROOT = Path(__file__).resolve().parent.parent
CODE_ARCHIVE = ROOT / "Code_Archive"
SAMPLE_CSV = ROOT / "整理好的数据集" / "建表数据" / "0.25cm大" / "0.5cm深" / "1.CSV"
OUT_PATH = ROOT / "tmp" / "optimized_gui_demo.png"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

for p in (ROOT, CODE_ARCHIVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_module():
    module_path = CODE_ARCHIVE / "modern_detection_gui_optimized.py"
    spec = importlib.util.spec_from_file_location("modern_detection_gui_optimized", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module()

    # Force the file picker to open a known sample CSV.
    module.filedialog.askopenfilename = lambda **kwargs: str(SAMPLE_CSV)

    root = module.tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        root.geometry("1600x1000+0+0")
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))

    app = module.OptimizedDetectionGUI(root)

    def load_sample():
        app.load_csv_file()
        app.current_frame = min(115, len(app.data) - 1) if app.data is not None else 0
        if hasattr(app, "frame_var"):
            app.frame_var.set(app.current_frame)
        if hasattr(app, "view_mode_var"):
            app.view_mode_var.set("全部")
            app.update_subplot_layout()
        app.update_display()
        root.update_idletasks()
        root.update()

    def capture_and_exit():
        root.update()
        bbox = (
            root.winfo_rootx(),
            root.winfo_rooty(),
            root.winfo_rootx() + root.winfo_width(),
            root.winfo_rooty() + root.winfo_height(),
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab(bbox=bbox).save(OUT_PATH)
        print(str(OUT_PATH))
        root.destroy()

    root.after(1500, load_sample)
    root.after(5000, capture_and_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
