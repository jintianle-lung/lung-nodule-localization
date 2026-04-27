from pathlib import Path
import runpy
import sys
import os


ROOT = Path(__file__).resolve().parent.parent
UI_RELEASE = ROOT / "ui_release"
for p in (ROOT, UI_RELEASE, ROOT / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

runpy.run_path(str(UI_RELEASE / "modern_detection_gui_optimized.py"), run_name="__main__")
