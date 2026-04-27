from pathlib import Path
import runpy
import sys
import os


ROOT = Path(__file__).resolve().parent.parent
CODE_ARCHIVE = ROOT / "Code_Archive"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_ARCHIVE) not in sys.path:
    sys.path.insert(0, str(CODE_ARCHIVE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

runpy.run_path(str(CODE_ARCHIVE / "modern_detection_gui_optimized.py"), run_name="__main__")
