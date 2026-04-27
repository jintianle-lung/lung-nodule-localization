# Reviewer-Ready UI Bundle

This folder packages the optimized, runnable GUI version used for on-screen loading, CSV playback, and model inference.

Use the optimized GUI screenshot from this release when documenting the runnable software bundle.

## Run

From the repository root:

```bash
python ui_release/main.py
```

This launches `Code_Archive/modern_detection_gui_optimized.py` directly.

The legacy top-level script is still available, but it is not the preferred reviewer entry point:

```bash
python main_gui.py
```

## Notes

- The current GUI uses a legacy-compatible checkpoint path under `Code_Archive/discussion/dualstream_3dcnn_lstm_active.pth`.
- The checkpoint can be swapped later for the newest weight file without changing the UI entry point.
- The UI opens even if the serial device is not connected; it can still be used for local CSV playback and layout verification.
