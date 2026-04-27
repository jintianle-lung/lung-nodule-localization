# Reviewer-Ready UI Bundle

This folder packages the optimized, runnable GUI version used for on-screen loading, CSV playback, and model inference.

Use the optimized GUI screenshot from this release when documenting the runnable software bundle.

## Run

From the repository root:

```bash
python ui_release/main.py
```

This launches `ui_release/modern_detection_gui_optimized.py` directly.
The GUI and its checkpoints live under `ui_release/`, so this bundle can run without the archive tree.

## Notes

- The current GUI loads the paper two-stage pipeline with the hierarchical inverter and current release weights from `ui_release/checkpoints/`.
- The UI opens even if the serial device is not connected; it can still be used for local CSV playback and layout verification.
