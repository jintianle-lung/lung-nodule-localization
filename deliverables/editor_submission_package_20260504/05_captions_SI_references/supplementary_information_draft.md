# Supplementary Information Draft

## Supplementary Methods

The tactile sensing dataset was collected as sequential 12 x 8 pressure matrices during active contact with lung-tissue specimens containing controlled nodule phantoms. Raw windows were normalized using a window-level min-max procedure before model inference. The V5 cascade first generated a detection probability and then performed conditional size and depth inversion on gate-positive windows. Feature analyses quantified FEM-guided descriptors including peak intensity, P95 amplitude, center-border contrast, hotspot radius, and second-moment spatial spread. Counterfactual cue ablations were evaluated under the same detection-gated cascade, with detection loss assessed at the gate and size/depth losses calculated only for samples that remained gate-positive.

## Supplementary Results

Supplementary tables summarize task-level AUC, model comparisons, size-class performance, depth-class performance, and failure-mode evidence. The error analysis supports three bounded interpretations: detection false positives often show nodule-like high-amplitude and high-contrast tactile responses consistent with hard heterogeneous contacts; size errors are more frequent in larger, deformation-prone contacts but remain near adjacent bins for most samples; and depth errors reflect weak and mechanically coupled diffusion cues that may be further distorted by contact sliding or target displacement.

## Supplementary Tables

- Table S1. AUC task summary.
- Table S2. Full model comparison.
- Table S3. Readout summary.
- Table S4. Size seven-class performance.
- Table S5. Depth three-class performance.
- Table S6. Failure mechanism summary.

## Supplementary Figure Candidates

- Counterfactual cue-family audit matrix.
- Single CAM panels for representative correct and failure cases.
- Global error-mechanism panels split into detection, size, and depth mechanisms.
