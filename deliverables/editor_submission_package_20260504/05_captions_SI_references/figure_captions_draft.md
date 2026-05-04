# Draft Figure Captions

These captions are draft text for manuscript preparation. Items marked uncertain need user confirmation before submission.

## Figure 1. Clinical motivation and system overview for thoracoscopic tactile nodule localization

**Status:** draft

Figure 1. Clinical motivation and system overview for thoracoscopic tactile nodule localization. Preoperative computed tomography provides approximate lesion location, but intraoperative lung collapse and deformation can distort the spatial relationship between the preoperative plan and the operative field. The proposed workflow combines a flexible tactile sensor, signal acquisition hardware, and a deep-learning inference interface to convert 12 x 8 pressure matrix sequences into real-time nodule detection, size estimation, depth estimation, heatmap visualization, and probability traces. The system is intended to support intraoperative palpation-like assessment while reducing reliance on subjective manual palpation alone.

**Need confirmation:** Confirm whether Figure 1 should emphasize clinical workflow, software interface, hardware architecture, or all three.

## Figure 2. Flexible tactile sensing head and ex vivo lung-nodule phantom platform

**Status:** draft

Figure 2. Flexible tactile sensing head and ex vivo lung-nodule phantom platform. The tactile probe integrates an encapsulated sensing head, flexible printed-circuit connection, interface conversion, signal conditioning, and an ADC-based readout module to acquire a 12 x 8 pressure matrix during contact. The experimental platform used implanted nodules with controlled sizes and depths in lung-tissue specimens, enabling systematic evaluation of nodule and non-nodule contact responses under applied stress. Structural schematics show the sensing layer, electrode configuration, protective shell, and contact relationship between the sensor, lung tissue, and embedded nodule.

**Need confirmation:** Confirm exact sensor material stack and whether the specimen is lung only or includes liver/other tissue controls.

## Figure 3. Dataset construction, labeling, and preliminary tactile-response characterization

**Status:** draft

Figure 3. Dataset construction, labeling, and preliminary tactile-response characterization. Raw tactile frames were recorded as 12 x 8 stress matrices and rendered as interpolated pressure maps for visualization. A controlled design matrix spanning multiple nodule diameters and burial depths was acquired across repeated experiments, with nodule and non-nodule windows labeled for detection and downstream inversion. Data were partitioned by experimental repeat to reduce leakage between training, validation, and testing. Preliminary analysis shows distinct stress-map patterns between nodular and non-nodular contacts and motivates subsequent feature-guided model development.

**Need confirmation:** Confirm final exact design matrix: the PDF text shows six sizes in one place, but other analyses use seven size bins from 0.25 to 1.75 cm.

## Figure 4. Model development and performance evaluation

**Status:** uncertain

Figure 4. Model development and performance evaluation. The detection and inversion pipeline uses a deep-learning cascade to first identify nodule-positive tactile windows and then estimate nodule size and depth from gate-positive sequences. Performance is evaluated on held-out experimental data using detection metrics and conditional size/depth readouts. Model comparisons and task-specific curves summarize the benefit of the V5 cascade relative to alternative baselines, while confusion-style summaries indicate the residual difficulty of exact size and depth classification under mechanically coupled tactile responses.

**Need confirmation:** Need visual confirmation of page 4 panel content and final model names before this caption is locked.

## Figure 5. Mechanistic interpretability, counterfactual cue ablation, CAM visualization, and error analysis

**Status:** draft

Figure 5. Mechanistic interpretability, counterfactual cue ablation, CAM visualization, and error analysis. FEM-guided feature analysis identified three tactile cue families associated with the cascade outputs: abnormal stress hotspots and center-border contrast for detection, high-response amplitude and edge/contour contrast for size inversion, and spatial diffusion descriptors including hotspot radius and second-moment spread for depth inversion. Feature-guided counterfactual ablation was evaluated under the detection-gated cascade: the detection gate was assessed first, and size and depth losses were computed only for gate-positive samples. CAM visualizations further illustrate representative correct and failed cases, showing hotspot-focused detection, contour-related size estimation, and diffusion-related depth estimation. Global error analysis indicates that detection false positives resemble hard heterogeneous contacts, size errors are more common in deformation-prone larger contacts but often remain close to adjacent bins, and depth remains the most physically limited output because diffusion cues are weak and can be distorted by sliding or displacement during palpation.

**Need confirmation:** Confirm whether all interpretability/error panels are intended as Figure 5 or split between main Figure 5 and SI.

