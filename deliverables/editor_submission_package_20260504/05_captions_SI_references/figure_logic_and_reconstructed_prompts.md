# Figure Logic and Reconstructed Prompt Intent

Date: 2026-05-04

This document records the logic behind the current figure package for manuscript writing and editor handoff. The prompt notes are reconstructed from the working conversation, filenames, generation scripts, and figure source data. They are not guaranteed to be verbatim user prompts.

## Core Story Closure

The intended manuscript logic is a closed evidence chain:

1. Surgical problem: preoperative CT gives an approximate target, but lung collapse, deformation, and limited thoracoscopic touch make intraoperative localization difficult.
2. Physical expectation: a hidden nodule should alter the tactile stress field during active pressing or sliding.
3. FEM/static feature layer: nodule-positive contacts show stronger focal stress, center-border contrast, high-response amplitude, and contour/spread descriptors.
4. Experimental feature layer: the strongest positive-vs-negative descriptors include center-border contrast, center pressure, global pressure variation, peak amplitude, high-response amplitude, temporal slope, peak persistence, and weaker spatial diffusion measures.
5. Model layer: the V5/R5 detection-gated cascade first identifies nodule-positive tactile windows, then performs conditional size and depth readout only for gate-positive windows.
6. Interpretability layer: V5 outputs and embeddings align with the same physically motivated cue families, and targeted counterfactual cue ablations degrade the gated readouts.
7. Error layer: residual errors are structured by tactile physics rather than random noise: false detections can be caused by hard heterogeneous contacts, size errors are concentrated in deformation-prone large contacts but often remain in nearby bins, and depth remains weak/auxiliary because diffusion cues are coupled with size, contact state, and possible sliding.

## Main Figure Logic

### Figure 1. Clinical Need and System Overview

Purpose: introduce the intraoperative localization problem and the proposed tactile sensing plus deep learning workflow.

Likely panels from the PDF page: clinical CT planning, alignment difficulty after lung collapse, thoracoscopic surgery workflow, sensor-to-laptop hardware chain, raw 12 x 8 pressure matrix, real-time software interface, probability traces, and detection/size/depth outputs.

Main message: the system converts dynamic tactile contact into objective runtime nodule evidence and conditional characterization, providing an adjunct to subjective manual palpation.

Main source assets:

- `01_pdf_pages_as_svg/MainFigure_page_01.svg`
- `05_captions_SI_references/pdf_extracted_text_by_page.csv`

Reconstructed prompt intent:

- "Open the main program and show the same six output-style maps from the V5 model."
- "Use English labels and make the interface clearer or split the result into larger panels."
- "The clinical case supports the problem statement: lung collapse can distort depth, so the model should provide practical intraoperative guidance rather than perfect CT registration."

Uncertain items:

- Whether Figure 1 should emphasize clinical workflow, hardware architecture, software interface, or all three equally.
- Whether the real-time interface screenshots are final or need replacement with cleaner English UI captures.

### Figure 2. Flexible Sensor and Ex Vivo Phantom Platform

Purpose: describe the tactile sensor head, readout chain, and lung-nodule phantom preparation.

Likely panels from the PDF page: nodule implantation, prepared specimen, depth probe, encapsulated sensing head, FPC cable, interface converter, signal conditioning, ADC board, sensor stack, applied stress schematic, lung tissue, pulmonary nodule, and non-nodular control.

Main message: the system has a physical sensor platform and controlled ex vivo specimen setup, not only a software model.

Main source assets:

- `01_pdf_pages_as_svg/MainFigure_page_02.svg`
- hardware/specimen photographs embedded in the source PDF

Reconstructed prompt intent:

- "Show the experimental platform clearly: sensor, lung tissue, nodule, depth control, and signal chain."
- "Keep the figure usable for a high-impact manuscript, with a clear hardware-to-data path."

Uncertain items:

- Exact sensor material stack needs user confirmation before Methods.
- The PDF text includes "liver specimen"; confirm whether it is a control/preparation image or an unrelated specimen photograph that should be removed or reframed.

### Figure 3. Dataset Construction and Preliminary Tactile Features

Purpose: explain data acquisition, labeling, design matrix, data split, and early tactile evidence.

Likely panels from the PDF page: raw 12 x 8 stress map, interpolated rendering, nodosity versus non-nodular examples, data records and label tags, repeated experimental procedure, experiment-wise train/validation/test split, and size-depth design matrix.

Main message: the dataset is structured by repeated experiments and controlled size-depth combinations, supporting both detection and conditional inversion.

Main source assets:

- `01_pdf_pages_as_svg/MainFigure_page_03.svg`
- raw/derived CSVs collected in `03_source_data_and_tables`
- raw-prior feature analysis from `figure_backup_rawprior_20260416/raw_prior_analysis`

Feature-analysis closure:

- Detection-positive contacts showed large differences in center-border contrast, center pressure, global pressure variation, and peak amplitude.
- In the physical-prior feature table, center-border contrast had effect_rbc about 0.62 and peak amplitude about 0.59 for positive versus negative contacts.
- High-response amplitude and pressure contour descriptors motivate the later size interpretation.
- Spatial spread and hotspot radius were significant but weaker and more coupled, matching the later depth interpretation.

FEM/contact-simulation closure:

- The available FEM document describes a simplified Ansys Static Structural contact model with a 45 mm x 50 mm x 1 mm pressure-sensor layer, an equivalent lung-tissue layer, and spherical nodules with diameters of 2.5, 5.0, and 7.5 mm.
- Depth comparisons in the document use a 5 mm nodule at depths of 2, 5, and 7 mm; the stated load is 25 N, applied vertically under simplified static conditions.
- The simulation output is total deformation, used as a proxy for contact-response distribution. This supports the qualitative mechanism that stiff embedded structures alter surface tactile patterns, larger nodules alter spatial footprint, and deeper nodules produce weaker surface evidence.
- Writing boundary: this FEM layer should be presented as mechanical plausibility evidence, not as a patient-specific lung mechanics model. Material settings include highly simplified surrogates such as a steel-like nodule, so Methods must explicitly state the simplifications.

Supporting documents:

- `10_original_supporting_documents/FEM_contact_simulation_docx_xwechat_20260503.docx`
- `10_original_supporting_documents/FEM_contact_simulation_docx_literature_enhanced.docx`
- `10_original_supporting_documents/FEM_contact_simulation_docx_figure_notes.docx`
- Extracted text files in the same folder for manuscript planning.

Reconstructed prompt intent:

- "First look at what the FEM/static tests and experimental violin plots told us."
- "The later interpretability figure must close the loop with the feature analysis."
- "Detection should correspond to abnormal stress concentration; size should correspond to contour or edge; depth should correspond to diffusion or spread."

Uncertain items:

- The PDF text says "6 sizes (0.25-1.75 cm)", while later analyses use seven size bins: 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, and 1.75 cm. This needs final correction.
- Need final sample counts per experiment/repetition for Methods.

### Figure 4. Model Architecture and Performance

Purpose: define the model and show held-out performance.

Likely panels from the PDF page: detection gate, frame encoder, MS-TCN temporal encoder, residual morphology branch, feature fusion, detailed module architecture, and performance panels.

Main message: the final model is a detection-gated cascade. Detection is primary; size and depth are conditional readouts rather than unconditional independent outputs.

Model logic from local docs:

- Input: 10-frame tactile window from 12 x 8 pressure matrices.
- Backbone: time-distributed/shared CNN frame encoder plus multi-scale temporal convolution.
- Gate: detection probability controls whether size and depth should be displayed.
- Size branch: residual morphology/fusion branch predicts seven size bins and continuous size.
- Depth branch: coarse depth readout, used as auxiliary guidance.

Performance numbers currently available:

- Detection, all held-out windows: sensitivity 92.6 percent, specificity 72.9 percent, TP 1282, FP 641, FN 103.
- Size, detected positive windows: exact 75.9 percent, adjacent/exact 88.2 percent, Top-2 90.6 percent.
- Depth, detected positive windows: exact 61.1 percent, adjacent/exact 86.2 percent, Top-2 85.2 percent.

Main source assets:

- `01_pdf_pages_as_svg/MainFigure_page_04.svg`
- `manuscript_table_pack_20260504/csv/Table_S3_Readout_summary.csv`
- `docs/MODEL_AND_ALGO.md`
- `docs/MODEL_AND_SYSTEM_BLUEPRINT.md`
- `docs/ALGORITHM_STEP_BY_STEP_20260429_CN.md`

Reconstructed prompt intent:

- "Do not put all AUC plots in the main figure; package AUC as table data and decide placement later."
- "Model comparison and performance should support the story, but the remaining main figure space should focus on interpretability and error explanation."

Uncertain items:

- Final model name should be standardized: V5/R5, DG-TIN, detection-gated cascade, frozen detector plus residual inversion, or another final manuscript name.
- Need visual confirmation of all Figure 4 panels because the current PDF text extraction is noisy.

### Figure 5. Mechanistic Interpretability and Error Explanation

Purpose: show that the model learned tactile features consistent with the physical and experimental evidence, then explain structured failure modes.

Likely panels from the PDF page: V5 learned-feature alignment, feature-guided counterfactual intervention, CAM examples, model comparison/table-related panels, and error mechanism panels.

Main message: the model does not merely classify; it uses tactile cue families that are consistent with the experimental feature analysis, and its errors can be interpreted through the same mechanics.

Main source assets:

- `01_pdf_pages_as_svg/MainFigure_page_05.svg`
- `v5_feature_learning_final6_20260504/figure5_v5_learned_features_candidate.svg`
- `fig5_causal_intervention_20260504/Fig5F_feature_guided_gated_ablation_bar.svg`
- `cam_feature_alignment_20260504/G_cam_mechanism_error_sixpack.svg`
- `error_mechanism_global_20260504/Fig_error_mechanism_global_summary.svg`
- `fig5_intuitive_explainability_20260504/Fig5_overall_explainability_story.svg`
- `fig5_intuitive_explainability_20260504/Fig5_failure_mode_visual_cards.svg`

Reconstructed prompt intent:

- "For panel F, use causal/counterfactual intervention and make it align with the earlier feature conclusions."
- "Detection should look at hotspots; size should look at contour/edge; depth should look at spread/diffusion."
- "Use gated logic because size and depth are only opened after detection."
- "Do not overclaim the intervention as a unique causal tissue law; write it as model reliance on feature-guided cue families."
- "CAM examples should include correct cases plus failed cases, and the failed cases must support the error mechanism discussion."
- "Keep AUC/table data outside the figure and pack it separately for later insertion."

Uncertain items:

- Confirm which Figure 5 components remain in the main text and which move to SI.
- Confirm whether the final CAM panel should be the sixpack, clean single panels, or the more intuitive story cards.

## Figure 5 Detailed Evidence Chain

### Feature Alignment Panel

Question addressed: did the V5 model encode the same features found by FEM/static testing and experimental feature analysis?

Evidence:

- Output-feature correlations: detection probability correlated with peak intensity (rho 0.731) and center-border contrast (rho 0.693).
- Size readout correlated with high-response amplitude (rho 0.676) and center contrast (rho 0.727).
- Depth expected value correlated more weakly with hotspot radius and spatial spread (rho 0.291 each), supporting the claim that depth is auxiliary and mechanically difficult.
- Latent probes found that frozen detector and fused inversion embeddings strongly encode peak, amplitude, contrast, radius, and spread descriptors.

Source data:

- `v5_feature_learning_final6_20260504/v5_output_descriptor_alignment.csv`
- `v5_feature_learning_final6_20260504/v5_latent_descriptor_probe.csv`
- `v5_feature_learning_final6_20260504/run_summary.json`

Manuscript wording boundary:

- Good: "The trained representations retained measurable information about physically motivated tactile descriptors."
- Avoid: "The model proves the true causal mechanical law."

### Panel F. Feature-Guided Gated Counterfactual Ablation

Question addressed: when physically meaningful tactile cues are removed, do the gated model readouts degrade?

Final preferred asset:

- `fig5_causal_intervention_20260504/Fig5F_feature_guided_gated_ablation_bar.svg`

Source data:

- `fig5_causal_intervention_20260504/Fig5F_gated_mechanism_closure.csv`

Cue families:

- Stress hotspot: amplitude/peak evidence.
- Boundary contrast: center-border contrast and edge/contour evidence.
- Spatial spread: hotspot radius and second-moment spread evidence.
- Temporal control: order shuffle control to show that the large losses are not caused by arbitrary perturbation alone.

Observed losses:

- Stress hotspot masking: gate loss 6.5 percent, size loss 31.4 percent, depth loss 31.8 percent.
- Boundary contrast masking: gate loss 17.2 percent, size loss 36.9 percent, depth loss 39.7 percent.
- Spatial spread contraction: gate loss 14.0 percent, size loss 58.1 percent, depth loss 44.5 percent.
- Temporal control: much smaller losses, about 0.4 percent for gate, 2.3 percent for size, and 4.5 percent for depth.

Interpretation:

- The losses are not one-to-one labels of "detection feature", "size feature", and "depth feature". The tactile field is physically coupled.
- The correct logic is: cue families identified in earlier feature analysis are necessary for stable gated readout, while the largest sensitivity differs by output.
- Spatial spread affects size and depth because spread also changes the apparent footprint of a deformed target. This supports coupling rather than contradicting the feature analysis.

Reconstructed prompt intent:

- "Why are there three masks if the feature analysis had six features?"
- "Use the six features as three mechanistic cue families, then evaluate the gated cascade using accuracy/loss rather than mixed confidence scales."
- "Make the text short and figure F-slot friendly."

### Panel G. CAM Mechanism and Error Sixpack

Question addressed: can representative CAM examples visually match the mechanism claims?

Final preferred asset:

- `cam_feature_alignment_20260504/G_cam_mechanism_error_sixpack.svg`

Clean single-panel assets:

- `cam_feature_alignment_20260504/single_cam_panels/G_cam_A_detection_hotspot_clean.svg`
- `cam_feature_alignment_20260504/single_cam_panels/G_cam_B_size_contour_clean.svg`
- `cam_feature_alignment_20260504/single_cam_panels/G_cam_C_depth_diffusion_clean.svg`
- `cam_feature_alignment_20260504/single_cam_panels/G_cam_D_detection_heterogeneity_clean.svg`
- `cam_feature_alignment_20260504/single_cam_panels/G_cam_E_size_deformation_clean.svg`
- `cam_feature_alignment_20260504/single_cam_panels/G_cam_F_depth_weak_separability_clean.svg`

Source data:

- `cam_feature_alignment_20260504/G_cam_mechanism_error_sixpack_sources.json`

Mechanism examples:

- Detection learns hotspot: 1.0 cm shallow case, predicted 1.00 cm shallow, CAM locks on focal high-response hotspot.
- Size learns contour: 1.75 cm deep case, predicted 1.75 cm deep, CAM follows footprint/contour.
- Depth uses diffusion: same deep case, depth CAM emphasizes broader spread and position.

Error examples:

- Detection error: a negative window is predicted positive with P > gate and predicted size around 1.25 cm; interpretation should be "hard heterogeneous/non-nodule contact can mimic a hotspot" unless the user confirms bronchus-like anatomy.
- Size error: 1.25 cm case predicted near 1.5 cm; CAM expands over a broad deformed footprint, consistent with adjacent-bin deformation error.
- Depth error: 0.5 cm deep case predicted as middle; detection and size remain correct but CAM is shifted, consistent with weak spread cue and contact/sliding ambiguity.

Reconstructed prompt intent:

- "Find three examples to show detection-hotspot, size-contour, depth-diffusion."
- "Then find three error examples: detection can hit heterogeneous structures, size errors can come from deformation, and depth errors can come from weak physical cues plus possible sliding."
- "Make each single figure clean, centered, without A/B/C labels if used alone; red box marks the maximum CAM region and can be explained in the caption."

### Panel H or SI. Global Error Mechanism

Question addressed: can we explain error mechanisms from the whole test set rather than cherry-picked examples?

Assets:

- `error_mechanism_global_20260504/Fig_error_mechanism_global_summary.svg`
- `error_mechanism_global_20260504/Error_detection_FP_hard_hotspot.svg`
- `error_mechanism_global_20260504/Error_size_larger_deformable_contacts.svg`
- `error_mechanism_global_20260504/Error_depth_weak_diffusion_cue.svg`

Source data:

- `error_mechanism_global_20260504/global_error_mechanism_summary.csv`
- `manuscript_table_pack_20260504/csv/Table_S6_Failure_mechanism_summary.csv`

Dataset-level conclusions:

- Detection false positives: 641 FPs; median P95 77.2 versus 3.0 in true negatives; median contrast 20.4 versus 0.2. This supports "hard heterogeneous contact can mimic nodule-like focal evidence."
- Size errors: exact size accuracy is lower for larger targets (>=1.0 cm: 61.6 percent) than small targets (<=0.75 cm: 90.0 percent), while Top-2 remains 90.6 percent overall. This supports "deformation-prone larger contacts cause exact-bin ambiguity but near-bin guidance remains useful."
- Depth errors: exact depth is 61.1 percent and Top-2 is 85.2 percent; spread-depth and radius-depth correlations are weak. This supports "depth is a bounded auxiliary readout, not a precise standalone inversion."

Reconstructed prompt intent:

- "From the overall data, explain why errors happen, not just from selected CAM panels."
- "Detection wrong cases may contact other hard or heterogeneous structures and can be excluded by operator cross-check with size/depth/shape."
- "Size wrong cases are more common in large nodules because pressing deformation changes the apparent footprint, but nearby-bin guidance is still useful."
- "Depth errors should be written honestly as weak/coupled physics plus possible target displacement during pressing."

## Tables and Data Package Logic

The AUC and performance tables are intentionally outside the main figure package for later placement. They should be inserted as main or supplementary tables after the manuscript layout is fixed.

Table pack:

- `manuscript_table_pack_20260504/manifest.json`
- `manuscript_table_pack_20260504/csv/Table_S1_AUC_task_summary.csv`
- `manuscript_table_pack_20260504/csv/Table_S2_AUC_full_model_comparison.csv`
- `manuscript_table_pack_20260504/csv/Table_S3_Readout_summary.csv`
- `manuscript_table_pack_20260504/csv/Table_S4_Size_7class.csv`
- `manuscript_table_pack_20260504/csv/Table_S5_Depth_3class.csv`
- `manuscript_table_pack_20260504/csv/Table_S6_Failure_mechanism_summary.csv`

Reconstructed prompt intent:

- "Do not put AUC curves in the figure now. Put the table data aside and decide insertion later."
- "Size seven-class and depth three-class tables should support the exact, adjacent, and Top-2 discussion."
- "Use the remaining figure space for the mechanism closure and error explanation."

## Prompt Intent Archive

These are the main recoverable user intents from the working history, paraphrased in English:

1. Make the clinical V5 output resemble the main software output, with six clear panels and English labels.
2. Build the interpretability section so it closes the loop with earlier FEM static tests and experimental feature violin plots.
3. Use one F panel for counterfactual or causal-style intervention, but keep it rigorous and aligned with feature analysis.
4. Detection should be explained by abnormal stress concentration or focal hotspot.
5. Size should be explained by contour, edge, footprint, amplitude, and deformation.
6. Depth should be explained by diffusion/spread, while admitting that depth is physically weak and coupled.
7. CAM images must be visually clean and should include successful cases and failure cases.
8. False detection should be discussed as heterogeneous/hard non-nodule contact, not overclaimed as bronchus unless confirmed.
9. Size failure should be linked to deformation, especially in larger contacts and around the 1 cm scale.
10. Depth failure should be linked to weak separability, sliding, and displacement during pressing.
11. AUC curves should not occupy main figure space; keep tables and supporting numerical data packaged for later manuscript insertion.
12. Final package should provide SVG assets, source data, scripts, captions, SI notes, references, and uncertainty notes for editor/submission use.

## Submission Wording Boundaries

Use careful language:

- "supports", "is consistent with", "suggests", "provides evidence that"
- "detection-gated", "conditional size/depth readout", "auxiliary depth guidance"
- "hard heterogeneous contact" unless user confirms bronchus or another anatomical source
- "nearby size band" or "Top-2 tolerance" for size rather than claiming perfect size classification
- "depth guidance" rather than "precise depth reconstruction"

Avoid overclaims:

- Do not claim true causal tissue mechanics from post-hoc ablation alone.
- Do not claim depth is independent of size or contact state.
- Do not claim false positives are definitively bronchus without anatomical confirmation.
- Do not claim the ex vivo phantom exactly reproduces all in vivo surgical mechanics.
