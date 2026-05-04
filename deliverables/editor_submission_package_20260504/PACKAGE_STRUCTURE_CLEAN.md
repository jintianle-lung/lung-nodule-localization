# Clean Package Structure

Updated: 2026-05-04

This package has two layers:

1. Clean working layer for manuscript writing and editor handoff.
2. History layer for full dumps, superseded drafts, and conversion audits.

No files were deleted during cleanup. Redundant or full-dump assets were moved into `99_history_full_dump_and_superseded_assets`, and the move log is stored at `99_history_full_dump_and_superseded_assets/_move_log.csv`.

## Clean Working Layer

- `SUBMISSION_PACKAGE_INDEX.md`: start here.
- `00_original_pdf`: original WPS-exported manuscript figure PDF.
- `01_pdf_pages_as_svg`: five main figure pages converted to SVG.
- `04_generation_scripts`: scripts used for figure/data packaging and figure generation.
- `05_captions_SI_references`: captions, SI draft, figure logic, references, and uncertain items.
- `07_preferred_main_and_si_figures`: final/preferred figure entry points for main text and SI.
- `08_curated_source_data_for_figures`: figure-specific data sources and table CSVs.
- `09_integrity_audit`: SHA256 inventory, SVG validation, CSV readability checks, and final package summary.
- `10_original_supporting_documents`: original user-provided PDF/CSV/DOCX materials and extracted text previews.

## History Layer

- `99_history_full_dump_and_superseded_assets/full_svg_dump`: all previously collected SVG files from `deliverables`.
- `99_history_full_dump_and_superseded_assets/full_source_data_dump`: all previously collected CSV/JSON/XLSX files from `deliverables`.
- `99_history_full_dump_and_superseded_assets/conversion_audit`: PNG/PDF-to-SVG wrapping/conversion audit outputs.
- `99_history_full_dump_and_superseded_assets/superseded_reference_seed`: the first small reference seed, superseded by `references_seed_expanded.bib`.

## Practical Use

Use `07_preferred_main_and_si_figures` and `08_curated_source_data_for_figures` for manuscript assembly. Use the history layer only when tracing older figure candidates, checking full provenance, or recovering a discarded variant.
