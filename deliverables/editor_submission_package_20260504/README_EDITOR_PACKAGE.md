# Editor Submission Package

This package collects figure assets, source data, generation scripts, draft captions, Supplementary Information notes, references, original supporting documents, and integrity reports for manuscript preparation.

Start with `SUBMISSION_PACKAGE_INDEX.md`.

## Folder Structure

- `00_original_pdf`: original WPS-exported PDF.
- `01_pdf_pages_as_svg`: each PDF page converted to SVG.
- `04_generation_scripts`: Python scripts used to generate the final interpretability, error-analysis, and table figures.
- `05_captions_SI_references`: draft figure captions, SI draft, uncertain items, extracted PDF text, and seed references.
- `07_preferred_main_and_si_figures`: clean entry point for figures most likely to be used in the manuscript or SI.
- `08_curated_source_data_for_figures`: clean entry point for the specific data supporting preferred figures and tables.
- `09_integrity_audit`: file inventory, SHA256 hashes, SVG parse report, CSV readability report, and package summary.
- `10_original_supporting_documents`: user-supplied PDF/CSV/DOCX source materials and extracted text previews.
- `99_history_full_dump_and_superseded_assets`: moved historical/full-dump material. Nothing was deleted.

## Notes

The caption and SI files are drafts. Items in `uncertain_items_for_user_confirmation.md` should be resolved before final manuscript writing.

`figure_logic_and_reconstructed_prompts.md` records the figure-by-figure evidence chain, source files, reconstructed prompt intent, and wording boundaries. Prompt notes are reconstructed from the working history and should not be treated as verbatim chat logs.

PNG-to-SVG wrapping preserves visual appearance but does not create true vector artwork. Prefer native SVG files from `07_preferred_main_and_si_figures` whenever available.

The full flattened SVG archive, full flattened source-data archive, and PNG/PDF conversion audit were moved into `99_history_full_dump_and_superseded_assets` to keep the main package easier to inspect.
