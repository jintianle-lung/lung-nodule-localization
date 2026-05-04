from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "deliverables").exists():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "deliverables").exists() and (candidate / "latest_algorithm").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root(Path(__file__).resolve().parent)
PKG = ROOT / "deliverables" / "editor_submission_package_20260504"
DOCS = PKG / "05_captions_SI_references"
PREFERRED = PKG / "07_preferred_main_and_si_figures"
CURATED = PKG / "08_curated_source_data_for_figures"
AUDIT = PKG / "09_integrity_audit"


PREFERRED_FIGURES = [
    ("Main_Figure_1_clinical_system_overview.svg", PKG / "01_pdf_pages_as_svg" / "MainFigure_page_01.svg", "Current composed PDF page for Figure 1."),
    ("Main_Figure_2_sensor_and_phantom_platform.svg", PKG / "01_pdf_pages_as_svg" / "MainFigure_page_02.svg", "Current composed PDF page for Figure 2."),
    ("Main_Figure_3_dataset_and_feature_characterization.svg", PKG / "01_pdf_pages_as_svg" / "MainFigure_page_03.svg", "Current composed PDF page for Figure 3."),
    ("Main_Figure_4_model_architecture_and_performance.svg", PKG / "01_pdf_pages_as_svg" / "MainFigure_page_04.svg", "Current composed PDF page for Figure 4."),
    ("Main_Figure_5_interpretability_and_errors.svg", PKG / "01_pdf_pages_as_svg" / "MainFigure_page_05.svg", "Current composed PDF page for Figure 5."),
    ("Fig5F_feature_guided_gated_ablation_bar.svg", ROOT / "deliverables" / "fig5_causal_intervention_20260504" / "Fig5F_feature_guided_gated_ablation_bar.svg", "Preferred F panel for feature-guided gated counterfactual ablation."),
    ("Fig5G_CAM_mechanism_error_sixpack.svg", ROOT / "deliverables" / "cam_feature_alignment_20260504" / "G_cam_mechanism_error_sixpack.svg", "Preferred CAM mechanism/error sixpack."),
    ("Fig5_error_mechanism_global_summary.svg", ROOT / "deliverables" / "error_mechanism_global_20260504" / "Fig_error_mechanism_global_summary.svg", "Global dataset-level error mechanism summary."),
    ("SI_Error_detection_FP_hard_hotspot.svg", ROOT / "deliverables" / "error_mechanism_global_20260504" / "Error_detection_FP_hard_hotspot.svg", "Split error panel for detection false positives."),
    ("SI_Error_size_larger_deformable_contacts.svg", ROOT / "deliverables" / "error_mechanism_global_20260504" / "Error_size_larger_deformable_contacts.svg", "Split error panel for size errors."),
    ("SI_Error_depth_weak_diffusion_cue.svg", ROOT / "deliverables" / "error_mechanism_global_20260504" / "Error_depth_weak_diffusion_cue.svg", "Split error panel for depth errors."),
    ("SI_Fig5_overall_explainability_story.svg", ROOT / "deliverables" / "fig5_intuitive_explainability_20260504" / "Fig5_overall_explainability_story.svg", "Intuitive task-cue-readout explainability summary."),
    ("SI_Fig5_failure_mode_visual_cards.svg", ROOT / "deliverables" / "fig5_intuitive_explainability_20260504" / "Fig5_failure_mode_visual_cards.svg", "Intuitive failure-mode card summary."),
    ("SI_V5_learned_features_candidate.svg", ROOT / "deliverables" / "v5_feature_learning_final6_20260504" / "figure5_v5_learned_features_candidate.svg", "V5 feature-learning alignment candidate."),
]


CURATED_DATA = [
    ("Fig3_physical_prior_feature_significance.csv", ROOT / "deliverables" / "figure_backup_rawprior_20260416" / "raw_prior_analysis" / "physical_prior_feature_significance.csv", "Experimental/FEM-guided physical-prior feature significance."),
    ("Fig3_all_significant_features_fdr005.csv", ROOT / "deliverables" / "figure_backup_rawprior_20260416" / "raw_prior_analysis" / "all_significant_features_fdr005.csv", "All significant features under FDR 0.05."),
    ("Fig5_v5_output_descriptor_alignment.csv", ROOT / "deliverables" / "v5_feature_learning_final6_20260504" / "v5_output_descriptor_alignment.csv", "Correlation between V5 outputs and tactile descriptors."),
    ("Fig5_v5_latent_descriptor_probe.csv", ROOT / "deliverables" / "v5_feature_learning_final6_20260504" / "v5_latent_descriptor_probe.csv", "Latent probe results for descriptor encoding."),
    ("Fig5_v5_targeted_perturbation_response.csv", ROOT / "deliverables" / "v5_feature_learning_final6_20260504" / "v5_targeted_perturbation_response.csv", "Targeted perturbation response by output metric."),
    ("Fig5F_gated_mechanism_closure.csv", ROOT / "deliverables" / "fig5_causal_intervention_20260504" / "Fig5F_gated_mechanism_closure.csv", "Feature-guided gated counterfactual ablation source data."),
    ("Fig5G_cam_mechanism_error_sixpack_sources.json", ROOT / "deliverables" / "cam_feature_alignment_20260504" / "G_cam_mechanism_error_sixpack_sources.json", "CAM panel source cases and labels."),
    ("Fig5_error_mechanism_global_summary.csv", ROOT / "deliverables" / "error_mechanism_global_20260504" / "global_error_mechanism_summary.csv", "Dataset-level error mechanism source data."),
    ("Table_S1_AUC_task_summary.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S1_AUC_task_summary.csv", "Task-level AUC summary."),
    ("Table_S2_AUC_full_model_comparison.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S2_AUC_full_model_comparison.csv", "Full AUC model comparison."),
    ("Table_S3_Readout_summary.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S3_Readout_summary.csv", "Readout-level performance summary."),
    ("Table_S4_Size_7class.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S4_Size_7class.csv", "Size seven-class performance by true size."),
    ("Table_S5_Depth_3class.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S5_Depth_3class.csv", "Depth three-class performance by true depth."),
    ("Table_S6_Failure_mechanism_summary.csv", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "csv" / "Table_S6_Failure_mechanism_summary.csv", "Failure mechanism summary."),
    ("manuscript_table_pack_manifest.json", ROOT / "deliverables" / "manuscript_table_pack_20260504" / "manifest.json", "Manifest for table pack."),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(native_path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def file_size(path: Path) -> int:
    return os.stat(native_path(path)).st_size


def copy_with_index(items: list[tuple[str, Path, str]], out_dir: Path, index_name: str) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for dest_name, src, note in items:
        dst = out_dir / dest_name
        status = "copied" if src.exists() else "missing"
        if src.exists():
            shutil.copy2(src, dst)
        rows.append(
            {
                "file": dest_name,
                "source_path": str(src),
                "status": status,
                "size_bytes": str(dst.stat().st_size) if dst.exists() else "",
                "sha256": sha256_file(dst) if dst.exists() else "",
                "note": note,
            }
        )
    with (out_dir / index_name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source_path", "status", "size_bytes", "sha256", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_svgs() -> list[dict[str, str]]:
    rows = []
    for svg in sorted(PKG.rglob("*.svg")):
        try:
            size = file_size(svg)
            ET.parse(native_path(svg))
            status = "ok"
            error = ""
        except Exception as exc:
            try:
                size = file_size(svg)
            except Exception:
                size = 0
            status = "parse_error"
            error = str(exc)
        rows.append(
            {
                "relative_path": str(svg.relative_to(PKG)),
                "size_bytes": str(size),
                "status": status,
                "error": error,
            }
        )
    return rows


def audit_tables() -> list[dict[str, str]]:
    rows = []
    for path in sorted(CURATED.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
            n_rows = max(0, len(data) - 1)
            n_cols = len(data[0]) if data else 0
            status = "ok" if data else "empty"
            error = ""
        except Exception as exc:
            n_rows = 0
            n_cols = 0
            status = "read_error"
            error = str(exc)
        rows.append(
            {
                "file": path.name,
                "rows_excluding_header": str(n_rows),
                "columns": str(n_cols),
                "status": status,
                "error": error,
            }
        )
    return rows


def package_inventory() -> list[dict[str, str]]:
    rows = []
    for path in sorted(p for p in PKG.rglob("*") if p.is_file()):
        try:
            size = file_size(path)
            digest = sha256_file(path)
        except Exception as exc:
            size = 0
            digest = f"ERROR: {exc}"
        rows.append(
            {
                "relative_path": str(path.relative_to(PKG)),
                "size_bytes": str(size),
                "sha256": digest,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_zip() -> tuple[Path, int, str]:
    zip_dir = ROOT / "deliverables" / "_local_zip_archives_not_for_git_20260504"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / "editor_submission_package_20260504.zip"
    if zip_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path.replace(zip_dir / f"editor_submission_package_20260504_previous_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in PKG.rglob("*") if p.is_file()):
            try:
                zf.write(native_path(path), path.relative_to(PKG.parent))
            except FileNotFoundError:
                continue
    return zip_path, zip_path.stat().st_size, sha256_file(zip_path)


def write_index_md(summary: dict[str, object], preferred_rows: list[dict[str, str]], data_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Editor Submission Package Index",
        "",
        f"Created/updated: {summary['updated_at']}",
        "",
        "## Package Summary",
        "",
        f"- Package root: `{PKG}`",
        f"- Preferred figure SVGs: {summary['preferred_figure_count']}",
        f"- Curated source data files: {summary['curated_source_data_count']}",
        f"- All SVG files in package: {summary['all_svg_count']}",
        f"- All package files: {summary['total_file_count']}",
        f"- Package size: {summary['total_size_mb']} MB",
        f"- ZIP archive: `{summary['zip_path']}`",
        f"- ZIP SHA256: `{summary['zip_sha256']}`",
        "",
        "## Recommended Figure Entry Points",
        "",
    ]
    for row in preferred_rows:
        lines.append(f"- `{row['file']}`: {row['note']}")
    lines.extend(["", "## Curated Source Data Entry Points", ""])
    for row in data_rows:
        lines.append(f"- `{row['file']}`: {row['note']}")
    lines.extend(
        [
            "",
            "## Key Draft Documents",
            "",
            "- `05_captions_SI_references/figure_captions_draft.md`",
            "- `05_captions_SI_references/supplementary_information_draft.md`",
            "- `05_captions_SI_references/figure_logic_and_reconstructed_prompts.md`",
            "- `05_captions_SI_references/references_seed_expanded.bib`",
            "- `05_captions_SI_references/uncertain_items_for_user_confirmation.md`",
            "",
            "## Audit Files",
            "",
            "- `09_integrity_audit/package_file_inventory_sha256.csv`",
            "- `09_integrity_audit/svg_validation_report.csv`",
            "- `09_integrity_audit/curated_csv_readability_report.csv`",
            "- `09_integrity_audit/final_package_summary.json`",
            "",
            "PNG-only assets wrapped in SVG are visually preserved raster images, not true vector traces. Native SVG files should be used whenever available.",
            "",
        ]
    )
    (PKG / "SUBMISSION_PACKAGE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    preferred_rows = copy_with_index(PREFERRED_FIGURES, PREFERRED, "_preferred_figure_index.csv")
    data_rows = copy_with_index(CURATED_DATA, CURATED, "_curated_source_data_index.csv")

    svg_rows = validate_svgs()
    write_csv(AUDIT / "svg_validation_report.csv", svg_rows, ["relative_path", "size_bytes", "status", "error"])

    table_rows = audit_tables()
    write_csv(AUDIT / "curated_csv_readability_report.csv", table_rows, ["file", "rows_excluding_header", "columns", "status", "error"])

    inventory_rows = package_inventory()
    write_csv(AUDIT / "package_file_inventory_sha256.csv", inventory_rows, ["relative_path", "size_bytes", "sha256"])

    zip_path, zip_size, zip_sha = make_zip()

    all_files = [p for p in PKG.rglob("*") if p.is_file()]
    total_size = 0
    for p in all_files:
        try:
            total_size += file_size(p)
        except Exception:
            pass
    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "package_root": str(PKG),
        "preferred_figure_count": sum(1 for r in preferred_rows if r["status"] == "copied"),
        "preferred_figure_missing": [r for r in preferred_rows if r["status"] != "copied"],
        "curated_source_data_count": sum(1 for r in data_rows if r["status"] == "copied"),
        "curated_source_data_missing": [r for r in data_rows if r["status"] != "copied"],
        "all_svg_count": len([p for p in all_files if p.suffix.lower() == ".svg"]),
        "svg_parse_errors": [r for r in svg_rows if r["status"] != "ok"],
        "curated_csv_read_errors": [r for r in table_rows if r["status"] != "ok"],
        "total_file_count": len(all_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_size,
        "zip_size_mb": round(zip_size / 1024 / 1024, 2),
        "zip_sha256": zip_sha,
        "git_note": "ZIP may exceed preferred GitHub size. Commit folder assets and audit files; keep ZIP local unless explicitly needed.",
    }
    (AUDIT / "final_package_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index_md(summary, preferred_rows, data_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
