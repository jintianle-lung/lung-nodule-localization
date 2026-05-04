from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Pillow is required to inspect PNG dimensions: {exc}")


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "deliverables" / "editor_submission_package_20260504"
OUT = PKG / "06_converted_or_wrapped_svg_assets"

KEY_DIRS = [
    "fig5_causal_intervention_20260504",
    "cam_feature_alignment_20260504",
    "error_mechanism_global_20260504",
    "fig5_intuitive_explainability_20260504",
    "manuscript_table_pack_20260504",
    "v5_feature_learning_final6_20260504",
    "figure_backup_rawprior_20260416",
    "fig5_system_upgrade_20260430",
]


def safe_name(path: Path) -> str:
    rel = path.relative_to(ROOT / "deliverables")
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in path.stem)[:74]
    return f"{stem}__{digest}.svg"


def native_svg_exists(path: Path) -> bool:
    return path.with_suffix(".svg").exists()


def wrap_png_as_svg(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        width, height = img.size
    encoded = base64.b64encode(src.read_bytes()).decode("ascii")
    title = html.escape(src.name)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">\n'
        f'  <title>{title}</title>\n'
        f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n"
    )
    dst.write_text(svg, encoding="utf-8")


def convert_pdf_to_svg(src: Path, dst: Path) -> bool:
    if shutil.which("pdftocairo") is None:
        return False
    try:
        subprocess.run(["pdftocairo", "-svg", str(src), str(dst)], cwd=str(ROOT), check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for dirname in KEY_DIRS:
        folder = ROOT / "deliverables" / dirname
        if not folder.exists():
            continue

        for src in sorted(folder.rglob("*")):
            if not src.is_file() or PKG in src.parents:
                continue
            if src.suffix.lower() not in {".png", ".pdf"}:
                continue

            rel = src.relative_to(ROOT / "deliverables")
            dst = OUT / safe_name(src)
            method = ""
            status = ""
            note = ""

            if native_svg_exists(src):
                method = "native_svg_available"
                status = "not_converted"
                note = "A same-stem native SVG already exists in deliverables and is copied in 02_all_available_svg_figures."
            elif src.suffix.lower() == ".png":
                wrap_png_as_svg(src, dst)
                method = "png_embedded_in_svg"
                status = "created"
                note = "Raster image wrapped in SVG for editor-friendly packaging; not a true vector trace."
            elif src.suffix.lower() == ".pdf":
                if convert_pdf_to_svg(src, dst):
                    method = "pdf_to_svg_pdftocairo"
                    status = "created"
                    note = "PDF converted to SVG with pdftocairo."
                else:
                    method = "pdf_to_svg_pdftocairo"
                    status = "failed"
                    note = "pdftocairo unavailable or conversion failed."

            rows.append(
                {
                    "source_path": str(src),
                    "relative_source": str(rel),
                    "source_type": src.suffix.lower(),
                    "native_svg_same_stem": str(native_svg_exists(src)),
                    "created_file": dst.name if dst.exists() else "",
                    "method": method,
                    "status": status,
                    "note": note,
                }
            )

    with (OUT / "_converted_or_wrapped_svg_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "relative_source",
                "source_type",
                "native_svg_same_stem",
                "created_file",
                "method",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "folder": str(OUT),
        "audited_assets": len(rows),
        "created_svg_wrappers_or_conversions": sum(1 for r in rows if r["status"] == "created"),
        "native_svg_already_available": sum(1 for r in rows if r["status"] == "not_converted"),
        "failed_conversions": sum(1 for r in rows if r["status"] == "failed"),
        "note": "PNG wrapping preserves visual appearance but remains raster content inside an SVG container.",
    }
    (OUT / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
