from __future__ import annotations

import csv
import html
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "deliverables" / "editor_submission_package_20260504"
OUT = PKG / "10_original_supporting_documents"

SUPPORTING_FILES = [
    (
        "FEM_contact_simulation_docx",
        Path(r"C:\Users\SWH\Documents\xwechat_files\wxid_ty6107lq8k9q22_6147\temp\RWTemp\2026-05\9fe8b532cdec31240dadffc0e8591450\肺结节接触仿真.docx"),
        "Original FEM/contact simulation document mentioned during interpretability closure.",
    ),
    (
        "FEM_contact_simulation_docx_xwechat_20260503",
        Path(r"C:\Users\SWH\Documents\xwechat_files\wxid_ty6107lq8k9q22_6147\msg\file\2026-05\肺结节接触仿真.docx"),
        "Available WeChat-file copy of the FEM/contact simulation document, dated 2026-05-03.",
    ),
    (
        "FEM_contact_simulation_docx_literature_enhanced",
        ROOT / "deliverables" / "肺结节接触仿真_文献补强版_20260430.docx",
        "Deliverables copy of the FEM/contact simulation document with literature strengthening.",
    ),
    (
        "FEM_contact_simulation_docx_figure_notes",
        ROOT / "deliverables" / "肺结节接触仿真_图表说明修改版_20260430.docx",
        "Deliverables copy of the FEM/contact simulation document with figure-note modifications.",
    ),
    (
        "Clinical_experiment_csv",
        Path(r"C:\Users\SWH\Downloads\临床实验数据.csv"),
        "Original clinical experiment data CSV used for V5 clinical inference checks.",
    ),
    (
        "Main_WPS_exported_pdf",
        Path(r"C:\Users\SWH\WPSDrive\1755855549\WPS企业云盘\四川大学WPS\我的企业文档\应用\输出为PDF\肺结节实时检测系统设计与实现(1)_20260504140751.pdf"),
        "Current WPS-exported assembled figure PDF supplied by the user.",
    ),
]


def safe_ascii_name(label: str, src: Path) -> str:
    suffix = src.suffix.lower()
    return f"{label}{suffix}"


def docx_text(path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    lines: list[str] = []

    for block in root.findall(".//w:body/*", ns):
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = "".join(t.text or "" for t in block.findall(".//w:t", ns)).strip()
            if text:
                lines.append(text)
        elif tag == "tbl":
            table_rows = []
            for tr in block.findall(".//w:tr", ns):
                cells = []
                for tc in tr.findall(".//w:tc", ns):
                    cell_text = " ".join("".join(t.text or "" for t in p.findall(".//w:t", ns)).strip() for p in tc.findall(".//w:p", ns))
                    cells.append(re.sub(r"\s+", " ", cell_text).strip())
                if cells:
                    table_rows.append(cells)
            if table_rows:
                lines.append("")
                lines.append("[Table]")
                for row in table_rows:
                    lines.append(" | ".join(row))
                lines.append("")
    return "\n\n".join(lines)


def csv_preview(path: Path, max_rows: int = 20) -> tuple[str, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    n_rows = max(0, len(rows) - 1)
    n_cols = len(rows[0]) if rows else 0
    preview = rows[: max_rows + 1]
    rendered = []
    for row in preview:
        rendered.append("| " + " | ".join(html.escape(str(cell)) for cell in row) + " |")
    return "\n".join(rendered), n_rows, n_cols


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, src, note in SUPPORTING_FILES:
        status = "missing"
        copied = ""
        derived = ""
        extra: dict[str, object] = {}
        if src.exists():
            dst = OUT / safe_ascii_name(label, src)
            shutil.copy2(src, dst)
            status = "copied"
            copied = dst.name

            if src.suffix.lower() == ".docx":
                text = docx_text(src)
                md = OUT / f"{label}_extracted_text.md"
                md.write_text(
                    "# Extracted Text\n\n"
                    f"Source: `{src}`\n\n"
                    "This is a plain-text extraction for manuscript planning; inspect the original DOCX for formatting and figures.\n\n"
                    + text
                    + "\n",
                    encoding="utf-8",
                )
                derived = md.name
                extra["extracted_characters"] = len(text)
            elif src.suffix.lower() == ".csv":
                preview, n_rows, n_cols = csv_preview(src)
                md = OUT / f"{label}_preview.md"
                md.write_text(
                    "# CSV Preview\n\n"
                    f"Source: `{src}`\n\n"
                    f"Rows excluding header: {n_rows}\n\n"
                    f"Columns: {n_cols}\n\n"
                    "Preview:\n\n"
                    + preview
                    + "\n",
                    encoding="utf-8",
                )
                derived = md.name
                extra["rows_excluding_header"] = n_rows
                extra["columns"] = n_cols

        rows.append(
            {
                "label": label,
                "source_path": str(src),
                "status": status,
                "copied_file": copied,
                "derived_file": derived,
                "note": note,
                "extra": json.dumps(extra, ensure_ascii=False),
            }
        )

    with (OUT / "_supporting_documents_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "source_path", "status", "copied_file", "derived_file", "note", "extra"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "folder": str(OUT),
        "items": rows,
    }
    (OUT / "supporting_documents_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
