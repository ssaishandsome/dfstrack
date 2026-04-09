#!/usr/bin/env python3
"""Export tracker txt results to an Excel workbook.

This utility is intentionally zero-dependency so it can run in the current
research environment without installing extra packages.

Expected result layout:

results_root/
    dataset_a/
        seq_name.txt
        seq_name_iou.txt
        seq_name_time.txt
    dataset_b/
        ...

Each dataset sheet is sorted by ``mean_iou`` ascending, which makes the worst
performing sequences float to the top for quick inspection.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


TXT_KIND_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("_iou.txt", "iou"),
    ("_time.txt", "time"),
    (".txt", "bbox"),
)

INVALID_SHEET_CHARS = set('[]:*?/\\')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tracker txt result files and export an Excel workbook sorted "
            "by per-sequence mean IoU."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Tracker result root. It can be a tracker directory containing "
            "dataset subfolders, or a single dataset folder."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path. Defaults to <results-dir>/txt_result_summary.xlsx.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many worst sequences to surface in the summary sheet.",
    )
    return parser.parse_args()


def split_txt_name(file_name: str) -> Tuple[str, str]:
    for suffix, kind in TXT_KIND_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)], kind
    raise ValueError(f"Unsupported txt file name: {file_name}")


def infer_dataset_dirs(results_dir: Path) -> "OrderedDict[str, Path]":
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    dataset_dirs = OrderedDict(
        (path.name, path)
        for path in sorted(results_dir.iterdir())
        if path.is_dir() and any(path.glob("*.txt"))
    )
    if dataset_dirs:
        return dataset_dirs

    if any(results_dir.glob("*.txt")):
        return OrderedDict([(results_dir.name, results_dir)])

    raise ValueError(
        "No txt result files found. Expect either dataset subfolders or txt files "
        f"directly under: {results_dir}"
    )


def read_scalar_series(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [token for token in re.split(r"[\s,\t]+", line) if token]
            for token in tokens:
                values.append(float(token))
    return values


def count_non_empty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def ratio_below(values: Sequence[float], threshold: float) -> Optional[float]:
    if not values:
        return None
    return sum(value < threshold for value in values) / len(values)


def filter_non_negative(values: Sequence[float]) -> List[float]:
    return [value for value in values if value >= 0.0]


def sanitize_sheet_name(name: str, used_names: set) -> str:
    clean = "".join("_" if char in INVALID_SHEET_CHARS else char for char in name)
    clean = clean.strip() or "Sheet"
    clean = clean[:31]
    if clean not in used_names:
        used_names.add(clean)
        return clean

    base = clean[:28] if len(clean) > 28 else clean
    index = 1
    while True:
        candidate = f"{base}_{index}"
        candidate = candidate[:31]
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def sort_key_for_row(row: Dict[str, object]) -> Tuple[bool, float, str]:
    mean_iou = row.get("mean_iou")
    if mean_iou is None:
        return (True, math.inf, str(row["sequence"]))
    return (False, float(mean_iou), str(row["sequence"]))


def build_sequence_rows(results_dir: Path, dataset_name: str, dataset_dir: Path) -> List[Dict[str, object]]:
    records: "OrderedDict[str, Dict[str, object]]" = OrderedDict()

    for txt_path in sorted(dataset_dir.glob("*.txt")):
        sequence_name, kind = split_txt_name(txt_path.name)
        record = records.setdefault(
            sequence_name,
            {
                "dataset": dataset_name,
                "sequence": sequence_name,
                "bbox_file": "",
                "iou_file": "",
                "time_file": "",
                "bbox_frames": None,
                "iou_frames": None,
                "time_frames": None,
            },
        )
        record[f"{kind}_file"] = txt_path.relative_to(results_dir).as_posix()

        if kind == "bbox":
            record["bbox_frames"] = count_non_empty_lines(txt_path)
        elif kind == "iou":
            iou_values = read_scalar_series(txt_path)
            valid_iou_values = filter_non_negative(iou_values)
            record["iou_frames"] = len(iou_values)
            record["mean_iou"] = round_or_none(mean_or_none(iou_values))
            record["mean_iou_valid_only"] = round_or_none(mean_or_none(valid_iou_values))
            record["median_iou"] = round_or_none(median_or_none(iou_values))
            record["min_iou"] = round_or_none(min(iou_values) if iou_values else None)
            record["max_iou"] = round_or_none(max(iou_values) if iou_values else None)
            record["valid_iou_frames"] = len(valid_iou_values)
            record["invalid_iou_ratio"] = round_or_none(ratio_below(iou_values, 0.0))
            record["iou_lt_0_1_ratio"] = round_or_none(ratio_below(iou_values, 0.1))
            record["iou_lt_0_3_ratio"] = round_or_none(ratio_below(iou_values, 0.3))
            record["iou_lt_0_5_ratio"] = round_or_none(ratio_below(iou_values, 0.5))
        elif kind == "time":
            time_values = read_scalar_series(txt_path)
            avg_time_s = mean_or_none(time_values)
            record["time_frames"] = len(time_values)
            record["avg_time_s"] = round_or_none(avg_time_s)
            record["fps"] = round_or_none((1.0 / avg_time_s) if avg_time_s and avg_time_s > 0 else None)
            record["max_time_s"] = round_or_none(max(time_values) if time_values else None)

    rows = list(records.values())
    rows.sort(key=sort_key_for_row)
    for index, row in enumerate(rows, start=1):
        row["bad_rank"] = index
    return rows


def worst_items_text(rows: Sequence[Dict[str, object]], top_k: int) -> str:
    snippets: List[str] = []
    for row in rows:
        mean_iou = row.get("mean_iou")
        if mean_iou is None:
            continue
        snippets.append(f"{row['sequence']} ({float(mean_iou):.3f})")
        if len(snippets) >= top_k:
            break
    return "; ".join(snippets)


def build_summary_rows(dataset_to_rows: Dict[str, List[Dict[str, object]]], top_k: int) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for dataset_name, rows in dataset_to_rows.items():
        iou_values = [float(row["mean_iou"]) for row in rows if row.get("mean_iou") is not None]
        iou_valid_only_values = [
            float(row["mean_iou_valid_only"])
            for row in rows
            if row.get("mean_iou_valid_only") is not None
        ]
        fps_values = [float(row["fps"]) for row in rows if row.get("fps") is not None]
        worst_row = next((row for row in rows if row.get("mean_iou") is not None), None)

        summary_rows.append(
            {
                "dataset": dataset_name,
                "num_sequences": len(rows),
                "num_with_iou": len(iou_values),
                "num_with_time": len(fps_values),
                "dataset_mean_iou": round_or_none(mean_or_none(iou_values)),
                "dataset_mean_iou_valid_only": round_or_none(mean_or_none(iou_valid_only_values)),
                "dataset_median_iou": round_or_none(median_or_none(iou_values)),
                "dataset_mean_fps": round_or_none(mean_or_none(fps_values)),
                "worst_sequence": "" if worst_row is None else worst_row["sequence"],
                "worst_mean_iou": None if worst_row is None else worst_row["mean_iou"],
                "worst_examples": worst_items_text(rows, top_k),
            }
        )
    return summary_rows


def build_overall_rows(dataset_to_rows: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    merged_rows: List[Dict[str, object]] = []
    for rows in dataset_to_rows.values():
        for row in rows:
            merged_rows.append(dict(row))
    merged_rows.sort(key=sort_key_for_row)
    for index, row in enumerate(merged_rows, start=1):
        row["overall_bad_rank"] = index
    return merged_rows


def excel_col_name(index: int) -> str:
    name = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name.append(chr(ord("A") + remainder))
    return "".join(reversed(name))


def infer_column_widths(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[int]:
    widths: List[int] = []
    for col_idx, header in enumerate(headers):
        max_length = len(str(header))
        for row in rows:
            if col_idx >= len(row):
                continue
            value = row[col_idx]
            if value is None:
                continue
            max_length = max(max_length, len(str(value)))
        widths.append(min(max(max_length + 2, 10), 60))
    return widths


def format_cell(value: object, is_header: bool = False) -> str:
    style_attr = ' s="1"' if is_header else ""
    if value is None:
        return f'<c{style_attr}/>'
    if isinstance(value, bool):
        return f'<c t="b"{style_attr}><v>{int(value)}</v></c>'
    if isinstance(value, int):
        return f'<c{style_attr}><v>{value}</v></c>'
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f'<c{style_attr}/>'
        return f'<c{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def make_sheet_xml(headers: Sequence[str], rows: Sequence[Sequence[object]], widths: Sequence[int]) -> str:
    last_row = len(rows) + 1
    last_col_name = excel_col_name(len(headers))
    dimension = f"A1:{last_col_name}{last_row}"

    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )

    xml_rows: List[str] = []
    header_cells = "".join(
        f'<c r="{excel_col_name(col_idx)}1" t="inlineStr" s="1"><is><t>{escape(str(header))}</t></is></c>'
        for col_idx, header in enumerate(headers, start=1)
    )
    xml_rows.append(f'<row r="1">{header_cells}</row>')

    for row_idx, row_values in enumerate(rows, start=2):
        row_cells: List[str] = []
        for col_idx, value in enumerate(row_values, start=1):
            cell_ref = f'{excel_col_name(col_idx)}{row_idx}'
            cell_xml = format_cell(value)
            if cell_xml == "<c/>":
                row_cells.append(f'<c r="{cell_ref}"/>')
            else:
                row_cells.append(cell_xml.replace("<c", f'<c r="{cell_ref}"', 1))
        xml_rows.append(f'<row r="{row_idx}">{"".join(row_cells)}</row>')

    auto_filter = f"A1:{last_col_name}{last_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{cols_xml}</cols>"
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'<autoFilter ref="{auto_filter}"/>'
        "</worksheet>"
    )


def workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets_xml = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<bookViews><workbookView activeTab="0"/></bookViews>'
        f"<sheets>{sheets_xml}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(num_sheets: int) -> str:
    relationships = "".join(
        (
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        for idx in range(1, num_sheets + 1)
    )
    style_rid = num_sheets + 1
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}"
        f'<Relationship Id="rId{style_rid}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )


def content_types_xml(num_sheets: int) -> str:
    overrides = "".join(
        (
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        for idx in range(1, num_sheets + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{overrides}"
        "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFDCE6F1"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def app_xml(sheet_names: Sequence[str]) -> str:
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>DFSTrack txt summary exporter</Application>'
        f'<HeadingPairs><vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>'
        f'<vt:variant><vt:i4>{len(sheet_names)}</vt:i4></vt:variant>'
        "</vt:vector></HeadingPairs>"
        f'<TitlesOfParts><vt:vector size="{len(sheet_names)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>'
        "</Properties>"
    )


def core_xml() -> str:
    created = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Codex</dc:creator>'
        '<cp:lastModifiedBy>Codex</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def write_xlsx(output_path: Path, sheets: Sequence[Tuple[str, Sequence[str], Sequence[Sequence[object]]]]) -> None:
    used_names: set = set()
    safe_sheet_names = [sanitize_sheet_name(name, used_names) for name, _, _ in sheets]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        workbook.writestr("_rels/.rels", root_rels_xml())
        workbook.writestr("xl/workbook.xml", workbook_xml(safe_sheet_names))
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        workbook.writestr("xl/styles.xml", styles_xml())
        workbook.writestr("docProps/app.xml", app_xml(safe_sheet_names))
        workbook.writestr("docProps/core.xml", core_xml())

        for idx, (_, headers, rows) in enumerate(sheets, start=1):
            widths = infer_column_widths(headers, rows)
            sheet_xml = make_sheet_xml(headers, rows, widths)
            workbook.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)


def rows_from_dicts(rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> List[List[object]]:
    return [[row.get(header) for header in headers] for row in rows]


def build_sheet_payloads(dataset_to_rows: Dict[str, List[Dict[str, object]]], top_k: int) -> List[Tuple[str, List[str], List[List[object]]]]:
    summary_rows = build_summary_rows(dataset_to_rows, top_k=top_k)
    overall_rows = build_overall_rows(dataset_to_rows)

    summary_headers = [
        "dataset",
        "num_sequences",
        "num_with_iou",
        "num_with_time",
        "dataset_mean_iou",
        "dataset_mean_iou_valid_only",
        "dataset_median_iou",
        "dataset_mean_fps",
        "worst_sequence",
        "worst_mean_iou",
        "worst_examples",
    ]
    sequence_headers = [
        "bad_rank",
        "dataset",
        "sequence",
        "mean_iou",
        "mean_iou_valid_only",
        "median_iou",
        "min_iou",
        "max_iou",
        "valid_iou_frames",
        "invalid_iou_ratio",
        "iou_lt_0_1_ratio",
        "iou_lt_0_3_ratio",
        "iou_lt_0_5_ratio",
        "avg_time_s",
        "fps",
        "max_time_s",
        "bbox_frames",
        "iou_frames",
        "time_frames",
        "bbox_file",
        "iou_file",
        "time_file",
    ]
    overall_headers = ["overall_bad_rank"] + sequence_headers

    sheets: List[Tuple[str, List[str], List[List[object]]]] = [
        ("summary", summary_headers, rows_from_dicts(summary_rows, summary_headers)),
        ("worst_overall", overall_headers, rows_from_dicts(overall_rows, overall_headers)),
    ]
    for dataset_name, rows in dataset_to_rows.items():
        sheets.append((dataset_name, sequence_headers, rows_from_dicts(rows, sequence_headers)))
    return sheets


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_path = args.output.resolve() if args.output else (results_dir / "txt_result_summary.xlsx").resolve()

    dataset_dirs = infer_dataset_dirs(results_dir)
    dataset_to_rows = OrderedDict(
        (dataset_name, build_sequence_rows(results_dir, dataset_name, dataset_dir))
        for dataset_name, dataset_dir in dataset_dirs.items()
    )

    sheets = build_sheet_payloads(dataset_to_rows, top_k=args.top_k)
    write_xlsx(output_path, sheets)

    num_sequences = sum(len(rows) for rows in dataset_to_rows.values())
    print(f"Saved workbook: {output_path}")
    print(f"Datasets: {len(dataset_to_rows)}")
    print(f"Sequences summarized: {num_sequences}")


if __name__ == "__main__":
    main()
