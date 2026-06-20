"""
AgroSense AI — Export Module
Handles GeoTIFF, CSV, and PDF report generation for all analysis outputs.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — gracefully degrade if unavailable
# ---------------------------------------------------------------------------

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    logger.warning("rasterio not installed — GeoTIFF export disabled")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
        PageBreak,
    )

    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    logger.warning("reportlab not installed — PDF export disabled")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEOTIFF_COMPRESS, GEOTIFF_CRS


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------


def export_geotiff(
    data: np.ndarray,
    bounds: List[List[float]],
    filename: str,
    dtype: str = "float32",
    nodata_value: float = -9999.0,
) -> bytes:
    """Export a 2-D numpy array as a GeoTIFF byte stream.

    The output is a single-band GeoTIFF georeferenced in WGS-84 (EPSG:4326)
    using an affine transform derived from the supplied bounding box.  LZW
    (or the project-configured) compression is applied to keep file sizes
    manageable.

    Args:
        data: 2-D array ``(H, W)`` to export.  NaN values are written as
            *nodata_value*.
        bounds: ``[[south, west], [north, east]]`` bounding box in decimal
            degrees.
        filename: Logical output filename — written into the GeoTIFF tags for
            traceability but does not affect file I/O.
        dtype: NumPy dtype string for the raster band, e.g. ``"float32"``,
            ``"int16"``.
        nodata_value: Sentinel value substituted for NaN / masked pixels.

    Returns:
        Raw GeoTIFF bytes ready to be written to disk or sent as a download.

    Raises:
        RuntimeError: If rasterio is not installed.
        ValueError: If *data* is not a 2-D array.
    """
    if not RASTERIO_OK:
        raise RuntimeError("rasterio not installed — cannot export GeoTIFF")

    if data.ndim != 2:
        raise ValueError(
            f"export_geotiff expects a 2-D array, got shape {data.shape}"
        )

    south, west = bounds[0]
    north, east = bounds[1]
    H, W = data.shape

    transform = from_bounds(west, south, east, north, W, H)

    # Replace NaN with nodata sentinel
    out_data = data.astype(np.float32).copy()
    out_data[np.isnan(out_data)] = nodata_value

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype=dtype,
        crs=CRS.from_string(GEOTIFF_CRS),
        transform=transform,
        compress=GEOTIFF_COMPRESS,
        nodata=nodata_value,
    ) as dst:
        dst.write(out_data, 1)
        dst.update_tags(
            filename=filename,
            created=datetime.now().isoformat(),
            generator="AgroSense AI",
        )

    buf.seek(0)
    logger.debug("GeoTIFF exported: %s (%d x %d)", filename, W, H)
    return buf.read()


def export_all_geotiffs_zip(
    results: Dict[str, Any],
    bounds: List[List[float]],
    prefix: str = "agrosense",
) -> bytes:
    """Package all spatial output layers from an advisory pipeline run as a ZIP.

    Iterates over a predefined set of layer keys expected in *results*, skips
    layers that are ``None`` or non-array, and bundles the remainder into a
    single ZIP archive.  A CSV advisory summary is appended when available.

    Args:
        results: Dict returned by ``advisory_engine.run_advisory_pipeline()``.
        bounds: ``[[south, west], [north, east]]``.
        prefix: Filename prefix applied to every file inside the archive.

    Returns:
        ZIP archive as bytes.  Returns a valid (empty) ZIP if no exportable
        layers are found.
    """
    layers_to_export: Dict[str, Optional[np.ndarray]] = {
        "crop_type": results.get("crop_map"),
        "stress_class": results.get("stress_stress_class"),
        "irrigation_advisory": results.get("advisory_code"),
        "water_deficit_mm": results.get("irr_need_mm"),
        "ndvi_current": results.get("ndvi_current"),
        "vhi": results.get("stress_vhi"),
        "et0_daily": results.get("et0_daily"),
        "etc_period": results.get("etc_period"),
        "taw_mm": results.get("taw_map"),
        "dr_current_mm": results.get("dr_current"),
    }

    zip_buf = io.BytesIO()
    exported_count = 0

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in layers_to_export.items():
            if arr is None or not isinstance(arr, np.ndarray):
                logger.debug("Skipping layer '%s' — not available", name)
                continue
            try:
                tif_filename = f"{prefix}_{name}.tif"
                tif_bytes = export_geotiff(arr, bounds, tif_filename)
                zf.writestr(tif_filename, tif_bytes)
                exported_count += 1
                logger.debug("Packed %s into ZIP", tif_filename)
            except Exception as exc:
                logger.error("Failed to export layer '%s': %s", name, exc)

        # Append CSV advisory summary when present
        crop_summary = results.get("crop_summary")
        if crop_summary:
            try:
                csv_bytes = export_advisory_csv(crop_summary)
                zf.writestr(f"{prefix}_advisory_summary.csv", csv_bytes)
                logger.debug("Advisory CSV appended to ZIP")
            except Exception as exc:
                logger.error("Failed to append advisory CSV to ZIP: %s", exc)

    logger.info(
        "GeoTIFF ZIP created — %d layers exported, prefix='%s'",
        exported_count,
        prefix,
    )
    zip_buf.seek(0)
    return zip_buf.read()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_advisory_csv(
    crop_summary: List[Dict[str, Any]],
    block_summary_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """Export advisory summary data as UTF-8 CSV bytes.

    Concatenates a crop-wise advisory list and an optional block-level
    DataFrame into a single flat CSV.

    Args:
        crop_summary: List of per-crop advisory dicts containing at minimum
            ``crop``, ``area_ha``, ``mean_irr_need_mm``, ``mean_stress_score``,
            and ``advisory`` keys.
        block_summary_df: Optional block-level ``pd.DataFrame`` to append
            after the crop-wise rows.

    Returns:
        UTF-8 encoded CSV bytes.  Returns ``b""`` if both inputs are empty.
    """
    frames: List[pd.DataFrame] = []

    if crop_summary:
        frames.append(pd.DataFrame(crop_summary))

    if block_summary_df is not None and not block_summary_df.empty:
        frames.append(block_summary_df)

    if not frames:
        logger.warning("export_advisory_csv called with no data — returning empty CSV")
        return b""

    combined = pd.concat(frames, ignore_index=True)
    csv_bytes = combined.to_csv(index=False).encode("utf-8")
    logger.debug("Advisory CSV generated: %d rows", len(combined))
    return csv_bytes


def export_pixel_timeseries_csv(
    timeseries: Dict[str, List[float]],
    dates: List[str],
) -> bytes:
    """Export a multi-variable pixel time series as CSV bytes.

    Args:
        timeseries: Dict mapping variable names to lists of values aligned
            with *dates*, e.g. ``{"ndvi": [...], "et0": [...]}``.
        dates: ISO-8601 date strings matching the length of each value list.

    Returns:
        UTF-8 CSV bytes with ``date`` as the first column.

    Raises:
        ValueError: If any value list has a different length from *dates*.
    """
    n = len(dates)
    for key, vals in timeseries.items():
        if len(vals) != n:
            raise ValueError(
                f"Length mismatch: dates has {n} entries but '{key}' has {len(vals)}"
            )

    df = pd.DataFrame({"date": dates, **timeseries})
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------


def generate_pdf_report(
    results: Dict[str, Any],
    area_name: str = "Study Area",
    season: str = "Kharif",
    year: int = 2024,
    generated_date: Optional[str] = None,
) -> bytes:
    """Generate a printable A4 PDF irrigation advisory report.

    Builds a multi-section report containing:
    - Title header with area, season, and generation date
    - Executive summary metrics table
    - Crop-wise advisory table (when ``results["crop_summary"]`` is present)
    - Weather inputs summary (when ``results["weather_summary"]`` is present)
    - Footer with data source attribution

    Args:
        results: Advisory pipeline results dict.  Expected keys (all optional):
            ``total_area_ha``, ``total_irr_volume_ML``, ``stress_summary``,
            ``crop_summary``, ``weather_summary``.
        area_name: Human-readable name of the study area.
        season: Crop season name (e.g. ``"Kharif"``, ``"Rabi"``).
        year: Four-digit analysis year.
        generated_date: Report generation timestamp string.  Defaults to
            ``datetime.now()`` formatted as ``YYYY-MM-DD HH:MM``.

    Returns:
        PDF file content as bytes, suitable for direct download or disk write.

    Raises:
        RuntimeError: If reportlab is not installed.
    """
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed — cannot generate PDF report")

    if generated_date is None:
        generated_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        fontName="Helvetica-Bold",
        textColor=rl_colors.HexColor("#2E7D32"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica",
        textColor=rl_colors.HexColor("#555555"),
        spaceAfter=8,
    )
    header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=rl_colors.HexColor("#1565C0"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        fontName="Helvetica",
        spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica",
        textColor=rl_colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )

    # ------------------------------------------------------------------
    # Story assembly
    # ------------------------------------------------------------------
    story = []

    # Title block
    story.append(
        Paragraph("AgroSense AI — Irrigation Advisory Report", title_style)
    )
    story.append(
        Paragraph(
            f"Study Area: <b>{area_name}</b> | Season: <b>{season} {year}</b> | "
            f"Generated: {generated_date}",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=rl_colors.HexColor("#2E7D32"),
            spaceAfter=10,
        )
    )

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    story.append(Paragraph("Executive Summary", header_style))

    stress_summary = results.get("stress_summary", {})
    summary_rows = [
        ["Metric", "Value"],
        [
            "Total Area Analyzed",
            f"{results.get('total_area_ha', 0.0):.1f} ha",
        ],
        [
            "Total Irrigation Need",
            f"{results.get('total_irr_volume_ML', 0.0):.3f} ML",
        ],
        [
            "Mean Stress Score",
            f"{stress_summary.get('mean_stress', 0.0):.1f} / 100",
        ],
        [
            "Severely Stressed Area",
            f"{stress_summary.get('pct_severe_stress', 0.0):.1f} %",
        ],
        [
            "Moderately Stressed Area",
            f"{stress_summary.get('pct_moderate_stress', 0.0):.1f} %",
        ],
    ]

    summary_table = Table(summary_rows, colWidths=[9 * cm, 8 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2E7D32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 7),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [rl_colors.white, rl_colors.HexColor("#F1F8E9")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Crop-wise advisory table
    # ------------------------------------------------------------------
    crop_summary: List[Dict[str, Any]] = results.get("crop_summary", [])
    if crop_summary:
        story.append(Paragraph("Crop-Wise Irrigation Advisory", header_style))

        crop_header = [
            "Crop",
            "Area (ha)",
            "Deficit (mm)",
            "Stress /100",
            "Advisory",
        ]
        crop_rows = [crop_header]
        for row in crop_summary:
            crop_rows.append(
                [
                    str(row.get("crop", "")).capitalize(),
                    f"{row.get('area_ha', 0):.1f}",
                    f"{row.get('mean_irr_need_mm', 0):.1f}",
                    f"{row.get('mean_stress_score', 0):.0f}",
                    str(row.get("advisory", "")),
                ]
            )

        crop_table = Table(
            crop_rows,
            colWidths=[3.2 * cm, 2.5 * cm, 2.8 * cm, 2.5 * cm, 6.0 * cm],
        )
        crop_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        rl_colors.HexColor("#1565C0"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [rl_colors.white, rl_colors.HexColor("#E3F2FD")],
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        rl_colors.HexColor("#CCCCCC"),
                    ),
                    ("ALIGN", (1, 1), (3, -1), "CENTER"),
                    ("WORDWRAP", (4, 1), (4, -1), True),
                ]
            )
        )
        story.append(crop_table)
        story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Weather inputs summary (optional)
    # ------------------------------------------------------------------
    weather_summary: Dict[str, Any] = results.get("weather_summary", {})
    if weather_summary:
        story.append(Paragraph("Weather Inputs Summary", header_style))

        wx_rows = [["Parameter", "Value"]]
        for key, val in weather_summary.items():
            label = key.replace("_", " ").title()
            wx_rows.append([label, str(val)])

        wx_table = Table(wx_rows, colWidths=[9 * cm, 8 * cm])
        wx_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        rl_colors.HexColor("#546E7A"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [rl_colors.white, rl_colors.HexColor("#ECEFF1")],
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        rl_colors.HexColor("#CCCCCC"),
                    ),
                ]
            )
        )
        story.append(wx_table)
        story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Methodology note
    # ------------------------------------------------------------------
    story.append(Paragraph("Methodology", header_style))
    story.append(
        Paragraph(
            "Irrigation scheduling follows the FAO-56 dual crop coefficient approach. "
            "Reference evapotranspiration (ET\u2080) is computed using the Penman-Monteith "
            "equation from ERA5 reanalysis fields. Crop water uptake is parameterised "
            "by NDVI-scaled Kc values. Soil water balance uses per-pixel total available "
            "water (TAW) derived from ISRIC SoilGrids texture data. Stress classification "
            "is based on the Vegetation Health Index (VHI) combining NDVI and LST "
            "anomalies from Sentinel-2 and Landsat-8/9 composites.",
            body_style,
        )
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=rl_colors.HexColor("#AAAAAA"),
            spaceBefore=4,
            spaceAfter=4,
        )
    )
    story.append(
        Paragraph(
            f"AgroSense AI | Data: Sentinel-1/2, ERA5, CHIRPS, ISRIC SoilGrids | "
            f"Generated: {generated_date}",
            footer_style,
        )
    )

    # ------------------------------------------------------------------
    # Build PDF
    # ------------------------------------------------------------------
    doc.build(story)
    buf.seek(0)
    pdf_bytes = buf.read()
    logger.info(
        "PDF report generated for '%s' — %d bytes", area_name, len(pdf_bytes)
    )
    return pdf_bytes


# ---------------------------------------------------------------------------
# Utility: save bytes to disk
# ---------------------------------------------------------------------------


def save_bytes_to_file(data: bytes, output_path: Path) -> None:
    """Write raw bytes to a file, creating parent directories as needed.

    Args:
        data: Raw bytes to write (e.g. GeoTIFF, CSV, PDF, ZIP).
        output_path: Destination :class:`pathlib.Path`.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    logger.info("Saved %d bytes -> %s", len(data), output_path)
