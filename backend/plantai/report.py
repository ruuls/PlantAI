"""PDF field report."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .analytics import (
    CONFIDENCE_BANDS,
    StandMetrics,
    class_confidence,
    confidence_bands,
)

INK = (24, 28, 34)
MUTED = (120, 130, 140)
RULE = (205, 212, 220)
GREEN = (46, 125, 74)
TONE_RGB = {"green": (58, 150, 90), "amber": (186, 138, 40), "red": (198, 62, 58)}


class ReportError(RuntimeError):
    """Raised when the PDF toolchain is unavailable."""


def build_report(
    df: pd.DataFrame,
    metrics: StandMetrics,
    *,
    dataset_name: str = "",
    plot_id: str = "",
    scope: str = "All detections",
    min_conf: float = 0.0,
    rows: list[dict] | None = None,
) -> bytes:
    """Render a one-page summary of the current selection."""
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ReportError(
            "fpdf2 is not installed. Add it to the backend environment to export PDFs."
        ) from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 16, 18)

    def rule(gap_before: float = 3.0, gap_after: float = 4.0) -> None:
        pdf.ln(gap_before)
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.3)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(gap_after)

    def heading(text: str) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def field(label: str, value: str) -> None:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*MUTED)
        pdf.cell(78, 6.4, label)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6.4, value, new_x="LMARGIN", new_y="NEXT")

    # ---- masthead
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(*GREEN)
    pdf.cell(0, 11, "PlantAI - Field Detection Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"Generated {stamp}"
    if dataset_name:
        subtitle += f"  |  {dataset_name}"
    if plot_id:
        subtitle += f"  |  plot {plot_id}"
    pdf.cell(0, 5.5, subtitle, new_x="LMARGIN", new_y="NEXT")
    rule()

    # ---- status banner
    tone = TONE_RGB.get(metrics.tone, TONE_RGB["amber"])
    pdf.set_fill_color(*tone)
    pdf.rect(18, pdf.get_y(), 2.2, 12, style="F")
    y0 = pdf.get_y()
    pdf.set_xy(23, y0 + 0.5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*tone)
    pdf.cell(0, 5.5, metrics.status, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(23)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(
        0,
        5,
        f"{metrics.plants_per_acre:,.0f} plants/acre  |  {metrics.pct_of_target:.1f}% of "
        f"{metrics.target_plants_acre:,.0f} target  |  {metrics.area_acres:.4f} acres",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_y(y0 + 13)
    rule(1, 4)

    # ---- counts
    heading("Detection summary")
    total = int(len(df))
    avg_conf = float(df["confidence"].mean()) * 100 if total else 0.0
    field("Scope", scope)
    field("Confidence filter", f"{min_conf * 100:.0f}% and above")
    field("Total detections", f"{total:,}")
    field("Plants", f"{metrics.plants:,}")
    field("Weeds", f"{metrics.weeds:,}")
    field("Weed pressure", f"{metrics.weed_pressure_pct:.1f}%")
    field("Average confidence", f"{avg_conf:.1f}%")
    rule()

    # ---- stand assessment
    heading("Stand assessment")
    field("Area basis", "Plot row geometry" if metrics.basis == "rows" else "Mapped area")
    field("Area", f"{metrics.area_acres:.4f} acres ({metrics.area_ha:.4f} ha)")
    field("Plants per acre", f"{metrics.plants_per_acre:,.0f}")
    field("Weeds per acre", f"{metrics.weeds_per_acre:,.0f}")
    if metrics.basis == "rows":
        field("Rows counted", f"{metrics.geometry.rows}")
        field("Row length", f"{metrics.geometry.row_length_ft:g} ft")
        field("Row spacing", f"{metrics.geometry.row_spacing_in:g} in")
        field("Total row length", f"{metrics.total_row_length_ft:,.1f} ft")
        field("Plants per foot", f"{metrics.plants_per_ft:.2f}")
        field("Weeds per foot", f"{metrics.weeds_per_ft:.2f}")
    field("Target population", f"{metrics.target_plants_acre:,.0f} plants/acre")
    field("Alert threshold", f"{metrics.threshold_plants_acre:,.0f} plants/acre")
    field("Percent of target", f"{metrics.pct_of_target:.1f}%")
    rule()

    # ---- confidence
    heading("Confidence distribution")
    bands = confidence_bands(df)
    widest = max([b["count"] for b in bands] + [1])
    for band, (_, _, label) in zip(bands, CONFIDENCE_BANDS):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*MUTED)
        pdf.cell(28, 6, f"{label}%")
        bar_y = pdf.get_y() + 1.6
        width = 96.0 * band["count"] / widest
        if width > 0:
            pdf.set_fill_color(*GREEN)
            pdf.rect(pdf.get_x(), bar_y, max(width, 0.6), 2.8, style="F")
        pdf.set_x(pdf.get_x() + 100)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, f"{band['count']:,}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    heading("Class confidence")
    for entry in class_confidence(df):
        avg = f"{entry['avg_conf']:.1f}%" if entry["avg_conf"] is not None else "-"
        low = f"{entry['min_conf']:.1f}%" if entry["min_conf"] is not None else "-"
        field(entry["class"], f"{entry['count']:,} detections | avg {avg} | min {low}")

    if rows:
        rule()
        heading("Row-level summary")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*MUTED)
        for label, width in (("Row", 20), ("Length ft", 28), ("Plants", 26), ("Weeds", 26), ("Plants/ft", 28)):
            pdf.cell(width, 6, label)
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        for row in rows[:24]:
            pdf.cell(20, 5.6, str(row["row_index"]))
            pdf.cell(28, 5.6, f"{row['row_length_ft']:g}")
            pdf.cell(26, 5.6, f"{row['plants']:,}")
            pdf.cell(26, 5.6, f"{row['weeds']:,}")
            pdf.cell(28, 5.6, f"{row['plants_per_ft']:.2f}")
            pdf.ln(5.6)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.ln(1)
        pdf.multi_cell(
            0,
            4.4,
            "Row values are allocated evenly across the plot. Exact per-row counts "
            "require per-row polygons.",
        )

    return bytes(pdf.output())
