"""
report_generator.py
Generates a professional PDF clinical report for Diabetic Retinopathy diagnosis.
Uses ReportLab for PDF generation.
"""

import io
import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether


# ─── Color palette ────────────────────────────────────────────────────────────
BRAND_BLUE = colors.HexColor("#1e40af")
BRAND_LIGHT = colors.HexColor("#eff6ff")
TEXT_DARK = colors.HexColor("#1e293b")
TEXT_MUTED = colors.HexColor("#64748b")
DIVIDER = colors.HexColor("#cbd5e1")

URGENCY_COLORS = {
    "Low": colors.HexColor("#10b981"),
    "Low-Moderate": colors.HexColor("#eab308"),
    "Moderate": colors.HexColor("#f97316"),
    "High": colors.HexColor("#ef4444"),
    "Critical": colors.HexColor("#881337"),
}


def _urgency_color(urgency: str) -> colors.Color:
    return URGENCY_COLORS.get(urgency, colors.black)


def _make_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {}

    styles["Title"] = ParagraphStyle(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#bfdbfe"),
        alignment=TA_CENTER,
    )
    styles["SectionHeader"] = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=BRAND_BLUE,
        spaceBefore=12,
        spaceAfter=6,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_DARK,
        leading=13,
        spaceAfter=3,
    )
    styles["BulletItem"] = ParagraphStyle(
        "BulletItem",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=TEXT_DARK,
        leading=13,
        leftIndent=12,
        bulletIndent=0,
    )
    styles["Disclaimer"] = ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        leading=11,
        spaceBefore=8,
    )
    styles["Meta"] = ParagraphStyle(
        "Meta",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_DARK,
        leading=14,
    )
    styles["MetaBold"] = ParagraphStyle(
        "MetaBold",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=TEXT_DARK,
        leading=14,
    )
    return styles


class _HeaderFooterTemplate(BaseDocTemplate):
    """Custom page template with coloured header banner and footer."""

    def __init__(self, buffer, report_date: str, patient_id: str, **kwargs):
        self.report_date = report_date
        self.patient_id = patient_id
        super().__init__(buffer, **kwargs)
        frame = Frame(15 * mm, 20 * mm, 180 * mm, 245 * mm, id="main")
        template = PageTemplate(id="main_template", frames=[frame], onPage=self._draw_page)
        self.addPageTemplates([template])

    def _draw_page(self, canvas, doc):
        canvas.saveState()
        # ── Header banner ──────────────────────────────────────────────────
        canvas.setFillColor(BRAND_BLUE)
        canvas.rect(0, A4[1] - 30 * mm, A4[0], 30 * mm, fill=True, stroke=False)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawCentredString(A4[0] / 2, A4[1] - 14 * mm, "Diabetic Retinopathy Diagnostic Report")

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#bfdbfe"))
        canvas.drawCentredString(
            A4[0] / 2,
            A4[1] - 22 * mm,
            "AI-Assisted Clinical Decision Support · EfficientNetB0 Deep Learning Model",
        )

        # ── Footer ─────────────────────────────────────────────────────────
        canvas.setFillColor(DIVIDER)
        canvas.rect(15 * mm, 12 * mm, 180 * mm, 0.3 * mm, fill=True, stroke=False)

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(15 * mm, 8 * mm, f"Patient ID: {self.patient_id}")
        canvas.drawCentredString(A4[0] / 2, 8 * mm, f"Report Date: {self.report_date}")
        canvas.drawRightString(
            A4[0] - 15 * mm,
            8 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()


def generate_pdf_report(
    patient_info: dict,
    prediction_data: dict,
    image_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Generate a professional PDF clinical report.

    Args:
        patient_info: dict with keys: name, patient_id, scan_date
        prediction_data: full /predict response payload including 'recommendations'
        image_bytes: raw bytes of the uploaded retinal image (optional)

    Returns:
        PDF as raw bytes.
    """
    buffer = io.BytesIO()
    styles = _make_styles()

    report_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    patient_id = patient_info.get("patient_id", "N/A")

    doc = _HeaderFooterTemplate(
        buffer,
        report_date=report_date,
        patient_id=patient_id,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=35 * mm,
        bottomMargin=22 * mm,
    )

    story = []

    # ── 1. Patient Meta table ────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=6))

    scan_date = patient_info.get("scan_date", datetime.datetime.now().strftime("%Y-%m-%d"))
    meta_data = [
        ["Patient Name", patient_info.get("name", "Anonymous"), "Patient ID", patient_id],
        ["Scan Date", scan_date, "Report Generated", report_date],
    ]
    meta_table = Table(meta_data, colWidths=[40 * mm, 65 * mm, 40 * mm, 45 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_DARK),
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, DIVIDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8 * mm))

    # ── 2. Retinal image (if provided) ──────────────────────────────────────
    if image_bytes:
        try:
            img_io = io.BytesIO(image_bytes)
            img = Image(img_io, width=55 * mm, height=55 * mm)
            img.hAlign = "LEFT"

            # Diagnosis summary next to image
            recommendations = prediction_data.get("recommendations", {})
            urgency = recommendations.get("urgency", "N/A")
            u_color = _urgency_color(urgency)

            diag_summary = [
                ["", ""],  # spacer row
                [
                    Paragraph("<b>Diagnosis</b>", styles["MetaBold"]),
                    Paragraph(prediction_data.get("prediction", "N/A"), styles["Meta"]),
                ],
                [
                    Paragraph("<b>Confidence</b>", styles["MetaBold"]),
                    Paragraph(f"{prediction_data.get('confidence_percentage', 'N/A')}%", styles["Meta"]),
                ],
                [
                    Paragraph("<b>Class ID</b>", styles["MetaBold"]),
                    Paragraph(str(prediction_data.get("class_id", "N/A")), styles["Meta"]),
                ],
                [
                    Paragraph("<b>ICD-10 Code</b>", styles["MetaBold"]),
                    Paragraph(recommendations.get("icd10_code", "N/A"), styles["Meta"]),
                ],
                [
                    Paragraph("<b>Urgency</b>", styles["MetaBold"]),
                    Paragraph(
                        f'<font color="{u_color.hexval() if hasattr(u_color, "hexval") else "#000000"}"><b>{urgency}</b></font>',
                        styles["Meta"],
                    ),
                ],
                [
                    Paragraph("<b>Follow-up</b>", styles["MetaBold"]),
                    Paragraph(recommendations.get("follow_up_interval", "N/A"), styles["Meta"]),
                ],
            ]
            diag_table = Table(
                [[img, Table(diag_summary, colWidths=[38 * mm, 82 * mm])]],
                colWidths=[60 * mm, 120 * mm],
            )
            diag_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(Paragraph("Diagnostic Summary", styles["SectionHeader"]))
            story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=6))
            story.append(diag_table)
        except Exception:
            # Fallback to no-image layout
            image_bytes = None

    if not image_bytes:
        story.append(Paragraph("Diagnostic Summary", styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=6))
        recommendations = prediction_data.get("recommendations", {})
        urgency = recommendations.get("urgency", "N/A")
        diag_rows = [
            ["Diagnosis", prediction_data.get("prediction", "N/A")],
            ["Confidence", f"{prediction_data.get('confidence_percentage', 'N/A')}%"],
            ["Class ID", str(prediction_data.get("class_id", "N/A"))],
            ["ICD-10 Code", recommendations.get("icd10_code", "N/A")],
            ["Urgency", urgency],
            ["Recommended Follow-up", recommendations.get("follow_up_interval", "N/A")],
        ]
        diag_table = Table(diag_rows, colWidths=[60 * mm, 120 * mm])
        diag_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_DARK),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, DIVIDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(diag_table)

    story.append(Spacer(1, 8 * mm))

    # ── 3. Medical Action Plan ───────────────────────────────────────────────
    recommendations = prediction_data.get("recommendations", {})
    story.append(Paragraph("Medical Action Plan", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=4))

    action_summary = recommendations.get("action_summary", "")
    if action_summary:
        story.append(Paragraph(f"<b>Summary:</b> {action_summary}", styles["Body"]))
        story.append(Spacer(1, 3 * mm))

    bullet_items = recommendations.get("recommendations", [])
    bullet_table_data = []
    for i, item in enumerate(bullet_items):
        bg = BRAND_LIGHT if i % 2 == 0 else colors.white
        bullet_table_data.append([f"{i + 1}.", Paragraph(item, styles["BulletItem"])])

    if bullet_table_data:
        bullet_table = Table(bullet_table_data, colWidths=[8 * mm, 172 * mm])
        bullet_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_DARK),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, colors.white]),
        ]))
        story.append(bullet_table)

    story.append(Spacer(1, 8 * mm))

    # ── 4. Probability Breakdown table ──────────────────────────────────────
    breakdown = prediction_data.get("class_probability_breakdown") or prediction_data.get("probabilities", {})
    if breakdown:
        story.append(Paragraph("Probability Breakdown by Stage", styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=6))

        prob_header = [["Stage", "Label", "Probability (%)"]]
        prob_rows = []
        for label, prob in breakdown.items():
            parts = label.split(" - ", 1)
            stage_num = parts[0] if parts else label
            stage_label = parts[1] if len(parts) > 1 else ""
            prob_rows.append([stage_num, stage_label, f"{prob:.2f}%"])

        prob_data = prob_header + prob_rows
        prob_table = Table(prob_data, colWidths=[20 * mm, 120 * mm, 40 * mm])
        prob_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BRAND_LIGHT, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, DIVIDER),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(prob_table)

    story.append(Spacer(1, 10 * mm))

    # ── 5. Disclaimer footer ─────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER, spaceAfter=4))
    disclaimer_text = (
        "<b>Clinical Disclaimer:</b> This report is generated by an AI-assisted decision support tool "
        "and is intended to supplement, not replace, the clinical judgment of a qualified healthcare professional. "
        "All findings should be confirmed by a board-certified ophthalmologist or retinal specialist before any "
        "treatment or management decisions are made. This system is not FDA-cleared for standalone diagnostic use. "
        "Patient data is processed locally and not transmitted externally."
    )
    story.append(Paragraph(disclaimer_text, styles["Disclaimer"]))

    doc.build(story)
    return buffer.getvalue()


if __name__ == "__main__":
    # Quick self-test
    test_prediction = {
        "prediction": "3 - Severe Diabetic Retinopathy",
        "class_id": 3,
        "confidence": 82.4,
        "confidence_percentage": 82.4,
        "class_probability_breakdown": {
            "0 - No Diabetic Retinopathy": 2.1,
            "1 - Mild Diabetic Retinopathy": 5.3,
            "2 - Moderate Diabetic Retinopathy": 8.7,
            "3 - Severe Diabetic Retinopathy": 82.4,
            "4 - Proliferative Diabetic Retinopathy": 1.5,
        },
        "recommendations": {
            "stage": 3,
            "stage_label": "Severe Non-Proliferative DR",
            "urgency": "High",
            "urgency_color": "#ef4444",
            "action_summary": "Urgent retina specialist evaluation within 2–4 weeks.",
            "recommendations": [
                "Urgent referral to retinal specialist — evaluation within 2–4 weeks.",
                "Evaluate for panretinal photocoagulation (PRP).",
                "Aggressively optimize HbA1c, blood pressure, and lipid profile.",
            ],
            "follow_up_interval": "2–4 weeks",
            "icd10_code": "E11.3411",
        },
    }
    patient = {"name": "Test Patient", "patient_id": "TP-001", "scan_date": "2026-09-28"}
    pdf_bytes = generate_pdf_report(patient, test_prediction)
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF generated: test_report.pdf ({len(pdf_bytes)} bytes)")
    assert pdf_bytes[:4] == b"%PDF", "Invalid PDF header!"
    print("Self-test PASSED.")
