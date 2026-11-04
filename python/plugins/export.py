"""
Report export utilities for Stethos.
Supports PDF, DICOM-SR, FHIR, and JSON formats.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def export_report(format: str, data: Dict[str, Any], output_path: str) -> str:
    """
    Export report in specified format.

    Args:
        format: "pdf", "dicom-sr", "fhir", "json"
        data: Report data (patient context, modality results, fusion)
        output_path: Output file path

    Returns:
        Path to exported file
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        return str(output)

    elif format == "fhir":
        # FHIR export is handled by dedicated endpoint
        from plugins.fhir_export import export_fhir_from_stethos
        patient_id = data.get("patient_context", {}).get("patient_id", "unknown")
        modality_results = {
            "breath": data.get("breath", {}),
            "cxr": data.get("cxr", {}),
            "spirometry": data.get("spirometry", {})
        }
        fusion_result = data.get("fusion", {})
        patient_context = data.get("patient_context", {})
        result = export_fhir_from_stethos(patient_id, modality_results, fusion_result, patient_context, str(output))
        return str(output)

    elif format == "pdf":
        # Generate PDF using reportlab or weasyprint
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(str(output), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
            story.append(Paragraph("Stethos Respiratory Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Patient info
            patient = data.get("patient_context", {})
            story.append(Paragraph("Patient Information", styles['Heading2']))
            patient_data = [
                ["Patient ID", patient.get("patient_id", "N/A")],
                ["Age", str(patient.get("age", "N/A"))],
                ["Sex", patient.get("sex", "N/A")],
                ["Height (cm)", str(patient.get("height_cm", "N/A"))],
                ["Weight (kg)", str(patient.get("weight_kg", "N/A"))],
                ["BMI", str(patient.get("bmi", "N/A"))]
            ]
            table = Table(patient_data, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 24))

            # Modality results
            for modality in ["breath", "cxr", "spirometry", "fusion"]:
                if modality in data and data[modality].get("success"):
                    story.append(Paragraph(f"{modality.capitalize()} Analysis", styles['Heading2']))
                    _add_modality_to_pdf(story, modality, data[modality], styles)
                    story.append(PageBreak())

            doc.build(story)
            return str(output)

        except ImportError:
            logger.warning("reportlab not available, falling back to JSON")
            with open(output, 'w') as f:
                json.dump(data, f, indent=2)
            return str(output)

    elif format == "dicom-sr":
        # DICOM Structured Report - simplified placeholder
        # Full implementation would use pydicom
        sr_data = {
            "format": "DICOM-SR",
            "content": data,
            "note": "Full DICOM-SR implementation requires pydicom with SR module"
        }
        with open(output, 'w') as f:
            json.dump(sr_data, f, indent=2)
        return str(output)

    else:
        raise ValueError(f"Unsupported format: {format}")


def _add_modality_to_pdf(story, modality: str, result: Dict[str, Any], styles):
    """Add modality results to PDF story."""
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors

    r = result.get("results", {})

    if modality == "breath":
        pred = r.get("prediction", "N/A")
        probs = r.get("probabilities", {})
        interp = r.get("interpretation", "")

        story.append(Paragraph(f"Prediction: <b>{pred.capitalize()}</b>", styles['Normal']))
        story.append(Paragraph(f"Confidence: {probs.get(pred, 0):.1%}", styles['Normal']))
        story.append(Spacer(1, 6))

        prob_data = [["Class", "Probability"]]
        for cls, prob in probs.items():
            prob_data.append([cls.capitalize(), f"{prob:.1%}"])
        table = Table(prob_data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        if interp:
            story.append(Paragraph(f"Interpretation: {interp}", styles['Normal']))

    elif modality == "cxr":
        findings = r.get("positive_findings", [])
        story.append(Paragraph(f"Positive Findings: {len(findings)}", styles['Normal']))
        if findings:
            for f in findings:
                story.append(Paragraph(f"• {f}", styles['Normal']))

        meas = r.get("measurements", {})
        if meas:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Measurements", styles['Heading3']))
            meas_data = [["Parameter", "Value"]]
            for k, v in meas.items():
                meas_data.append([k.replace('_', ' ').title(), str(v)])
            table = Table(meas_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

    elif modality == "spirometry":
        pattern = r.get("pattern", "N/A")
        gold = r.get("gold_stage", "N/A")
        story.append(Paragraph(f"Pattern: <b>{pattern}</b>", styles['Normal']))
        story.append(Paragraph(f"GOLD Stage: {gold}", styles['Normal']))
        story.append(Paragraph(f"FEV1/FVC: {r.get('fev1_fvc_ratio', 'N/A')}", styles['Normal']))

        pred_vals = r.get("predicted_values", {})
        if pred_vals:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Predicted Values (z-scores)", styles['Heading3']))
            pv_data = [["Parameter", "z-score"]]
            for k, v in pred_vals.items():
                pv_data.append([k, f"{v:.2f}"])
            table = Table(pv_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

    elif modality == "fusion":
        fused = result.get("results", {}).get("fused_results", {})
        risk = fused.get("risk_level", "N/A")
        story.append(Paragraph(f"Risk Level: <b>{risk}</b>", styles['Normal']))

        contributions = result.get("results", {}).get("modality_contributions", {})
        if contributions:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Modality Contributions", styles['Heading3']))
            contrib_data = [["Modality", "Contribution"]]
            for k, v in contributions.items():
                contrib_data.append([k.capitalize(), f"{v:.1%}"])
            table = Table(contrib_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

        explanation = result.get("results", {}).get("explanation", "")
        if explanation:
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Explanation: {explanation}", styles['Normal']))

    story.append(Spacer(1, 12))