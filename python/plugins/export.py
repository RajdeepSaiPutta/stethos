"""
Export utilities for clinical reports
Generates PDF, DICOM-SR, FHIR Bundle, and JSON reports
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def export_report(format: str, data: Dict[str, Any], output_path: str) -> str:
    """Export clinical report in specified format."""
    format = format.lower()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        return _export_json(data, output_path)
    elif format == "pdf":
        return _export_pdf(data, output_path)
    elif format == "fhir":
        return _export_fhir(data, output_path)
    elif format == "dicom-sr":
        return _export_dicom_sr(data, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")

def _export_json(data: Dict[str, Any], output_path: Path) -> str:
    """Export as JSON."""
    export_data = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0",
            "generator": "Respiratory AI Workstation"
        },
        "clinical_data": data
    }
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    return str(output_path)

def _export_pdf(data: Dict[str, Any], output_path: Path) -> str:
    """Export as clinical PDF report."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        import io
        import base64
    except ImportError:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        'ClinicalTitle', parent=styles['Title'],
        fontSize=18, spaceAfter=12, textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'ClinicalHeading', parent=styles['Heading2'],
        fontSize=13, spaceBefore=12, spaceAfter=6, textColor=HexColor('#16213e')
    ))
    styles.add(ParagraphStyle(
        'ClinicalBody', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'RiskHigh', parent=styles['Normal'],
        fontSize=11, leading=14, textColor=HexColor('#dc2626'), spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'RiskModerate', parent=styles['Normal'],
        fontSize=11, leading=14, textColor=HexColor('#ea580c'), spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'RiskLow', parent=styles['Normal'],
        fontSize=11, leading=14, textColor=HexColor('#16a34a'), spaceAfter=6
    ))

    story = []

    # Header
    story.append(Paragraph("Respiratory AI Workstation — Clinical Report", styles['ClinicalTitle']))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles['ClinicalBody']))
    story.append(Spacer(1, 12))

    # Patient info
    if data.get("patient_context"):
        ctx = data["patient_context"]
        story.append(Paragraph("Patient Information", styles['ClinicalHeading']))
        patient_data = [
            ["Field", "Value"],
            ["Patient ID", ctx.get("patient_id", "N/A")],
            ["Age", str(ctx.get("age", "N/A"))],
            ["Sex", ctx.get("sex", "N/A")],
            ["Height (cm)", str(ctx.get("height_cm", "N/A"))],
            ["Weight (kg)", str(ctx.get("weight_kg", "N/A"))],
            ["BMI", str(ctx.get("bmi", "N/A"))],
        ]
        t = Table(patient_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), HexColor('#ffffff')]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Fusion risk assessment
    if data.get("fusion"):
        fusion = data["fusion"]
        story.append(Paragraph("Multimodal Risk Assessment", styles['ClinicalHeading']))

        risk_level = fusion.get("fused_results", {}).get("risk_level", "Unknown")
        risk_score = fusion.get("fused_results", {}).get("risk_score", 0)
        confidence = fusion.get("confidence", 0)
        explanation = fusion.get("explanation", "")

        risk_style = {
            "Very High": "RiskHigh", "High": "RiskHigh",
            "Moderate": "RiskModerate",
            "Low": "RiskLow", "Very Low": "RiskLow"
        }.get(risk_level, "ClinicalBody")

        story.append(Paragraph(f"<b>Overall Risk:</b> {risk_level} (Score: {risk_score:.1%}, Confidence: {confidence:.1%})", styles[risk_style]))
        story.append(Paragraph(explanation, styles['ClinicalBody']))
        story.append(Spacer(1, 6))

        # Modality contributions
        contrib = fusion.get("modality_contributions", {})
        if contrib:
            story.append(Paragraph("Modality Contributions", styles['ClinicalHeading']))
            contrib_data = [["Modality", "Contribution"]]
            for mod, val in sorted(contrib.items(), key=lambda x: x[1], reverse=True):
                contrib_data.append([mod.replace("_", " ").title(), f"{val:.1%}"])
            t = Table(contrib_data, colWidths=[3*inch, 2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

    # Individual modality results
    for mod_name in ["breath", "cxr", "spirometry"]:
        if data.get(mod_name):
            mod_data = data[mod_name]
            display_name = {
                "breath": "Breath Sound Analysis",
                "cxr": "Chest X-Ray Analysis",
                "spirometry": "Spirometry Analysis"
            }.get(mod_name, mod_name.title())

            story.append(Paragraph(display_name, styles['ClinicalHeading']))

            if mod_name == "breath":
                pred = mod_data.get("results", {}).get("prediction", "N/A")
                probs = mod_data.get("results", {}).get("probabilities", {})
                interp = mod_data.get("results", {}).get("interpretation", "")
                story.append(Paragraph(f"<b>Classification:</b> {pred.title()}", styles['ClinicalBody']))
                prob_data = [["Class", "Probability"]]
                for cls, prob in probs.items():
                    prob_data.append([cls.title(), f"{prob:.1%}"])
                t = Table(prob_data, colWidths=[2*inch, 2*inch])
                t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc'))]))
                story.append(t)
                story.append(Paragraph(interp, styles['ClinicalBody']))

            elif mod_name == "cxr":
                findings = mod_data.get("results", {}).get("positive_findings", [])
                measurements = mod_data.get("results", {}).get("measurements", {})
                if findings:
                    story.append(Paragraph(f"<b>Positive Findings:</b> {', '.join(findings)}", styles['ClinicalBody']))
                if measurements:
                    meas_data = [["Measurement", "Value"]]
                    for k, v in measurements.items():
                        meas_data.append([k.replace("_", " ").title(), f"{v:.3f}" if isinstance(v, float) else str(v)])
                    t = Table(meas_data, colWidths=[3*inch, 2*inch])
                    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc'))]))
                    story.append(t)
                # Segmentation overlay
                overlay = mod_data.get("results", {}).get("segmentation_overlay")
                if overlay and overlay.startswith("data:image"):
                    try:
                        img_data = base64.b64decode(overlay.split(",")[1])
                        img = RLImage(io.BytesIO(img_data), width=3*inch, height=3*inch)
                        story.append(Paragraph("<b>Segmentation Overlay:</b>", styles['ClinicalBody']))
                        story.append(img)
                    except Exception:
                        pass

            elif mod_name == "spirometry":
                pattern = mod_data.get("results", {}).get("pattern", "N/A")
                gold = mod_data.get("results", {}).get("gold_stage", "N/A")
                predicted = mod_data.get("results", {}).get("predicted_values", {})
                trend = mod_data.get("results", {}).get("trend", "N/A")
                story.append(Paragraph(f"<b>Pattern:</b> {pattern}", styles['ClinicalBody']))
                story.append(Paragraph(f"<b>GOLD Stage:</b> {gold}", styles['ClinicalBody']))
                story.append(Paragraph(f"<b>Trend:</b> {trend}", styles['ClinicalBody']))
                if predicted:
                    pred_data = [["Parameter", "Predicted"]]
                    for k, v in predicted.items():
                        pred_data.append([k, f"{v:.2f}" if isinstance(v, float) else str(v)])
                    t = Table(pred_data, colWidths=[2*inch, 2*inch])
                    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc'))]))
                    story.append(t)

            story.append(Spacer(1, 12))

    # Footer
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "<i>This report is generated by AI and should be reviewed by a qualified clinician. "
        "It is not a substitute for professional medical judgment.</i>",
        styles['ClinicalBody']
    ))

    doc.build(story)
    return str(output_path)

def _export_fhir(data: Dict[str, Any], output_path: Path) -> str:
    """Export as FHIR R4 Bundle."""
    try:
        from fhir.resources.bundle import Bundle
        from fhir.resources.composition import Composition
        from fhir.resources.observation import Observation
        from fhir.resources.patient import Patient
        from fhir.resources.diagnosticreport import DiagnosticReport
        from fhir.resources.reference import Reference
        from fhir.resources.codeableconcept import CodeableConcept
        from fhir.resources.coding import Coding
        from fhir.resources.quantity import Quantity
    except ImportError:
        raise RuntimeError("fhir.resources not installed. Run: pip install fhir.resources")

    bundle = Bundle(type="document", timestamp=datetime.utcnow())

    # Patient
    patient_ctx = data.get("patient_context", {})
    patient = Patient(
        id=patient_ctx.get("patient_id", "unknown"),
        identifier=[{"system": "http://hospital.example.org/patients", "value": patient_ctx.get("patient_id", "unknown")}],
        gender=patient_ctx.get("sex", "unknown").lower() if patient_ctx.get("sex") else "unknown",
        birthDate=_calculate_birthdate(patient_ctx.get("age")) if patient_ctx.get("age") else None
    )
    bundle.entry.append(Bundle.Entry(resource=patient, fullUrl=f"Patient/{patient.id}"))

    # Composition (the document)
    composition = Composition(
        status="final",
        type=CodeableConcept(coding=[Coding(system="http://loinc.org", code="11502-2", display="Clinical Report")]),
        subject=Reference(reference=f"Patient/{patient.id}"),
        date=datetime.utcnow(),
        title="Respiratory AI Workstation Multimodal Assessment",
        author=[Reference(reference=f"Patient/{patient.id}")]
    )
    bundle.entry.append(Bundle.Entry(resource=composition, fullUrl="Composition/report"))

    # Observations from each modality
    for mod_name in ["breath", "cxr", "spirometry"]:
        if data.get(mod_name):
            obs_list = _create_fhir_observations(data[mod_name], patient.id, mod_name)
            for obs in obs_list:
                bundle.entry.append(Bundle.Entry(resource=obs, fullUrl=f"Observation/{obs.id}"))

    # Diagnostic Report
    diag_report = DiagnosticReport(
        status="final",
        category=[CodeableConcept(coding=[Coding(system="http://terminology.hl7.org/CodeSystem/v2-0074", code="LP", display="Laboratory")])],
        code=CodeableConcept(coding=[Coding(system="http://loinc.org", code="11502-2", display="Clinical Report")]),
        subject=Reference(reference=f"Patient/{patient.id}"),
        effectiveDateTime=datetime.utcnow(),
        conclusion=_generate_fhir_conclusion(data)
    )
    bundle.entry.append(Bundle.Entry(resource=diag_report, fullUrl="DiagnosticReport/report"))

    with open(output_path, 'w') as f:
        json.dump(bundle.dict(), f, indent=2)

    return str(output_path)

def _calculate_birthdate(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    from datetime import date
    birth_year = date.today().year - age
    return f"{birth_year}-01-01"

def _create_fhir_observations(mod_data: Dict[str, Any], patient_id: str, mod_name: str) -> list:
    """Create FHIR Observations from modality results."""
    from fhir.resources.observation import Observation
    from fhir.resources.codeableconcept import CodeableConcept
    from fhir.resources.coding import Coding
    from fhir.resources.quantity import Quantity
    from fhir.resources.reference import Reference

    observations = []
    results = mod_data.get("results", {})

    if mod_name == "breath":
        pred = results.get("prediction")
        if pred:
            obs = Observation(
                status="final",
                category=[CodeableConcept(coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="exam", display="Examination")])],
                code=CodeableConcept(coding=[Coding(system="http://loinc.org", code="92791-5", display="Adventitious lung sounds")]),
                subject=Reference(reference=f"Patient/{patient_id}"),
                valueCodeableConcept=CodeableConcept(
                    coding=[Coding(system="http://snomed.info/sct", code=_breath_snomed(pred), display=pred.title())],
                    text=pred.title()
                ),
                interpretation=[CodeableConcept(coding=[Coding(system="http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", code="A", display="Abnormal")])] if pred != "normal" else []
            )
            observations.append(obs)

    elif mod_name == "spirometry":
        predicted = results.get("predicted_values", {})
        for param, value in predicted.items():
            if value > 0:
                obs = Observation(
                    status="final",
                    category=[CodeableConcept(coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="laboratory", display="Laboratory")])],
                    code=CodeableConcept(coding=[Coding(system="http://loinc.org", code=_spirometry_loinc(param), display=param)]),
                    subject=Reference(reference=f"Patient/{patient_id}"),
                    valueQuantity=Quantity(value=float(value), unit=_spirometry_unit(param), system="http://unitsofmeasure.org", code=_spirometry_ucum(param))
                )
                observations.append(obs)

    return observations

def _breath_snomed(pred: str) -> str:
    mapping = {
        "normal": "275287008",      # Normal breath sounds
        "crackle": "60046008",      # Crackles
        "wheeze": "431279007",      # Wheezing
        "both": "424981000124103"   # Crackles and wheezes
    }
    return mapping.get(pred, "275287008")

def _spirometry_loinc(param: str) -> str:
    mapping = {
        "FVC": "19924-1",
        "FEV1": "20154-2",
        "FEV1_FVC": "19925-8",
        "FEF25_75": "20158-3",
        "PEF": "19926-6",
    }
    return mapping.get(param, "19924-1")

def _spirometry_unit(param: str) -> str:
    mapping = {
        "FVC": "L", "FEV1": "L", "FEV1_FVC": "%",
        "FEF25_75": "L/s", "PEF": "L/s",
    }
    return mapping.get(param, "L")

def _spirometry_ucum(param: str) -> str:
    mapping = {
        "FVC": "L", "FEV1": "L", "FEV1_FVC": "%",
        "FEF25_75": "L/s", "PEF": "L/s",
    }
    return mapping.get(param, "L")

def _generate_fhir_conclusion(data: Dict[str, Any]) -> str:
    parts = []
    if data.get("fusion"):
        fusion = data["fusion"]
        risk = fusion.get("fused_results", {}).get("risk_level", "")
        if risk:
            parts.append(f"Overall respiratory risk: {risk}.")
        exp = fusion.get("explanation", "")
        if exp:
            parts.append(exp)
    return " ".join(parts) if parts else "Multimodal respiratory assessment completed."

def _export_dicom_sr(data: Dict[str, Any], output_path: Path) -> str:
    """Export as DICOM Structured Report (simplified)."""
    try:
        import pydicom
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from pydicom.uid import generate_uid, PYDICOM_IMPLEMENTATION_UID
    except ImportError:
        raise RuntimeError("pydicom not installed. Run: pip install pydicom")

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.22"  # Basic Text SR
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = data.get("patient_context", {}).get("patient_id", "ANONYMOUS")
    ds.PatientID = data.get("patient_context", {}).get("patient_id", "ANONYMOUS")
    ds.ContentDate = datetime.utcnow().strftime("%Y%m%d")
    ds.ContentTime = datetime.utcnow().strftime("%H%M%S")
    ds.Modality = "SR"
    ds.SeriesDescription = "Respiratory AI Workstation Multimodal Report"

    # Content sequence (simplified)
    content_seq = Sequence()
    # Would need full TID 1500 / TID 5000 implementation
    ds.ContentSequence = content_seq

    ds.save_as(str(output_path))
    return str(output_path)