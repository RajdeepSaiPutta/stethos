"""
FHIR R4 + CDS Hooks Export for Stethos.
Maps inference results to FHIR DiagnosticReport, ServiceRequest, and CDS Hooks cards.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FHIRResourceType(Enum):
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    SERVICE_REQUEST = "ServiceRequest"
    OBSERVATION = "Observation"
    PATIENT = "Patient"
    ENCOUNTER = "Encounter"
    DOCUMENT_REFERENCE = "DocumentReference"


class LOINC_CODES:
    """Standard LOINC codes for respiratory observations."""
    # Breath sounds
    BREATH_SOUNDS = "59285-3"  # Breath sounds Auscultation
    CRACKLES = "253671-7"      # Crackles Auscultation
    WHEEZES = "253672-5"       # Wheezes Auscultation

    # CXR
    CHEST_XRAY = "36627-0"     # Chest X-ray
    CARDIO_THORACIC_RATIO = "8302-2"  # Cardiothoracic ratio
    LUNG_FIELD = "39806-6"     # Lung field finding

    # Spirometry
    FVC = "19900-8"            # Forced vital capacity
    FEV1 = "19905-7"           # Forced expiratory volume in 1 second
    FEV1_FVC = "19898-4"       # FEV1/FVC ratio
    FEF25_75 = "19914-9"       # Forced expiratory flow 25-75%
    PEF = "19916-4"            # Peak expiratory flow
    TLC = "19904-0"            # Total lung capacity
    RV = "19903-2"             # Residual volume
    DLCO = "20565-6"           # Diffusing capacity

    # Patterns
    OBSTRUCTIVE_PATTERN = "266937-2"
    RESTRICTIVE_PATTERN = "266938-0"


class SNOMED_CODES:
    """SNOMED-CT codes for findings."""
    NORMAL_BREATH = "162140006"
    CRACKLES = "609282006"
    WHEEZE = "43127000"
    CONSOLIDATION = "233604007"
    EFFUSION = "60046008"
    ATELECTASIS = "19829001"
    PNEUMONIA = "233604007"
    COPD = "13645005"
    ASTHMA = "195967001"
    PULMONARY_FIBROSIS = "27967001"


@dataclass
class FHIRCoding:
    """FHIR Coding element."""
    system: str
    code: str
    display: str

    def to_dict(self) -> Dict[str, str]:
        return {"system": self.system, "code": self.code, "display": self.display}


@dataclass
class FHIRCodeableConcept:
    """FHIR CodeableConcept element."""
    coding: List[FHIRCoding]
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"coding": [c.to_dict() for c in self.coding]}
        if self.text:
            result["text"] = self.text
        return result


@dataclass
class FHIRQuantity:
    """FHIR Quantity element."""
    value: float
    unit: str
    system: str = "http://unitsofmeasure.org"
    code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "system": self.system,
            "code": self.code
        }


class FHIRBundleBuilder:
    """Builds FHIR R4 bundles from Stethos inference results."""

    def __init__(self, patient_id: str, encounter_id: Optional[str] = None):
        self.patient_id = patient_id
        self.encounter_id = encounter_id or str(uuid.uuid4())
        self.entries = []

    def add_diagnostic_report(
        self,
        modality: str,
        result: Dict[str, Any],
        performer_id: str = "Stethos ML System",
        issued: Optional[datetime] = None
    ) -> str:
        """Create DiagnosticReport from modality result."""
        report_id = str(uuid.uuid4())
        issued = issued or datetime.utcnow()

        # Modality-specific coding
        modality_codes = {
            "breath": (LOINC_CODES.BREATH_SOUNDS, "Breath sounds auscultation"),
            "cxr": (LOINC_CODES.CHEST_XRAY, "Chest X-ray"),
            "spirometry": (LOINC_CODES.FVC, "Spirometry"),
            "fusion": ("95416-6", "Multimodal respiratory risk assessment")
        }

        code, display = modality_codes.get(modality, ("82575-2", "Respiratory assessment"))

        # Build conclusion text
        conclusion = self._build_conclusion(modality, result)

        report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"]
            },
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD" if modality == "cxr" else "PULM",
                    "display": "Radiology" if modality == "cxr" else "Pulmonary"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": code,
                    "display": display
                }]
            },
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "encounter": {"reference": f"Encounter/{self.encounter_id}"},
            "effectiveDateTime": issued.isoformat() + "Z",
            "issued": issued.isoformat() + "Z",
            "performer": [{"reference": f"Device/{performer_id}", "display": performer_id}],
            "conclusion": conclusion,
            "conclusionCode": self._build_conclusion_codes(modality, result),
            "result": self._build_observation_references(modality, result),
            "note": [{
                "text": f"Generated by Stethos ML v1.0. Inference time: {result.get('inference_time_ms', 'N/A')}ms"
            }]
        }

        if modality == "cxr" and "xray_image" in result.get("results", {}):
            # Add imaging study reference
            report["imagingStudy"] = [{
                "reference": f"ImagingStudy/{uuid.uuid4()}",
                "display": "Chest X-ray"
            }]

        self.entries.append({"resource": report, "fullUrl": f"urn:uuid:{report_id}"})
        return report_id

    def add_observations(
        self,
        modality: str,
        result: Dict[str, Any],
        issued: Optional[datetime] = None
    ) -> List[str]:
        """Create individual Observation resources for each finding."""
        issued = issued or datetime.utcnow()
        obs_ids = []

        if modality == "breath":
            obs_ids.extend(self._breath_observations(result, issued))
        elif modality == "cxr":
            obs_ids.extend(self._cxr_observations(result, issued))
        elif modality == "spirometry":
            obs_ids.extend(self._spirometry_observations(result, issued))

        return obs_ids

    def _breath_observations(self, result: Dict[str, Any], issued: datetime) -> List[str]:
        obs_ids = []
        r = result.get("results", {})
        pred = r.get("prediction", "normal")

        # Main breath sound observation
        obs_id = str(uuid.uuid4())
        obs = {
            "resourceType": "Observation",
            "id": obs_id,
            "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": LOINC_CODES.BREATH_SOUNDS,
                    "display": "Breath sounds Auscultation"
                }]
            },
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "encounter": {"reference": f"Encounter/{self.encounter_id}"},
            "effectiveDateTime": issued.isoformat() + "Z",
            "issued": issued.isoformat() + "Z",
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": SNOMED_CODES.NORMAL_BREATH if pred == "normal" else (SNOMED_CODES.CRACKLES if "crackle" in pred else SNOMED_CODES.WHEEZE),
                    "display": pred.capitalize()
                }]
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "N" if pred == "normal" else "A",
                    "display": "Normal" if pred == "normal" else "Abnormal"
                }]
            }],
            "note": [{"text": f"Confidence: {r.get('probabilities', {}).get(pred, 0):.1%}"}]
        }
        self.entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs_id}"})
        obs_ids.append(obs_id)

        return obs_ids

    def _cxr_observations(self, result: Dict[str, Any], issued: datetime) -> List[str]:
        obs_ids = []
        r = result.get("results", {})

        # Pathologies
        for pathology, prob in r.get("pathologies", {}).items():
            if prob > 0.5:
                obs_id = str(uuid.uuid4())
                obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging", "display": "Imaging"}]}],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": LOINC_CODES.CHEST_XRAY,
                            "display": "Chest X-ray finding"
                        }]
                    },
                    "subject": {"reference": f"Patient/{self.patient_id}"},
                    "encounter": {"reference": f"Encounter/{self.encounter_id}"},
                    "effectiveDateTime": issued.isoformat() + "Z",
                    "valueCodeableConcept": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": self._pathology_to_snomed(pathology),
                            "display": pathology
                        }]
                    },
                    "interpretation": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "A",
                            "display": "Abnormal"
                        }]
                    }],
                    "note": [{"text": f"Probability: {prob:.1%}"}]
                }
                self.entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs_id}"})
                obs_ids.append(obs_id)

        # Measurements
        for meas, value in r.get("measurements", {}).items():
            if meas == "cardiothoracic_ratio":
                obs_id = str(uuid.uuid4())
                obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging", "display": "Imaging"}]}],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": LOINC_CODES.CARDIO_THORACIC_RATIO,
                            "display": "Cardiothoracic ratio"
                        }]
                    },
                    "subject": {"reference": f"Patient/{self.patient_id}"},
                    "encounter": {"reference": f"Encounter/{self.encounter_id}"},
                    "effectiveDateTime": issued.isoformat() + "Z",
                    "valueQuantity": {
                        "value": float(value),
                        "unit": "ratio",
                        "system": "http://unitsofmeasure.org",
                        "code": "ratio"
                    }
                }
                self.entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs_id}"})
                obs_ids.append(obs_id)

        return obs_ids

    def _spirometry_observations(self, result: Dict[str, Any], issued: datetime) -> List[str]:
        obs_ids = []
        r = result.get("results", {})

        # PFT parameters
        pft_mapping = {
            "FVC": (LOINC_CODES.FVC, "L"),
            "FEV1": (LOINC_CODES.FEV1, "L"),
            "FEV1_FVC": (LOINC_CODES.FEV1_FVC, "%"),
            "FEF25_75": (LOINC_CODES.FEF25_75, "L/s"),
            "PEF": (LOINC_CODES.PEF, "L/s"),
            "TLC": (LOINC_CODES.TLC, "L"),
            "RV": (LOINC_CODES.RV, "L"),
            "DLCO": (LOINC_CODES.DLCO, "mmol/min/kPa")
        }

        for param, (loinc, unit) in pft_mapping.items():
            val = r.get("predicted_values", {}).get(param)
            if val is not None:
                obs_id = str(uuid.uuid4())
                obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": loinc,
                            "display": param
                        }]
                    },
                    "subject": {"reference": f"Patient/{self.patient_id}"},
                    "encounter": {"reference": f"Encounter/{self.encounter_id}"},
                    "effectiveDateTime": issued.isoformat() + "Z",
                    "valueQuantity": {
                        "value": float(val),
                        "unit": unit,
                        "system": "http://unitsofmeasure.org",
                        "code": unit
                    },
                    "interpretation": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": self._interpret_pft(param, float(val)),
                            "display": self._interpret_pft_display(param, float(val))
                        }]
                    }]
                }
                self.entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs_id}"})
                obs_ids.append(obs_id)

        # Pattern
        pattern = r.get("pattern", "Normal")
        if pattern != "Normal":
            obs_id = str(uuid.uuid4())
            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": {"profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]},
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": LOINC_CODES.OBSTRUCTIVE_PATTERN if "Obstructive" in pattern else LOINC_CODES.RESTRICTIVE_PATTERN,
                        "display": f"{pattern} pattern"
                    }]
                },
                "subject": {"reference": f"Patient/{self.patient_id}"},
                "encounter": {"reference": f"Encounter/{self.encounter_id}"},
                "effectiveDateTime": issued.isoformat() + "Z",
                "valueCodeableConcept": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": self._pattern_to_snomed(pattern),
                        "display": pattern
                    }]
                }
            }
            self.entries.append({"resource": obs, "fullUrl": f"urn:uuid:{obs_id}"})
            obs_ids.append(obs_id)

        return obs_ids

    def _build_conclusion(self, modality: str, result: Dict[str, Any]) -> str:
        r = result.get("results", {})

        if modality == "breath":
            pred = r.get("prediction", "normal")
            prob = r.get("probabilities", {}).get(pred, 0)
            return f"Breath sound analysis: {pred.capitalize()} detected ({prob:.1%} confidence). {r.get('interpretation', '')}"

        elif modality == "cxr":
            findings = r.get("positive_findings", [])
            if findings:
                return f"Chest X-ray: {len(findings)} finding(s) - {', '.join(findings[:3])}{'...' if len(findings) > 3 else ''}. CTR: {r.get('measurements', {}).get('cardiothoracic_ratio', 'N/A')}"
            return "Chest X-ray: No significant findings detected."

        elif modality == "spirometry":
            pattern = r.get("pattern", "Normal")
            gold = r.get("gold_stage", "")
            return f"Spirometry: {pattern} pattern{f' ({gold})' if gold else ''}. FEV1/FVC: {r.get('fev1_fvc_ratio', 'N/A'):.2f}"

        elif modality == "fusion":
            risk = r.get("fused_results", {}).get("risk_level", "Unknown")
            expl = r.get("explanation", "")
            return f"Multimodal fusion risk: {risk}. {expl}"

        return "Analysis complete."

    def _build_conclusion_codes(self, modality: str, result: Dict[str, Any]) -> List[Dict]:
        r = result.get("results", {})
        codes = []

        if modality == "breath":
            pred = r.get("prediction", "normal")
            if pred != "normal":
                codes.append({
                    "system": "http://snomed.info/sct",
                    "code": SNOMED_CODES.CRACKLES if "crackle" in pred else SNOMED_CODES.WHEEZE,
                    "display": pred.capitalize()
                })

        elif modality == "cxr":
            for finding in r.get("positive_findings", [])[:5]:
                codes.append({
                    "system": "http://snomed.info/sct",
                    "code": self._pathology_to_snomed(finding),
                    "display": finding
                })

        return codes

    def _build_observation_references(self, modality: str, result: Dict[str, Any]) -> List[Dict]:
        # Return references to the observation resources
        return []  # Simplified - would reference created observations

    def _pathology_to_snomed(self, pathology: str) -> str:
        mapping = {
            "Atelectasis": "19829001",
            "Cardiomegaly": "3700000",
            "Consolidation": "233604007",
            "Edema": "46750008",
            "Effusion": "60046008",
            "Emphysema": "13645005",
            "Fibrosis": "27967001",
            "Hernia": "43913007",
            "Infiltration": "299509004",
            "Mass": "118939006",
            "Nodule": "31100000",
            "Pleural_Thickening": "26181000",
            "Pneumonia": "233604007",
            "Pneumothorax": "60046008"
        }
        return mapping.get(pathology, "118939006")

    def _pattern_to_snomed(self, pattern: str) -> str:
        mapping = {
            "Obstructive": "13645005",
            "Restrictive": "27967001",
            "Mixed": "13645005"
        }
        return mapping.get(pattern, "118939006")

    def _interpret_pft(self, param: str, value: float) -> str:
        # Simplified interpretation codes
        return "N"

    def _interpret_pft_display(self, param: str, value: float) -> str:
        return "Normal"

    def build_bundle(self) -> Dict[str, Any]:
        """Build the complete FHIR Bundle."""
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "id": str(uuid.uuid4()),
            "meta": {
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "entry": self.entries
        }

    def save_bundle(self, path: str) -> None:
        """Save bundle to file."""
        bundle = self.build_bundle()
        with open(path, 'w') as f:
            json.dump(bundle, f, indent=2)
        logger.info(f"Saved FHIR bundle to {path}")


class CDSHooksCardBuilder:
    """Build CDS Hooks cards for EHR integration."""

    @staticmethod
    def create_card(
        summary: str,
        detail: str,
        indicator: str = "info",
        source: str = "Stethos ML",
        links: Optional[List[Dict]] = None,
        suggestions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Create a CDS Hooks card."""
        card = {
            "summary": summary,
            "detail": detail,
            "indicator": indicator,  # info, warning, critical
            "source": {"label": source},
        }
        if links:
            card["links"] = links
        if suggestions:
            card["suggestions"] = suggestions
        return card

    @classmethod
    def from_fusion_result(cls, fusion_result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Create CDS Hooks response from fusion result."""
        r = fusion_result.get("results", {}) if "results" in fusion_result else fusion_result
        risk = r.get("fused_results", {}).get("risk_level", "Unknown")
        explanation = r.get("explanation", "")

        # Determine indicator
        indicator_map = {
            "Very High": "critical",
            "High": "warning",
            "Moderate": "warning",
            "Low": "info",
            "Very Low": "info"
        }
        indicator = indicator_map.get(risk, "info")

        # Build suggestions
        suggestions = []
        if risk in ["Very High", "High"]:
            suggestions.append({
                "label": "Order urgent chest CT",
                "uuid": str(uuid.uuid4()),
                "actions": [{
                    "type": "create",
                    "resource": {
                        "resourceType": "ServiceRequest",
                        "status": "active",
                        "intent": "order",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "24628-9", "display": "CT Chest"}]},
                        "subject": {"reference": f"Patient/{patient_id}"}
                    }
                }]
            })
            suggestions.append({
                "label": "Refer to pulmonology",
                "uuid": str(uuid.uuid4()),
                "actions": [{
                    "type": "create",
                    "resource": {
                        "resourceType": "ServiceRequest",
                        "status": "active",
                        "intent": "order",
                        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "409053007", "display": "Referral to pulmonology"}]},
                        "subject": {"reference": f"Patient/{patient_id}"}
                    }
                }]
            })

        cards = [cls.create_card(
            summary=f"Multimodal Respiratory Risk: {risk}",
            detail=explanation or f"Fusion of breath sounds, CXR, and spirometry indicates {risk} risk.",
            indicator=indicator,
            suggestions=suggestions
        )]

        # Add modality-specific cards
        modalities = r.get("modalities", {}) if "modalities" in r else {}
        for mod_name, mod_output in modalities.items():
            if mod_output.get("success") and mod_output.get("results", {}):
                mod_cards = cls._modality_cards(mod_name, mod_output)
                cards.extend(mod_cards)

        return {"cards": cards}

    @staticmethod
    def _modality_cards(modality: str, result: Dict[str, Any]) -> List[Dict]:
        r = result.get("results", {})
        cards = []

        if modality == "cxr":
            findings = r.get("positive_findings", [])
            if findings:
                cards.append(CDSHooksCardBuilder.create_card(
                    summary=f"CXR: {len(findings)} finding(s)",
                    detail=f"Positive findings: {', '.join(findings[:3])}",
                    indicator="warning" if any(f in ["Pneumonia", "Consolidation", "Mass"] for f in findings) else "info"
                ))

        elif modality == "spirometry":
            pattern = r.get("pattern", "Normal")
            if pattern != "Normal":
                cards.append(CDSHooksCardBuilder.create_card(
                    summary=f"Spirometry: {pattern} pattern",
                    detail=f"FEV1/FVC: {r.get('fev1_fvc_ratio', 0):.2f}. GOLD: {r.get('gold_stage', 'N/A')}",
                    indicator="warning" if "Obstructive" in pattern else "info"
                ))

        return cards


def export_fhir_from_stethos(
    patient_id: str,
    modality_results: Dict[str, Any],
    fusion_result: Optional[Dict[str, Any]] = None,
    patient_context: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export Stethos results as FHIR Bundle + CDS Hooks cards.

    Args:
        patient_id: Patient identifier
        modality_results: Dict of modality -> Stethos result
        fusion_result: Optional fusion result
        patient_context: Optional patient demographics
        output_path: Optional file path to save bundle
    Returns:
        Dict with 'bundle' and 'cds_hooks' keys
    """
    builder = FHIRBundleBuilder(patient_id)

    issued = datetime.utcnow()

    # Add modality diagnostic reports and observations
    for modality, result in modality_results.items():
        if result.get("success"):
            builder.add_diagnostic_report(modality, result, issued=issued)
            builder.add_observations(modality, result, issued=issued)

    # Add fusion report
    if fusion_result and fusion_result.get("success"):
        builder.add_diagnostic_report("fusion", fusion_result, issued=issued)

    # Build outputs
    bundle = builder.build_bundle()
    cds_hooks = CDSHooksCardBuilder.from_fusion_result(fusion_result or {}, patient_id)

    output = {"bundle": bundle, "cds_hooks": cds_hooks}

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Exported FHIR + CDS Hooks to {output_path}")

    return output