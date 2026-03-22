import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime

import numpy as np


class MedicalPredictionEvaluator:
    """
    对已有 prediction 文件进行离线评估：
    1) 医学表达代理指标
    2) 医学事实一致性指标
    3) 空间描述/空间关系一致性指标
    4) evidence support 指标
    5) 多实验对比输出
    6) 按 modality / question type 分组统计
    7) 相对 baseline 的 case-level delta 分析
    """

    def __init__(self, dataset_file: str, output_dir: str = "./evaluation_results"):
        self.dataset_file = dataset_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset_index = self._load_dataset_index(dataset_file)
        self.lexicons = self._build_medical_lexicons()

        self.medical_terms = sorted(
            set(
                self.lexicons["finding_terms"]
                + self.lexicons["anatomy_terms"]
                + self.lexicons["attribute_terms"]
                + self.lexicons["report_terms"]
                + self.lexicons["spatial_terms"]
                + self.lexicons["spatial_relation_terms"]
            ),
            key=lambda x: (-len(x), x)
        )

        self.clinical_reasoning_keywords = [
            "differential diagnosis", "likely", "consistent with", "suggestive of",
            "rule out", "consider", "possible", "probable", "suspicious for",
            "characteristic of", "typical of", "atypical", "pathognomonic"
        ]

        # 主指标：后续 delta 分析用这几个最关键
        self.primary_metrics = [
            "medical_fact_score",
            "spatial_fact_score",
            "finding_f1",
            "attribute_f1",
            "laterality_f1"
            # "anatomy_f1",
            # "polarity_match"
        ]

    # =========================
    # 数据加载
    # =========================
    def _load_dataset_index(self, dataset_file: str) -> Dict[str, Dict[str, Any]]:
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = {}
        for item in data:
            sample_id = item.get("id")
            if not sample_id:
                continue

            conversations = item.get("conversations", [])
            question = ""
            reference = ""

            if len(conversations) >= 1:
                question = conversations[0].get("value", "").replace("<image>", "").strip()
            if len(conversations) >= 2:
                reference = conversations[1].get("value", "").replace("<image>", "").strip()

            index[sample_id] = {
                "id": sample_id,
                "question": question,
                "reference": reference,
                "modality": item.get("modality", "Unknown"),
                "body_part": item.get("body_part", "Unknown"),
                "image": item.get("image", [])
            }

        print(f"Loaded dataset index: {len(index)} samples")
        return index

    def load_predictions_file(self, pred_file: str) -> List[Dict[str, Any]]:
        pred_file = str(pred_file)

        if pred_file.endswith(".jsonl"):
            records = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        if pred_file.endswith(".json"):
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "predictions" in data:
                return data["predictions"]

            raise ValueError(f"Unsupported JSON structure in {pred_file}")

        raise ValueError(f"Unsupported file format: {pred_file}")

    def normalize_predictions(self, raw_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []

        for item in raw_predictions:
            sample_id = item.get("id") or item.get("sample_id")
            if not sample_id:
                continue

            dataset_item = self.dataset_index.get(sample_id, {})

            prediction = (
                item.get("prediction")
                or item.get("final_report")
                or item.get("answer")
                or item.get("output")
                or ""
            ).strip()

            reference = (
                item.get("reference")
                or dataset_item.get("reference", "")
            ).strip()

            question = (
                item.get("question")
                or dataset_item.get("question", "")
            ).strip()

            modality = item.get("modality") or dataset_item.get("modality", "Unknown")
            body_part = item.get("body_part") or dataset_item.get("body_part", "Unknown")

            evidence_summary = (
                item.get("evidence_summary")
                or item.get("evidence")
                or ""
            ).strip()

            metadata = item.get("metadata", {})

            normalized.append({
                "id": sample_id,
                "prediction": prediction,
                "reference": reference,
                "question": question,
                "modality": modality,
                "body_part": body_part,
                "evidence_summary": evidence_summary,
                "metadata": metadata
            })

        return normalized

    # =========================
    # 词表
    # =========================
    def _build_medical_lexicons(self) -> Dict[str, List[str]]:
        finding_terms = [
            "lesion", "mass", "nodule", "cyst", "tumor", "neoplasm",
            "calcification", "hemorrhage", "edema", "inflammation", "fibrosis",
            "fracture", "effusion", "consolidation", "opacity", "pneumothorax",
            "atelectasis", "stenosis", "occlusion", "thrombosis", "embolism",
            "metastasis", "metastatic", "ischemia", "infarction", "necrosis",
            "vessel", "lymphatic vessels", "lymphatic vessel", "collagen",
            "staining", "stained", "green-stained", "red-stained",
            "aggregation", "density", "echogenicity", "echogenicities"
        ]

        anatomy_terms = [
            "lung", "liver", "kidney", "brain", "heart", "pleura", "spine",
            "parenchyma", "cortex", "medulla", "capsule", "septum",
            "ventricle", "atrium", "vessel", "artery", "vein",
            "lymph node", "organ", "tissue", "structure", "region",
            "cell", "cells", "cellular", "collagen", "lymphatic",
            "spleen", "abdomen", "ultrasound"
        ]

        attribute_terms = [
            "hyperechoic", "hypoechoic", "echogenic", "anechoic", "isoechoic",
            "hyperdense", "hypodense", "isodense", "hyperintense", "hypointense",
            "enhancement", "contrast", "signal", "intensity", "attenuation",
            "benign", "malignant", "invasive", "infiltrative",
            "acute", "chronic", "subacute", "progressive", "stable",
            "dilatation", "hypertrophy", "atrophy", "enlarged",
            "highest", "increased", "decreased", "close proximity",
            "intermingled", "surrounded", "homogeneous", "heterogeneous",
            "denser", "decreased blood flow"
        ]

        report_terms = [
            "findings", "impression", "conclusion", "assessment",
            "normal", "abnormal", "pathological", "physiological",
            "follow-up", "correlation", "clinical", "recommend",
            "suggest", "monitor", "surveillance", "biopsy",
            "intervention", "treatment", "diagnostic focus"
        ]

        spatial_terms = [
            "left", "right", "bilateral",
            "upper", "lower", "central", "centrally", "peripheral",
            "anterior", "posterior", "superior", "inferior",
            "medial", "lateral",
            "middle", "top", "bottom",
            "upper region", "lower region", "central region"
        ]

        spatial_relation_terms = [
            "adjacent", "surrounded", "intermingled", "close proximity",
            "near", "around", "within", "beside", "between",
            "overlapping", "separated from", "distributed around",
            "located in", "positioned in", "relative to", "surrounding"
        ]

        negation_terms = [
            "no", "not", "without", "absent", "negative for",
            "free of", "no evidence of", "no definite"
        ]

        uncertain_terms = [
            "possible", "possibly", "may", "could", "likely",
            "probable", "suggestive of", "consistent with", "suspicious for"
        ]

        return {
            "finding_terms": finding_terms,
            "anatomy_terms": anatomy_terms,
            "attribute_terms": attribute_terms,
            "report_terms": report_terms,
            "spatial_terms": spatial_terms,
            "spatial_relation_terms": spatial_relation_terms,
            "negation_terms": negation_terms,
            "uncertain_terms": uncertain_terms
        }

    # =========================
    # 文本工具
    # =========================
    def _contains_term(self, text: str, term: str) -> bool:
        text = text.lower()
        term = term.lower()
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        return re.search(pattern, text) is not None

    def _extract_terms(self, text: str, terms: List[str]) -> Set[str]:
        text = text.lower()
        found = set()
        for term in terms:
            if self._contains_term(text, term):
                found.add(term)
        return found

    def _set_f1(self, pred_set: Set[str], ref_set: Set[str]) -> float:
        if not pred_set and not ref_set:
            return 1.0
        if not pred_set or not ref_set:
            return 0.0

        inter = len(pred_set & ref_set)
        precision = inter / len(pred_set) if pred_set else 0.0
        recall = inter / len(ref_set) if ref_set else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # =========================
    # 问题类型识别
    # =========================
    def _classify_question_type(self, question: str, modality: str) -> str:
        q = (question or "").lower()
        modality = (modality or "").lower()

        spatial_cues = [
            "where", "position", "relative position", "relative to",
            "located", "location", "area", "region", "concentration",
            "highest", "distribution", "surrounded", "intermingled"
        ]
        diagnostic_cues = [
            "what might these differences indicate",
            "what does this indicate",
            "most likely diagnosis",
            "suggest", "consistent with", "diagnosis",
            "condition", "abnormality", "pathological"
        ]

        if any(cue in q for cue in spatial_cues):
            return "spatial_descriptive"

        if modality in ["microscopy images", "microscopy image", "pathology"] and any(cue in q for cue in spatial_cues):
            return "spatial_descriptive"

        if any(cue in q for cue in diagnostic_cues):
            return "diagnostic_interpretive"

        if modality in ["ultrasound", "ct", "mri", "x-ray", "radiology"]:
            return "diagnostic_interpretive"

        return "general_medical_vqa"

    # =========================
    # 事实抽取
    # =========================
    def _extract_medical_facts(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        anatomy = self._extract_terms(text_lower, self.lexicons["anatomy_terms"])
        findings = self._extract_terms(text_lower, self.lexicons["finding_terms"])
        attributes = self._extract_terms(text_lower, self.lexicons["attribute_terms"])

        laterality = set()
        for lat in ["left", "right", "bilateral", "upper", "lower", "central", "peripheral"]:
            if self._contains_term(text_lower, lat):
                laterality.add(lat)

        polarity = "present"
        if any(self._contains_term(text_lower, neg) for neg in self.lexicons["negation_terms"]):
            polarity = "absent_or_negated"

        uncertainty = "certain"
        if any(self._contains_term(text_lower, u) for u in self.lexicons["uncertain_terms"]):
            uncertainty = "uncertain"

        return {
            "anatomy": anatomy,
            "findings": findings,
            "attributes": attributes,
            "laterality": laterality,
            "polarity": polarity,
            "uncertainty": uncertainty
        }

    def _compare_medical_facts(self, pred_text: str, ref_text: str) -> Dict[str, float]:
        pred_facts = self._extract_medical_facts(pred_text)
        ref_facts = self._extract_medical_facts(ref_text)

        anatomy_f1 = self._set_f1(pred_facts["anatomy"], ref_facts["anatomy"])
        finding_f1 = self._set_f1(pred_facts["findings"], ref_facts["findings"])
        attribute_f1 = self._set_f1(pred_facts["attributes"], ref_facts["attributes"])
        laterality_f1 = self._set_f1(pred_facts["laterality"], ref_facts["laterality"])

        polarity_match = 1.0 if pred_facts["polarity"] == ref_facts["polarity"] else 0.0
        uncertainty_match = 1.0 if pred_facts["uncertainty"] == ref_facts["uncertainty"] else 0.0

        medical_fact_score = (
            0.30 * finding_f1 +
            0.20 * anatomy_f1 +
            0.15 * attribute_f1 +
            0.15 * laterality_f1 +
            0.10 * polarity_match +
            0.10 * uncertainty_match
        )

        return {
            "anatomy_f1": anatomy_f1,
            "finding_f1": finding_f1,
            "attribute_f1": attribute_f1,
            "laterality_f1": laterality_f1,
            "polarity_match": polarity_match,
            "uncertainty_match": uncertainty_match,
            "medical_fact_score": medical_fact_score
        }

    # =========================
    # 空间描述 / 空间关系
    # =========================
    def _extract_spatial_facts(self, text: str) -> Dict[str, Set[str]]:
        text_lower = text.lower()

        spatial_positions = self._extract_terms(text_lower, self.lexicons["spatial_terms"])
        spatial_relations = self._extract_terms(text_lower, self.lexicons["spatial_relation_terms"])

        if "centrally" in text_lower:
            spatial_positions.add("central")
        if "upper region" in text_lower:
            spatial_positions.add("upper")
        if "lower region" in text_lower:
            spatial_positions.add("lower")

        return {
            "spatial_positions": spatial_positions,
            "spatial_relations": spatial_relations
        }

    def _compare_spatial_facts(self, pred_text: str, ref_text: str) -> Dict[str, float]:
        pred_spatial = self._extract_spatial_facts(pred_text)
        ref_spatial = self._extract_spatial_facts(ref_text)

        spatial_description_f1 = self._set_f1(
            pred_spatial["spatial_positions"],
            ref_spatial["spatial_positions"]
        )

        spatial_relation_f1 = self._set_f1(
            pred_spatial["spatial_relations"],
            ref_spatial["spatial_relations"]
        )

        spatial_fact_score = 0.6 * spatial_description_f1 + 0.4 * spatial_relation_f1

        return {
            "spatial_description_f1": spatial_description_f1,
            "spatial_relation_f1": spatial_relation_f1,
            "spatial_fact_score": spatial_fact_score
        }

    # =========================
    # 风格指标
    # =========================
    def _evaluate_clinical_reasoning_style(self, text: str) -> float:
        score = 0.0
        text_lower = text.lower()

        reasoning_count = sum(
            1 for keyword in self.clinical_reasoning_keywords
            if self._contains_term(text_lower, keyword)
        )
        score += min(reasoning_count * 0.08, 0.30)

        cautious_phrases = ["may", "could", "possible", "likely", "suggest", "consistent with"]
        if any(self._contains_term(text_lower, phrase) for phrase in cautious_phrases):
            score += 0.25

        follow_up_phrases = ["follow-up", "correlation", "recommend", "consider", "further"]
        if any(self._contains_term(text_lower, phrase) for phrase in follow_up_phrases):
            score += 0.20

        structure_words = ["findings", "impression", "conclusion", "assessment"]
        if any(self._contains_term(text_lower, word) for word in structure_words):
            score += 0.25

        return min(score, 1.0)

    def _evaluate_diagnostic_reporting_style(self, text: str) -> float:
        score = 0.0
        text_lower = text.lower()

        core_terms = (
            self.lexicons["finding_terms"][:20] +
            self.lexicons["anatomy_terms"][:20] +
            self.lexicons["attribute_terms"][:20]
        )

        term_count = sum(1 for term in core_terms if self._contains_term(text_lower, term))
        score += min(term_count * 0.04, 0.35)

        structure_words = ["findings", "impression", "conclusion", "assessment"]
        if any(self._contains_term(text_lower, word) for word in structure_words):
            score += 0.25

        overconfident_words = ["definitely", "certainly", "absolutely", "without doubt"]
        if not any(self._contains_term(text_lower, word) for word in overconfident_words):
            score += 0.20

        if len(text.split()) >= 8:
            score += 0.20

        return min(score, 1.0)

    # =========================
    # 证据支持
    # =========================
    def _evaluate_evidence_support(self, pred_text: str, evidence_text: str) -> Dict[str, float]:
        if not evidence_text.strip():
            return {
                "finding_evidence_support": 0.0,
                "anatomy_evidence_support": 0.0,
                "attribute_evidence_support": 0.0,
                "spatial_evidence_support": 0.0,
                "evidence_support_score": 0.0,
                "evidence_polarity_contradiction": 0.0
            }

        pred_facts = self._extract_medical_facts(pred_text)
        evidence_facts = self._extract_medical_facts(evidence_text)

        pred_spatial = self._extract_spatial_facts(pred_text)
        evidence_spatial = self._extract_spatial_facts(evidence_text)

        finding_support = self._set_f1(pred_facts["findings"], evidence_facts["findings"])
        anatomy_support = self._set_f1(pred_facts["anatomy"], evidence_facts["anatomy"])
        attribute_support = self._set_f1(pred_facts["attributes"], evidence_facts["attributes"])

        spatial_support = 0.5 * self._set_f1(
            pred_spatial["spatial_positions"], evidence_spatial["spatial_positions"]
        ) + 0.5 * self._set_f1(
            pred_spatial["spatial_relations"], evidence_spatial["spatial_relations"]
        )

        polarity_contradiction = 0.0
        if pred_facts["polarity"] != evidence_facts["polarity"]:
            if pred_facts["findings"] and evidence_facts["findings"]:
                if len(pred_facts["findings"] & evidence_facts["findings"]) > 0:
                    polarity_contradiction = 1.0

        support_score = (
            0.40 * finding_support +
            0.20 * anatomy_support +
            0.15 * attribute_support +
            0.25 * spatial_support
        )

        return {
            "finding_evidence_support": finding_support,
            "anatomy_evidence_support": anatomy_support,
            "attribute_evidence_support": attribute_support,
            "spatial_evidence_support": spatial_support,
            "evidence_support_score": support_score,
            "evidence_polarity_contradiction": polarity_contradiction
        }

    # =========================
    # 聚合工具
    # =========================
    def _aggregate_case_metrics(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cases:
            return {"count": 0}

        keys = [
            "term_coverage",
            "term_precision",
            "clinical_reasoning_style_score",
            "diagnostic_reporting_style_score",
            "term_level_answer_evidence_overlap",
            "anatomy_f1",
            "finding_f1",
            "attribute_f1",
            "laterality_f1",
            "polarity_match",
            "uncertainty_match",
            "medical_fact_score",
            "spatial_description_f1",
            "spatial_relation_f1",
            "spatial_fact_score",
            "finding_evidence_support",
            "anatomy_evidence_support",
            "attribute_evidence_support",
            "spatial_evidence_support",
            "evidence_support_score",
            "evidence_polarity_contradiction"
        ]

        result = {"count": len(cases)}
        for k in keys:
            vals = [c[k] for c in cases if k in c]
            result[f"avg_{k}"] = float(np.mean(vals)) if vals else 0.0

        evidence_nonempty_vals = [1.0 if (c.get("evidence_summary", "").strip()) else 0.0 for c in cases]
        result["avg_evidence_nonempty_rate"] = float(np.mean(evidence_nonempty_vals)) if evidence_nonempty_vals else 0.0
        return result

    # =========================
    # 核心评估
    # =========================
    def evaluate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        predictions = [p for p in predictions if p.get("prediction", "").strip()]

        if not predictions:
            return {
                "sample_count": 0,
                "metrics": {},
                "cases": [],
                "type_breakdown": {},
                "modality_breakdown": {}
            }

        term_coverage_scores = []
        term_precision_scores = []
        clinical_reasoning_style_scores = []
        diagnostic_reporting_style_scores = []

        evidence_nonempty_scores = []
        term_level_answer_evidence_overlap_scores = []

        anatomy_f1_scores = []
        finding_f1_scores = []
        attribute_f1_scores = []
        laterality_f1_scores = []
        polarity_match_scores = []
        uncertainty_match_scores = []
        medical_fact_scores = []

        spatial_description_f1_scores = []
        spatial_relation_f1_scores = []
        spatial_fact_scores = []

        evidence_support_scores = []
        finding_evidence_support_scores = []
        spatial_evidence_support_scores = []
        evidence_polarity_contradiction_scores = []

        per_case_results = []
        type_groups = {}
        modality_groups = {}

        for pred in predictions:
            pred_text = pred["prediction"].lower()
            ref_text = pred["reference"].lower()
            evidence_text = (pred.get("evidence_summary") or "").lower()

            question_type = self._classify_question_type(pred["question"], pred["modality"])
            modality = pred["modality"] or "Unknown"

            pred_terms = self._extract_terms(pred_text, self.medical_terms)
            ref_terms = self._extract_terms(ref_text, self.medical_terms)

            coverage = len(pred_terms & ref_terms) / len(ref_terms) if ref_terms else 0.0
            precision = len(pred_terms & ref_terms) / len(pred_terms) if pred_terms else 0.0

            clinical_style = self._evaluate_clinical_reasoning_style(pred_text)
            reporting_style = self._evaluate_diagnostic_reporting_style(pred_text)

            evidence_nonempty = 1.0 if evidence_text.strip() else 0.0

            term_level_overlap = 0.0
            if evidence_text.strip():
                evidence_terms = self._extract_terms(evidence_text, self.medical_terms)
                term_level_overlap = len(pred_terms & evidence_terms) / len(pred_terms) if pred_terms else 0.0

            fact_metrics = self._compare_medical_facts(pred_text, ref_text)
            spatial_metrics = self._compare_spatial_facts(pred_text, ref_text)
            ev_metrics = self._evaluate_evidence_support(pred_text, evidence_text)

            term_coverage_scores.append(coverage)
            term_precision_scores.append(precision)
            clinical_reasoning_style_scores.append(clinical_style)
            diagnostic_reporting_style_scores.append(reporting_style)

            evidence_nonempty_scores.append(evidence_nonempty)
            term_level_answer_evidence_overlap_scores.append(term_level_overlap)

            anatomy_f1_scores.append(fact_metrics["anatomy_f1"])
            finding_f1_scores.append(fact_metrics["finding_f1"])
            attribute_f1_scores.append(fact_metrics["attribute_f1"])
            laterality_f1_scores.append(fact_metrics["laterality_f1"])
            polarity_match_scores.append(fact_metrics["polarity_match"])
            uncertainty_match_scores.append(fact_metrics["uncertainty_match"])
            medical_fact_scores.append(fact_metrics["medical_fact_score"])

            spatial_description_f1_scores.append(spatial_metrics["spatial_description_f1"])
            spatial_relation_f1_scores.append(spatial_metrics["spatial_relation_f1"])
            spatial_fact_scores.append(spatial_metrics["spatial_fact_score"])

            evidence_support_scores.append(ev_metrics["evidence_support_score"])
            finding_evidence_support_scores.append(ev_metrics["finding_evidence_support"])
            spatial_evidence_support_scores.append(ev_metrics["spatial_evidence_support"])
            evidence_polarity_contradiction_scores.append(ev_metrics["evidence_polarity_contradiction"])

            case_result = {
                "id": pred["id"],
                "question": pred["question"],
                "question_type": question_type,
                "modality": modality,
                "body_part": pred["body_part"],
                "prediction": pred["prediction"],
                "reference": pred["reference"],
                "evidence_summary": pred.get("evidence_summary", ""),
                "term_coverage": coverage,
                "term_precision": precision,
                "clinical_reasoning_style_score": clinical_style,
                "diagnostic_reporting_style_score": reporting_style,
                "term_level_answer_evidence_overlap": term_level_overlap,
                **fact_metrics,
                **spatial_metrics,
                **ev_metrics
            }
            per_case_results.append(case_result)

            type_groups.setdefault(question_type, []).append(case_result)
            modality_groups.setdefault(modality, []).append(case_result)

        metrics = {
            "avg_medical_term_coverage": float(np.mean(term_coverage_scores)),
            "avg_medical_term_precision": float(np.mean(term_precision_scores)),
            "avg_clinical_reasoning_style_score": float(np.mean(clinical_reasoning_style_scores)),
            "avg_diagnostic_reporting_style_score": float(np.mean(diagnostic_reporting_style_scores)),
            "avg_evidence_nonempty_rate": float(np.mean(evidence_nonempty_scores)),
            "avg_term_level_answer_evidence_overlap": float(np.mean(term_level_answer_evidence_overlap_scores)),
            "avg_anatomy_f1": float(np.mean(anatomy_f1_scores)),
            "avg_finding_f1": float(np.mean(finding_f1_scores)),
            "avg_attribute_f1": float(np.mean(attribute_f1_scores)),
            "avg_laterality_f1": float(np.mean(laterality_f1_scores)),
            "avg_polarity_match": float(np.mean(polarity_match_scores)),
            "avg_uncertainty_match": float(np.mean(uncertainty_match_scores)),
            "avg_medical_fact_score": float(np.mean(medical_fact_scores)),
            "avg_spatial_description_f1": float(np.mean(spatial_description_f1_scores)),
            "avg_spatial_relation_f1": float(np.mean(spatial_relation_f1_scores)),
            "avg_spatial_fact_score": float(np.mean(spatial_fact_scores)),
            "avg_evidence_support_score": float(np.mean(evidence_support_scores)),
            "avg_finding_evidence_support": float(np.mean(finding_evidence_support_scores)),
            "avg_spatial_evidence_support": float(np.mean(spatial_evidence_support_scores)),
            "avg_evidence_polarity_contradiction": float(np.mean(evidence_polarity_contradiction_scores)),
        }

        type_breakdown = {
            qtype: self._aggregate_case_metrics(cases)
            for qtype, cases in type_groups.items()
        }
        modality_breakdown = {
            mod: self._aggregate_case_metrics(cases)
            for mod, cases in modality_groups.items()
        }

        return {
            "sample_count": len(predictions),
            "metrics": metrics,
            "cases": per_case_results,
            "type_breakdown": type_breakdown,
            "modality_breakdown": modality_breakdown
        }

    # =========================
    # 单文件 / 多文件评估
    # =========================
    def evaluate_prediction_file(self, pred_file: str, experiment_name: str = None) -> Dict[str, Any]:
        raw_predictions = self.load_predictions_file(pred_file)
        predictions = self.normalize_predictions(raw_predictions)
        results = self.evaluate_predictions(predictions)

        exp_name = experiment_name or Path(pred_file).stem
        exp_dir = os.path.join(self.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results["metrics"], f, indent=2, ensure_ascii=False)

        with open(os.path.join(exp_dir, "per_case_results.json"), "w", encoding="utf-8") as f:
            json.dump(results["cases"], f, indent=2, ensure_ascii=False)

        with open(os.path.join(exp_dir, "type_breakdown.json"), "w", encoding="utf-8") as f:
            json.dump(results["type_breakdown"], f, indent=2, ensure_ascii=False)

        with open(os.path.join(exp_dir, "modality_breakdown.json"), "w", encoding="utf-8") as f:
            json.dump(results["modality_breakdown"], f, indent=2, ensure_ascii=False)

        error_cases = self._select_error_cases(results["cases"])
        with open(os.path.join(exp_dir, "error_cases.json"), "w", encoding="utf-8") as f:
            json.dump(error_cases, f, indent=2, ensure_ascii=False)

        summary = {
            "experiment_name": exp_name,
            "prediction_file": pred_file,
            "sample_count": results["sample_count"],
            "metrics": results["metrics"]
        }

        with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Finished evaluating: {exp_name} ({results['sample_count']} samples)")
        return {
            "summary": summary,
            "cases": results["cases"],
            "type_breakdown": results["type_breakdown"],
            "modality_breakdown": results["modality_breakdown"]
        }

    def evaluate_multiple_prediction_files(self, pred_files: List[str]) -> Dict[str, Any]:
        experiment_results = {}

        for pred_file in pred_files:
            exp_name = Path(pred_file).stem
            res = self.evaluate_prediction_file(pred_file, experiment_name=exp_name)
            experiment_results[exp_name] = res

        summaries = [experiment_results[name]["summary"] for name in experiment_results]

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "experiments": summaries
        }

        with open(os.path.join(self.output_dir, "comparison.json"), "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        self._write_comparison_csv(summaries)
        self._write_comparison_report(summaries)

        self._write_breakdown_all(
            experiment_results,
            filename="question_type_breakdown_all.json",
            breakdown_key="type_breakdown"
        )
        self._write_breakdown_all(
            experiment_results,
            filename="modality_breakdown_all.json",
            breakdown_key="modality_breakdown"
        )

        baseline_name = self._find_baseline_name(experiment_results.keys())
        if baseline_name:
            self._run_delta_vs_baseline(experiment_results, baseline_name)
        else:
            print("Warning: baseline experiment not found, delta-vs-baseline analysis skipped.")

        return {
            "comparison": comparison,
            "experiment_results": experiment_results
        }

    # =========================
    # 输出报告
    # =========================
    def _write_comparison_csv(self, summaries: List[Dict[str, Any]]):
        csv_file = os.path.join(self.output_dir, "comparison.csv")
        metric_keys = [
            "avg_medical_term_coverage",
            "avg_medical_term_precision",
            "avg_clinical_reasoning_style_score",
            "avg_diagnostic_reporting_style_score",
            "avg_anatomy_f1",
            "avg_finding_f1",
            "avg_attribute_f1",
            "avg_laterality_f1",
            "avg_polarity_match",
            "avg_uncertainty_match",
            "avg_medical_fact_score",
            "avg_spatial_description_f1",
            "avg_spatial_relation_f1",
            "avg_spatial_fact_score",
            "avg_evidence_nonempty_rate",
            "avg_term_level_answer_evidence_overlap",
            "avg_evidence_support_score",
            "avg_finding_evidence_support",
            "avg_spatial_evidence_support",
            "avg_evidence_polarity_contradiction"
        ]

        with open(csv_file, "w", encoding="utf-8") as f:
            header = ["experiment_name", "sample_count"] + metric_keys
            f.write(",".join(header) + "\n")

            for s in summaries:
                row = [s["experiment_name"], str(s["sample_count"])]
                for k in metric_keys:
                    row.append(str(s["metrics"].get(k, 0.0)))
                f.write(",".join(row) + "\n")

        print(f"Saved comparison CSV: {csv_file}")

    def _write_comparison_report(self, summaries: List[Dict[str, Any]]):
        report_file = os.path.join(self.output_dir, "comparison_report.txt")

        lines = []
        lines.append("=" * 140)
        lines.append("Medical Prediction Evaluation Comparison Report")
        lines.append("=" * 140)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        header = (
            f"{'Experiment':<32}"
            f"{'Samples':<8}"
            f"{'FactScore':<12}"
            f"{'Spatial':<12}"
            f"{'FindingF1':<12}"
            f"{'AttributeF1':<12}"
            f"{'LateralityF1':<14}"
            f"{'Uncertainty':<12}"
            # f"{'AnatomyF1':<12}"
            # f"{'Polarity':<10}"
            # f"{'EvSupport':<12}"
            # f"{'EvSpatial':<12}"
            # f"{'EvContra':<10}"
        )
        lines.append(header)
        lines.append("-" * 140)

        for s in summaries:
            m = s["metrics"]
            row = (
                f"{s['experiment_name']:<32}"
                f"{s['sample_count']:<8}"
                f"{m.get('avg_medical_fact_score', 0):<12.4f}"
                f"{m.get('avg_spatial_fact_score', 0):<12.4f}"
                f"{m.get('avg_finding_f1', 0):<12.4f}"
                f"{m.get('avg_attribute_f1', 0):<12.4f}"
                f"{m.get('avg_laterality_f1', 0):<14.4f}"
                f"{m.get('avg_uncertainty_match', 0):<12.4f}"
                # f"{m.get('avg_anatomy_f1', 0):<12.4f}"
                # f"{m.get('avg_polarity_match', 0):<10.4f}"
                # f"{m.get('avg_evidence_support_score', 0):<12.4f}"
                # f"{m.get('avg_spatial_evidence_support', 0):<12.4f}"
                # f"{m.get('avg_evidence_polarity_contradiction', 0):<10.4f}"
            )
            lines.append(row)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Saved comparison report: {report_file}")

    def _write_breakdown_all(self, experiment_results: Dict[str, Any], filename: str, breakdown_key: str):
        out = {}
        for exp_name, exp_res in experiment_results.items():
            out[exp_name] = exp_res[breakdown_key]

        out_file = os.path.join(self.output_dir, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"Saved {breakdown_key} all: {out_file}")

    # =========================
    # baseline delta 分析
    # =========================
    def _find_baseline_name(self, experiment_names) -> str:
        names = list(experiment_names)
        exact = [n for n in names if n == "baseline_predictions"]
        if exact:
            return exact[0]

        contains = [n for n in names if "baseline" in n.lower()]
        if contains:
            return contains[0]
        return ""

    def _run_delta_vs_baseline(self, experiment_results: Dict[str, Any], baseline_name: str):
        baseline_cases = experiment_results[baseline_name]["cases"]
        baseline_by_id = {c["id"]: c for c in baseline_cases}

        delta_summary = {
            "baseline_name": baseline_name,
            "generated_at": datetime.now().isoformat(),
            "experiments": {}
        }

        report_lines = []
        report_lines.append("=" * 140)
        report_lines.append("Delta vs Baseline Report")
        report_lines.append("=" * 140)
        report_lines.append(f"Baseline: {baseline_name}")
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        header = (
            f"{'Experiment':<32}"
            f"{'Overlap':<10}"
            f"{'FactΔ':<12}"
            f"{'SpatialΔ':<12}"
            f"{'EvSupportΔ':<12}"
            f"{'Improve(Fact)':<15}"
            f"{'Degrade(Fact)':<15}"
        )
        report_lines.append(header)
        report_lines.append("-" * 140)

        for exp_name, exp_res in experiment_results.items():
            if exp_name == baseline_name:
                continue

            current_cases = exp_res["cases"]
            delta_result = self._compute_delta_against_baseline(current_cases, baseline_by_id)

            delta_summary["experiments"][exp_name] = delta_result["summary"]

            row = (
                f"{exp_name:<32}"
                f"{delta_result['summary']['overlap_count']:<10}"
                f"{delta_result['summary']['avg_delta_medical_fact_score']:<12.4f}"
                f"{delta_result['summary']['avg_delta_spatial_fact_score']:<12.4f}"
                f"{delta_result['summary']['avg_delta_evidence_support_score']:<12.4f}"
                f"{delta_result['summary']['improved_medical_fact_count']:<15}"
                f"{delta_result['summary']['degraded_medical_fact_count']:<15}"
            )
            report_lines.append(row)

            exp_dir = os.path.join(self.output_dir, exp_name)
            with open(os.path.join(exp_dir, "delta_vs_baseline_cases.json"), "w", encoding="utf-8") as f:
                json.dump(delta_result["cases"], f, indent=2, ensure_ascii=False)

            with open(os.path.join(exp_dir, "delta_vs_baseline_summary.json"), "w", encoding="utf-8") as f:
                json.dump(delta_result["summary"], f, indent=2, ensure_ascii=False)

        summary_file = os.path.join(self.output_dir, "delta_vs_baseline_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(delta_summary, f, indent=2, ensure_ascii=False)

        report_file = os.path.join(self.output_dir, "delta_vs_baseline_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Saved delta summary: {summary_file}")
        print(f"Saved delta report: {report_file}")

    def _compute_delta_against_baseline(self, current_cases: List[Dict[str, Any]], baseline_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        overlap_cases = []

        for case in current_cases:
            case_id = case["id"]
            if case_id not in baseline_by_id:
                continue

            base = baseline_by_id[case_id]
            delta_case = {
                "id": case_id,
                "question": case["question"],
                "question_type": case["question_type"],
                "modality": case["modality"],
                "body_part": case["body_part"],

                "baseline_medical_fact_score": base["medical_fact_score"],
                "current_medical_fact_score": case["medical_fact_score"],
                "delta_medical_fact_score": case["medical_fact_score"] - base["medical_fact_score"],

                "baseline_spatial_fact_score": base["spatial_fact_score"],
                "current_spatial_fact_score": case["spatial_fact_score"],
                "delta_spatial_fact_score": case["spatial_fact_score"] - base["spatial_fact_score"],

                "baseline_evidence_support_score": base["evidence_support_score"],
                "current_evidence_support_score": case["evidence_support_score"],
                "delta_evidence_support_score": case["evidence_support_score"] - base["evidence_support_score"],

                "baseline_finding_f1": base["finding_f1"],
                "current_finding_f1": case["finding_f1"],
                "delta_finding_f1": case["finding_f1"] - base["finding_f1"],

                "baseline_anatomy_f1": base["anatomy_f1"],
                "current_anatomy_f1": case["anatomy_f1"],
                "delta_anatomy_f1": case["anatomy_f1"] - base["anatomy_f1"],

                "baseline_polarity_match": base["polarity_match"],
                "current_polarity_match": case["polarity_match"],
                "delta_polarity_match": case["polarity_match"] - base["polarity_match"],

                "baseline_prediction": base["prediction"],
                "current_prediction": case["prediction"],
                "reference": case["reference"],
                "current_evidence_summary": case.get("evidence_summary", "")
            }
            overlap_cases.append(delta_case)

        if not overlap_cases:
            return {"summary": {"overlap_count": 0}, "cases": []}

        def _avg(key):
            return float(np.mean([c[key] for c in overlap_cases])) if overlap_cases else 0.0

        improved_fact = [c for c in overlap_cases if c["delta_medical_fact_score"] > 1e-8]
        degraded_fact = [c for c in overlap_cases if c["delta_medical_fact_score"] < -1e-8]
        tied_fact = [c for c in overlap_cases if abs(c["delta_medical_fact_score"]) <= 1e-8]

        improved_spatial = [c for c in overlap_cases if c["delta_spatial_fact_score"] > 1e-8]
        degraded_spatial = [c for c in overlap_cases if c["delta_spatial_fact_score"] < -1e-8]

        by_question_type = {}
        for qtype in sorted(set(c["question_type"] for c in overlap_cases)):
            group = [c for c in overlap_cases if c["question_type"] == qtype]
            by_question_type[qtype] = {
                "count": len(group),
                "avg_delta_medical_fact_score": float(np.mean([x["delta_medical_fact_score"] for x in group])) if group else 0.0,
                "avg_delta_spatial_fact_score": float(np.mean([x["delta_spatial_fact_score"] for x in group])) if group else 0.0,
                "avg_delta_evidence_support_score": float(np.mean([x["delta_evidence_support_score"] for x in group])) if group else 0.0,
            }

        by_modality = {}
        for modality in sorted(set(c["modality"] for c in overlap_cases)):
            group = [c for c in overlap_cases if c["modality"] == modality]
            by_modality[modality] = {
                "count": len(group),
                "avg_delta_medical_fact_score": float(np.mean([x["delta_medical_fact_score"] for x in group])) if group else 0.0,
                "avg_delta_spatial_fact_score": float(np.mean([x["delta_spatial_fact_score"] for x in group])) if group else 0.0,
                "avg_delta_evidence_support_score": float(np.mean([x["delta_evidence_support_score"] for x in group])) if group else 0.0,
            }

        summary = {
            "overlap_count": len(overlap_cases),

            "avg_delta_medical_fact_score": _avg("delta_medical_fact_score"),
            "avg_delta_spatial_fact_score": _avg("delta_spatial_fact_score"),
            "avg_delta_evidence_support_score": _avg("delta_evidence_support_score"),
            "avg_delta_finding_f1": _avg("delta_finding_f1"),
            "avg_delta_anatomy_f1": _avg("delta_anatomy_f1"),
            "avg_delta_polarity_match": _avg("delta_polarity_match"),

            "improved_medical_fact_count": len(improved_fact),
            "degraded_medical_fact_count": len(degraded_fact),
            "tied_medical_fact_count": len(tied_fact),

            "improved_spatial_fact_count": len(improved_spatial),
            "degraded_spatial_fact_count": len(degraded_spatial),

            "question_type_delta_breakdown": by_question_type,
            "modality_delta_breakdown": by_modality,

            "top_10_improved_fact_ids": [c["id"] for c in sorted(overlap_cases, key=lambda x: x["delta_medical_fact_score"], reverse=True)[:10]],
            "top_10_degraded_fact_ids": [c["id"] for c in sorted(overlap_cases, key=lambda x: x["delta_medical_fact_score"])[:10]],
        }

        sorted_cases = sorted(overlap_cases, key=lambda x: x["delta_medical_fact_score"])

        cases_out = {
            "all_cases": overlap_cases,
            "top_20_improved_fact_cases": sorted(overlap_cases, key=lambda x: x["delta_medical_fact_score"], reverse=True)[:20],
            "top_20_degraded_fact_cases": sorted_cases[:20],
            "top_20_improved_spatial_cases": sorted(overlap_cases, key=lambda x: x["delta_spatial_fact_score"], reverse=True)[:20],
            "top_20_degraded_spatial_cases": sorted(overlap_cases, key=lambda x: x["delta_spatial_fact_score"])[:20],
        }

        return {"summary": summary, "cases": cases_out}

    # =========================
    # 误差分析样本选择
    # =========================
    def _select_error_cases(self, cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        sorted_by_fact = sorted(cases, key=lambda x: x.get("medical_fact_score", 0.0))
        sorted_by_spatial = sorted(cases, key=lambda x: x.get("spatial_fact_score", 0.0))
        sorted_by_contradiction = sorted(
            cases, key=lambda x: x.get("evidence_polarity_contradiction", 0.0), reverse=True
        )
        sorted_by_support = sorted(cases, key=lambda x: x.get("evidence_support_score", 0.0))

        low_fact_cases = sorted_by_fact[:20]
        low_spatial_cases = sorted_by_spatial[:20]
        contradiction_cases = [c for c in sorted_by_contradiction if c.get("evidence_polarity_contradiction", 0.0) > 0][:20]
        low_support_cases = sorted_by_support[:20]

        return {
            "low_medical_fact_score_cases": low_fact_cases,
            "low_spatial_fact_score_cases": low_spatial_cases,
            "evidence_contradiction_cases": contradiction_cases,
            "low_evidence_support_cases": low_support_cases
        }


def main():
    # ===== 改成你的实际路径 =====
    DATASET_FILE = r"D:\weng\llama-factory\data\PubMedVision_VALIDATION.json"

    PREDICTION_FILES = [
        r"D:\weng\llama-factory\500\pubmedvision_candidate_selector_v1\baseline_predictions.jsonl",
        r"D:\weng\llama-factory\500\pubmedvision_candidate_selector_v1\multi_candidate_first_predictions.jsonl",
        # r"D:\weng\llama-factory\pubmedvision_candidate_selector_v1\multi_candidate_heuristic_selector_predictions.jsonl",
        r"D:\weng\llama-factory\500\pubmedvision_candidate_selector_v1\multi_candidate_llm_selector_predictions.jsonl",
        r"D:\weng\llama-factory\500\pubmedvision_candidate_selector_v1\multi_candidate_oracle_predictions.jsonl",
        # r"D:\weng\llama-factory\pubmedvision_qtype_selector_results\qtype_multi_candidate_selector_predictions.jsonl",
        # r"D:\weng\llama-factory\pubmedvision_qtype_selector_results\qtype_multi_candidate_first_predictions.jsonl",
        # r"D:\weng\llama-factory\pubmedvision_qtype_selector_results\qtype_multi_candidate_oracle_predictions.jsonl",

         r"D:\weng\llama-factory\500\distilled_reranker_predictions\distilled_reranker_predictions_predictions.jsonl",
     ]

    OUTPUT_DIR = r"./evaluation_reranker"

    evaluator = MedicalPredictionEvaluator(
        dataset_file=DATASET_FILE,
        output_dir=OUTPUT_DIR
    )

    evaluator.evaluate_multiple_prediction_files(PREDICTION_FILES)
    print("All evaluations completed.")


if __name__ == "__main__":
    main()