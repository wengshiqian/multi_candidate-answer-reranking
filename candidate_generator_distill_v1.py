import os
import re
import json
import torch
from typing import List, Dict, TypedDict, Set
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


class MedicalSampleState(TypedDict):
    sample_id: str
    image_path: str
    question: str
    modality: str
    body_part: str
    candidate_answers: List[str]
    metadata: dict


class DistillCandidateGenerator:
    """
    用于：
    1) 生成多候选答案
    2) 训练集上用 reference 计算 oracle-style 质量分
    """

    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        print("=" * 80)
        print("🚀 初始化 DistillCandidateGenerator")
        print("=" * 80)

        print(f"📦 加载基础模型: {base_model_path}")
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"🔧 加载 LoRA: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        self.model.eval()

        self.lexicons = self._build_lexicons()

        print("✅ 初始化完成")
        print("=" * 80)

    # =========================================================
    # 基础生成
    # =========================================================
    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 220,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def _clean_answer(self, text: str, anchor: str = "Answer:") -> str:
        out = (text or "").strip()

        if anchor in out:
            out = out.split(anchor)[-1].strip()
        elif "assistant" in out.lower():
            out = out.split("assistant")[-1].strip()

        out = re.sub(
            r"^\s*(assistant|doctor|student|clinician|radiologist)\s*:??\s+",
            "",
            out,
            flags=re.IGNORECASE
        )
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _postprocess(self, text: str) -> str:
        out = (text or "").strip()
        if not out:
            return ""

        repls = [
            (r"\btumour\b", "tumor"),
            (r"\bhaemorrhage\b", "hemorrhage"),
            (r"\boedema\b", "edema"),
            (r"\bischaemia\b", "ischemia"),
            (r"\bmetastases\b", "metastasis"),
        ]
        for pat, rep in repls:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)

        out = re.sub(r"\s+", " ", out).strip()
        return out

    # =========================================================
    # 题型识别
    # =========================================================
    def classify_question_type(self, question: str) -> str:
        q = question.lower().strip()

        if any(k in q for k in ["where", "located", "location", "tip"]):
            return "location"

        if any(k in q for k in ["size", "measure", "diameter", "how large", "approximate size"]):
            return "size"

        if any(k in q for k in ["compare", "baseline", "progression", "response", "therapy", "uptake", "new metastasis", "new site"]):
            return "comparison"

        return "description"

    # =========================================================
    # 多候选 prompt
    # =========================================================
    def _build_candidate_prompts(self, question: str, qtype: str) -> List[str]:
        if qtype == "location":
            return [
                f"""You are a medical image QA assistant.

Question:
{question}

Instructions:
- Answer the location directly.
- Focus on anatomical level, cavity, or compartment.
- Do not add likely diagnosis unless necessary.
- Keep it short and precise.
- One sentence preferred.

Answer:""",
                f"""You are a careful medical image QA assistant.

Question:
{question}

Instructions:
- Give the most specific visible location.
- Mention level or compartment if visible.
- Be cautious if exact location is uncertain.
- Do not add symptoms or treatment suggestions.
- One sentence preferred.

Answer:""",
                f"""You are a concise radiology assistant.

Question:
{question}

Instructions:
- State only the anatomical location being asked.
- Avoid pathology speculation.
- Prefer the shortest correct answer.

Answer:""",
                f"""You are a grounded medical image assistant.

Question:
{question}

Instructions:
- Answer the location directly and conservatively.
- If the image suggests abnormal positioning, state that location rather than a routine normal placement.
- Keep it brief.

Answer:"""
            ]

        if qtype == "size":
            return [
                f"""You are a medical image QA assistant.

Question:
{question}

Instructions:
- Answer only the approximate size.
- Prefer a number with unit if possible.
- Do not add extra diagnosis or procedure details.
- One sentence preferred.

Answer:""",
                f"""You are a concise radiology assistant.

Question:
{question}

Instructions:
- State the approximate size directly.
- If two dimensions are visible, include both.
- Avoid guessing unrelated anatomy or pathology.
- One sentence preferred.

Answer:""",
                f"""You are a careful medical image QA assistant.

Question:
{question}

Instructions:
- Give a cautious approximate size estimate.
- If exact measurement is not certain, use approximately.
- Do not add clinical speculation.
- One sentence preferred.

Answer:""",
                f"""You are a grounded medical image assistant.

Question:
{question}

Instructions:
- Focus only on the mass size.
- Keep the answer short.
- No extra location or benign/malignant interpretation unless explicitly asked.

Answer:"""
            ]

        if qtype == "comparison":
            return [
                f"""You are a medical image QA assistant.

Question:
{question}

Instructions:
- Focus on comparison with baseline or treatment response.
- State clearly whether findings improved, worsened, progressed, or remained stable.
- Avoid unsupported symptoms or extra history.
- Keep it concise, 2-3 sentences.

Answer:""",
                f"""You are a careful medical image QA assistant.

Question:
{question}

Instructions:
- Prioritize the direction of change compared with baseline.
- Be careful not to reverse progression and improvement.
- Mention the main comparison finding only.
- Keep it concise.

Answer:""",
                f"""You are a concise radiology assistant.

Question:
{question}

Instructions:
- Answer with the key comparison conclusion.
- Mention baseline if relevant.
- Do not add extra pathology not needed for the comparison.
- Keep it short.

Answer:""",
                f"""You are a grounded medical image assistant.

Question:
{question}

Instructions:
- Focus on disease burden change and treatment response.
- Avoid overclaiming treatment success unless clearly visible.
- Keep the answer compact.

Answer:"""
            ]

        # description
        return [
            f"""You are a medical image QA assistant.

Question:
{question}

Instructions:
- Describe the key visible findings relevant to the question.
- Use concise professional language.
- Avoid unsupported severe pathology claims.
- Keep the answer in 2-4 sentences.

Answer:""",
            f"""You are a careful medical image QA assistant.

Question:
{question}

Instructions:
- Give a cautious interpretation of the visible findings.
- Mention important structures or abnormalities only if relevant.
- Avoid adding symptoms or clinical history not stated in the question.
- Keep the answer concise.

Answer:""",
            f"""You are a concise radiology assistant.

Question:
{question}

Instructions:
- Answer directly and briefly.
- Do not over-explain.
- Avoid hallucinated severe abnormalities.

Answer:""",
            f"""You are a grounded medical image assistant.

Question:
{question}

Instructions:
- Provide a balanced answer with visible findings and a cautious interpretation.
- If uncertain, state uncertainty briefly.
- Keep the answer concise.

Answer:"""
        ]

    def generate_candidates_for_question(
        self,
        image_path: str,
        question: str,
        modality: str = "Unknown",
        body_part: str = "Unknown",
    ) -> Dict:
        image = Image.open(image_path).convert("RGB")
        qtype = self.classify_question_type(question)
        prompts = self._build_candidate_prompts(question, qtype)

        answers = []
        for i, prompt in enumerate(prompts):
            response = self._generate(
                image=image,
                prompt=prompt,
                max_new_tokens=220,
                do_sample=False
            )
            answer = self._clean_answer(response, anchor="Answer:")
            answer = self._postprocess(answer)
            answers.append(answer)

        return {
            "question_type": qtype,
            "candidate_answers": answers,
            "modality": modality,
            "body_part": body_part,
        }

    # =========================================================
    # 训练集打分：reference-aware label
    # =========================================================
    def _build_lexicons(self) -> Dict[str, List[str]]:
        anatomy_terms = [
            "lung", "liver", "kidney", "brain", "heart", "pleura", "spine",
            "parenchyma", "cortex", "medulla", "capsule", "septum",
            "ventricle", "atrium", "vessel", "artery", "vein", "lymph node",
            "organ", "tissue", "structure", "region", "abdomen", "pelvis",
            "thoracic cavity", "stomach", "bronchus", "trachea", "mandible",
            "oral cavity", "spleen", "spinal cord", "dural sac", "tooth",
            "gallbladder", "cystic duct", "cystic artery"
        ]

        location_terms = [
            "left", "right", "bilateral", "upper", "lower", "central", "peripheral",
            "anterior", "posterior", "superior", "inferior", "medial", "lateral",
            "thoracic cavity", "stomach", "bronchus", "trachea", "d12", "l1", "t12",
            "level", "within", "above", "below", "intradural", "extramedullary",
            "intramedullary", "compartment", "thoracic", "lumbar"
        ]

        comparison_terms = [
            "baseline", "compared", "comparison", "increase", "decrease",
            "progression", "response", "poor response", "positive response",
            "new metastasis", "new lesion", "increased uptake", "reduction in uptake",
            "improvement", "worsening", "stable"
        ]

        finding_terms = [
            "lesion", "mass", "nodule", "cyst", "tumor", "calcification",
            "hemorrhage", "edema", "fracture", "effusion", "opacity",
            "pneumothorax", "stenosis", "occlusion", "metastasis",
            "ischemia", "infarction", "fibrosis", "inflammation",
            "echogenicity", "hyperechoic", "hypoechoic", "hyperintense",
            "hypointense", "sunburst", "radiolucency"
        ]

        return {
            "anatomy_terms": anatomy_terms,
            "location_terms": location_terms,
            "comparison_terms": comparison_terms,
            "finding_terms": finding_terms,
        }

    def _contains_term(self, text: str, term: str) -> bool:
        text = text.lower()
        term = term.lower()
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        return re.search(pattern, text) is not None

    def _extract_terms(self, text: str, terms: List[str]) -> Set[str]:
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

    def _has_number_and_unit(self, text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\b\d+(\.\d+)?\s?(cm|mm)\b",
            r"\b\d+\s?[x×]\s?\d+\s?(cm|mm)\b",
            r"\bapproximately\s+\d+",
            r"\babout\s+\d+"
        ]
        return any(re.search(p, t) for p in patterns)

    def question_demand_score(self, question: str, answer: str, qtype: str) -> float:
        q_lower = question.lower()
        a_lower = answer.lower()
        score = 0.0

        if qtype == "location":
            loc_terms = self._extract_terms(a_lower, self.lexicons["location_terms"])
            if loc_terms:
                score += 1.0
            if any(k in a_lower for k in ["level", "within", "above", "below", "thoracic cavity", "bronchus", "stomach", "d12", "l1", "t12"]):
                score += 0.8
            if any(k in a_lower for k in ["metastatic", "tumor", "infectious", "inflammatory", "deposit"]):
                score -= 0.6

        elif qtype == "size":
            if self._has_number_and_unit(a_lower):
                score += 1.4
            elif re.search(r"\b\d+(\.\d+)?\b", a_lower):
                score += 0.8
            if len(a_lower.split()) > 30:
                score -= 0.5
            if any(k in a_lower for k in ["benign", "malignant", "cyst", "tumor", "subcutaneous", "thigh", "parotid"]):
                score -= 0.5

        elif qtype == "comparison":
            if any(k in a_lower for k in ["baseline", "compared", "comparison"]):
                score += 0.8
            if any(k in a_lower for k in ["progression", "poor response", "improvement", "decrease", "increase", "stable", "worsened"]):
                score += 0.8
            if any(k in a_lower for k in ["new metastasis", "new lesion", "uptake"]):
                score += 0.4
            if ("increase" in a_lower or "progression" in a_lower) and ("decrease" in a_lower or "improvement" in a_lower):
                score -= 0.5
        else:
            anatomy_terms = self._extract_terms(a_lower, self.lexicons["anatomy_terms"])
            finding_terms = self._extract_terms(a_lower, self.lexicons["finding_terms"])
            if anatomy_terms:
                score += 0.4
            if finding_terms:
                score += 0.5
            if len(a_lower.split()) <= 80:
                score += 0.3

            heavy_terms = [
                "abscess", "appendicitis", "severe infection", "critical condition",
                "fever", "chills", "pelvic pain", "immediate veterinary intervention"
            ]
            if any(k in a_lower for k in heavy_terms) and not any(k in q_lower for k in heavy_terms):
                score -= 0.8

        return score

    def reference_alignment_score(self, question: str, answer: str, reference: str) -> Dict:
        qtype = self.classify_question_type(question)

        pred_anatomy = self._extract_terms(answer.lower(), self.lexicons["anatomy_terms"])
        ref_anatomy = self._extract_terms(reference.lower(), self.lexicons["anatomy_terms"])

        pred_location = self._extract_terms(answer.lower(), self.lexicons["location_terms"])
        ref_location = self._extract_terms(reference.lower(), self.lexicons["location_terms"])

        pred_comparison = self._extract_terms(answer.lower(), self.lexicons["comparison_terms"])
        ref_comparison = self._extract_terms(reference.lower(), self.lexicons["comparison_terms"])

        pred_findings = self._extract_terms(answer.lower(), self.lexicons["finding_terms"])
        ref_findings = self._extract_terms(reference.lower(), self.lexicons["finding_terms"])

        anatomy_f1 = self._set_f1(pred_anatomy, ref_anatomy)
        location_f1 = self._set_f1(pred_location, ref_location)
        comparison_f1 = self._set_f1(pred_comparison, ref_comparison)
        finding_f1 = self._set_f1(pred_findings, ref_findings)
        demand = self.question_demand_score(question, answer, qtype)

        if qtype == "location":
            final_score = 0.45 * location_f1 + 0.20 * anatomy_f1 + 0.20 * finding_f1 + 0.15 * max(demand, 0)
        elif qtype == "size":
            size_match = 1.0 if (self._has_number_and_unit(answer) and re.search(r"\d", reference)) else 0.0
            final_score = 0.60 * size_match + 0.40 * max(demand, 0)
        elif qtype == "comparison":
            final_score = 0.45 * comparison_f1 + 0.20 * anatomy_f1 + 0.20 * finding_f1 + 0.15 * max(demand, 0)
        else:
            final_score = 0.35 * finding_f1 + 0.25 * anatomy_f1 + 0.20 * location_f1 + 0.20 * max(demand, 0)

        return {
            "question_type": qtype,
            "anatomy_f1": float(anatomy_f1),
            "location_f1": float(location_f1),
            "comparison_f1": float(comparison_f1),
            "finding_f1": float(finding_f1),
            "demand_score": float(demand),
            "final_score": float(final_score)
        }

    # =========================================================
    # 路径工具
    # =========================================================
    @staticmethod
    def resolve_image_path(images_path: str, image_field) -> str:
        image_relpath = image_field[0] if isinstance(image_field, list) else image_field
        if not image_relpath:
            return ""

        candidates = [
            os.path.join(images_path, str(image_relpath).replace("/", os.sep).replace("\\", os.sep)),
            os.path.join(images_path, os.path.basename(str(image_relpath))),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return ""