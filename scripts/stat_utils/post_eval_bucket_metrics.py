import argparse
import json
import os
import re
import random
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

# Optional: sacrebleu and rouge-score for text metrics
try:
    import sacrebleu
except Exception:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


NUMERIC_PATTERNS = [
    r"\d+(\.\d+)?\s*%",                # percentages
    r"\b\d+(\.\d+)?\b",                # plain numbers
    r"\b(mm|cm|mg/dL|g/dL|mmHg|mL|L|kg|mg|μg|nm|μm)\b",
    r"\bSUV\b",
    r"\b(diameter|size|dimension|count|score|index|level|grade|stage)\b",
    r"\b×10\^\d+\b",
]

PANEL_PATTERNS = [
    r"\bpanel(s)?\b",
    r"\bsubfigure(s)?\b",
    r"\bfigure\s*\d+[A-Z]\b",
    r"\b[A-D]\d?\b",  # e.g., A, B, C, D, B4
]

# For parsing panel letters from text
PANEL_LETTER_RE = re.compile(r"\b([A-Z])\d?\b")

WHITESPACE_RE = re.compile(r"\s+")


def is_numeric_question(text: str) -> bool:
    t = text.lower()
    for pat in NUMERIC_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def is_multi_panel(images: List[str], text: str) -> bool:
    if len(images) > 1:
        return True
    t = text.lower()
    for pat in PANEL_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def classify_sample(question: str, images: List[str]) -> str:
    # precedence: numeric -> multi_panel -> semantic
    if is_numeric_question(question):
        return "numeric"
    if is_multi_panel(images, question):
        return "multi_panel"
    return "semantic"


def simple_tokenize(s: str) -> List[str]:
    return [tok for tok in WHITESPACE_RE.split(s.strip()) if tok]


def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip() == gold.strip())


def token_f1(pred: str, gold: str) -> float:
    p_tokens = simple_tokenize(pred.lower())
    g_tokens = simple_tokenize(gold.lower())
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    p_set, g_set = set(p_tokens), set(g_tokens)
    tp = len(p_set & g_set)
    precision = tp / len(p_set) if p_set else 0.0
    recall = tp / len(g_set) if g_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_numbers(s: str) -> List[float]:
    nums = []
    for m in re.finditer(r"-?\d+(\.\d+)?", s):
        try:
            nums.append(float(m.group()))
        except Exception:
            pass
    return nums


def nem_score(pred_nums: List[float], gold_nums: List[float], tol_ratio: float) -> float:
    # Numerator: matched gold numbers within tolerance
    # Denominator: number of gold numbers
    if not gold_nums:
        return float("nan")
    matched = 0
    for g in gold_nums:
        found = False
        for p in pred_nums:
            if tol_ratio == 0.0:
                if p == g:
                    found = True
                    break
            else:
                # absolute tolerance in proportion to gold magnitude
                if g == 0 and p == 0:
                    found = True
                    break
                if abs(p - g) <= tol_ratio * max(abs(g), 1e-8):
                    found = True
                    break
        if found:
            matched += 1
    return matched / len(gold_nums)


def bootstrap_ci(values: List[float], B: int = 1000, conf: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), (float("nan"), float("nan"))
    n = len(clean)
    means = []
    for _ in range(B):
        sample = [clean[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    mean_val = sum(clean) / len(clean)
    low_idx = int(((1 - conf) / 2) * B)
    high_idx = int((1 - (1 - conf) / 2) * B)
    return mean_val, (means[low_idx], means[high_idx])


def parse_panels_from_text(s: str) -> List[str]:
    letters = []
    for m in PANEL_LETTER_RE.finditer(s):
        letter = m.group(1)
        if "A" <= letter <= "Z":
            letters.append(letter)
    # de-duplicate preserving order
    seen = set()
    out = []
    for l in letters:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def compute_bleu_corpus(preds: List[str], refs: List[str]) -> Optional[float]:
    if sacrebleu is None:
        return None
    return sacrebleu.corpus_bleu(preds, [refs]).score


def compute_rouge_avg(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if rouge_scorer is None:
        return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)  # ROUGE recall by default; change if needed
        r1.append(scores["rouge1"].recall)
        r2.append(scores["rouge2"].recall)
        rl.append(scores["rougeL"].recall)
    return {
        "rouge1": sum(r1) / len(r1) if r1 else float("nan"),
        "rouge2": sum(r2) / len(r2) if r2 else float("nan"),
        "rougeL": sum(rl) / len(rl) if rl else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="Path to eval output_dir containing generated_predictions.jsonl")
    ap.add_argument("--dataset", required=True, help="Path to PubMedVision_VALIDATION.json")
    ap.add_argument("--panel-gt", default=None, help="Optional path to panel GT mapping (id -> [letters])")
    ap.add_argument("--ocr", default=None, help="Optional path to OCR outputs (id -> list of evidence strings/numbers)")
    ap.add_argument("--short-max-tokens", type=int, default=6, help="Max tokens for short answer EM/F1")
    args = ap.parse_args()

    pred_path = os.path.join(args.eval_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing {pred_path}. Ensure eval finished and saved predictions.")

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)

    with open(pred_path, "r", encoding="utf-8") as f:
        preds_raw = [json.loads(line) for line in f if line.strip()]

    if len(preds_raw) != len(ds):
        print(f"Warning: predictions ({len(preds_raw)}) != dataset size ({len(ds)}). Will align by index.")

    panel_gt = {}
    if args.panel_gt and os.path.exists(args.panel_gt):
        with open(args.panel_gt, "r", encoding="utf-8") as f:
            panel_gt = json.load(f)  # expect {sample_id: ["A","C",...], ...}

    ocr_map = {}
    if args.ocr and os.path.exists(args.ocr):
        with open(args.ocr, "r", encoding="utf-8") as f:
            ocr_map = json.load(f)  # expect {sample_id: ["text or number", ...]}

    buckets = {"numeric": [], "multi_panel": [], "semantic": []}
    per_sample = []

    for i, sample in enumerate(ds):
        question = sample["conversations"][0]["value"]
        gold = sample["conversations"][1]["value"]
        images = sample.get("image", [])
        sid = sample.get("id", str(i))

        # predictions are in the same order as dataset
        if i < len(preds_raw):
            pred_text = preds_raw[i]["predict"]
        else:
            pred_text = ""

        bucket = classify_sample(question, images)
        buckets[bucket].append(i)

        per_sample.append({
            "id": sid,
            "bucket": bucket,
            "question": question,
            "gold": gold,
            "pred": pred_text,
            "images": images,
        })

    # Metrics per bucket
    results = {}
    for bucket_name, idxs in buckets.items():
        subset = [per_sample[i] for i in idxs]
        golds = [x["gold"] for x in subset]
        preds = [x["pred"] for x in subset]

        # BLEU/ROUGE (reference for language quality)
        bleu = compute_bleu_corpus(preds, golds)
        rouge = compute_rouge_avg(preds, golds)

        # Short-answer EM/F1
        em_list, f1_list = [], []
        for x in subset:
            g_toks = simple_tokenize(x["gold"])
            p_toks = simple_tokenize(x["pred"])
            if len(g_toks) <= args.short_max_tokens and len(p_toks) <= max(args.short_max_tokens * 2, 12):
                em_list.append(exact_match(x["pred"], x["gold"]))
                f1_list.append(token_f1(x["pred"], x["gold"]))

        em = sum(em_list) / len(em_list) if em_list else float("nan")
        f1 = sum(f1_list) / len(f1_list) if f1_list else float("nan")

        results[bucket_name] = {
            "count": len(subset),
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "em_short": em,
            "f1_short": f1,
        }

    # Numeric Extraction Metric (NEM) on numeric bucket
    nem_stats = {}
    tol_set = [0.0, 0.01, 0.05]
    numeric_subset = [per_sample[i] for i in buckets["numeric"]]
    if numeric_subset:
        gold_nums_list = [extract_numbers(x["gold"]) for x in numeric_subset]
        pred_nums_list = [extract_numbers(x["pred"]) for x in numeric_subset]
        for tol in tol_set:
            scores = []
            for gold_nums, pred_nums in zip(gold_nums_list, pred_nums_list):
                scores.append(nem_score(pred_nums, gold_nums, tol))
            mean_val, ci = bootstrap_ci(scores, B=1000, conf=0.95)
            nem_stats[f"tol_{int(tol*100)}%"] = {"mean": mean_val, "ci95": [ci[0], ci[1]]}
    results["numeric_nem"] = nem_stats

    # SFR (requires GT panels). If not provided, skip or do weak proxy.
    sfr_values = []
    multi_subset = [per_sample[i] for i in buckets["multi_panel"]]
    for x in multi_subset:
        sid = x["id"]
        gi = set(panel_gt.get(sid, []))
        if gi:
            pi = set(parse_panels_from_text(x["pred"]))
            sfr = len(gi & pi) / len(gi) if len(gi) > 0 else float("nan")
            sfr_values.append(sfr)
    if sfr_values:
        mean_val, ci = bootstrap_ci(sfr_values, B=1000, conf=0.95)
        results["sfr"] = {"mean": mean_val, "ci95": [ci[0], ci[1]], "count": len(sfr_values)}
    else:
        results["sfr"] = {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "count": 0, "note": "No panel GT provided"}

    # Provenance Precision (optional, needs OCR or structured evidence).
    # Here as a simple proxy: numbers in pred that also appear in OCR map for the sample are counted true.
    prov_true, prov_total = 0, 0
    if ocr_map:
        for x in per_sample:
            sid = x["id"]
            pred_nums = extract_numbers(x["pred"])
            if pred_nums:
                prov_total += len(pred_nums)
                ocr_nums = []
                for item in ocr_map.get(sid, []):
                    ocr_nums.extend(extract_numbers(str(item)))
                # simple exact match proxy; can extend to tolerance or string matching
                for pn in pred_nums:
                    if pn in ocr_nums:
                        prov_true += 1
        results["provenance_precision"] = {
            "precision": (prov_true / prov_total) if prov_total else float("nan"),
            "count_pred_evidence": prov_total,
        }
    else:
        results["provenance_precision"] = {"precision": float("nan"), "note": "No OCR map provided"}

    # Save results
    out_json = os.path.join(args.eval_dir, "bucketed_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "bucket_counts": {k: len(v) for k, v in buckets.items()}}, f, ensure_ascii=False, indent=2)

    print(f"Saved bucketed metrics to: {out_json}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()