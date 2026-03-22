import json
import regex as re
import sys
from pathlib import Path
from typing import List, Tuple
from rouge_score import rouge_scorer
import argparse
import sacrebleu

try:
    from bert_score import score as bert_score
    _has_bert = True
except Exception:
    _has_bert = False

def normalize(text: str):
    text = text.lower()
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\s]", "", text)  # 去标点
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 新增：去除 Teacher:/Student: 前缀的扩展归一化
def normalize_extended(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(student|teacher)\s*:\s*", " ", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_preds(pred_file: Path) -> List[Tuple[str, str]]:
    preds = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append((obj.get("predict", ""), obj.get("label", "")))
    return preds

# 可选：术语同义归一化与关键术语召回（需要 d:/weng/config/terminology.json）
def load_terminology_map(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        term_map = {}
        for canonical, synonyms in data.items():
            sset = {canonical.lower()}
            for s in synonyms if isinstance(synonyms, list) else []:
                sset.add(str(s).lower())
            term_map[canonical.lower()] = sset
        return term_map
    except Exception:
        return {}

def apply_terminology(text: str, term_map: dict) -> str:
    t = text.lower()
    for canonical, sset in term_map.items():
        for term in sset:
            t = re.sub(rf"\b{re.escape(term)}\b", canonical, t)
    return t

def compute_keyword_recall(pred: str, label: str, term_map: dict) -> float:
    if not term_map:
        return -1.0
    lp = apply_terminology(label, term_map)
    pp = apply_terminology(pred, term_map)
    label_terms = set()
    pred_terms = set()
    for canonical, sset in term_map.items():
        if re.search(rf"\b{re.escape(canonical)}\b", lp):
            label_terms.add(canonical)
        if re.search(rf"\b{re.escape(canonical)}\b", pp):
            pred_terms.add(canonical)
    if not label_terms:
        return -1.0
    return len(label_terms & pred_terms) / len(label_terms)

def log_to_swanlab(metrics: dict, run_name: str = "eval-qwen2_5vl_lora_sft_generate-post") -> None:
    try:
        import swanlab
    except Exception:
        print("SwanLab 未安装，跳过上报。可运行: pip install swanlab")
        return
    # 如果你已在 YAML 中设置了 SWANLAB_API_KEY 或已登录，这里会复用账号环境
    swanlab.init(run_name=run_name)
    swanlab.log(metrics)
    swanlab.finish()

def main():
    # 默认路径：与当前 YAML 的 output_dir 保持一致
    default_path = Path(r"d:/weng/llama-factory/saves/qwen2_5vl-7b/lora/sft/eval/generated_predictions.jsonl")
    # 新增命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=str(default_path))
    parser.add_argument("--run_name", type=str, default="eval-qwen2_5vl_lora_sft_generate-post")
    parser.add_argument("--upload_to_swanlab", action="store_true")
    parser.add_argument("--use_bert_score", action="store_true")
    parser.add_argument("--bert_model", type=str, default="roberta-large")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    assert pred_path.exists(), f"File not found: {pred_path}"

    pairs = load_preds(pred_path)
    if not pairs:
        print("No predictions found.")
        return

    # 可选：加载术语表
    term_map = load_terminology_map(Path(r"d:/weng/config/terminology.json"))

    strict_em = 0
    ci_em = 0
    norm_em = 0

    # 改为英文版 ROUGE-1/2/L
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_f1 = 0.0
    rouge2_f1 = 0.0
    rougeL_f1 = 0.0

    hyps = []
    refs = []
    total_keyword_recall = 0.0
    total_pred_len = 0
    total_label_len = 0

    for pred, label in pairs:
        hyps.append(pred)
        refs.append(label)
        total_pred_len += len(pred.split())
        total_label_len += len(label.split())

        if pred == label:
            strict_em += 1
        if pred.lower().strip() == label.lower().strip():
            ci_em += 1
        if normalize_extended(pred) == normalize_extended(label):
            norm_em += 1

        scores = scorer.score(label, pred)
        rouge1_f1 += scores["rouge1"].fmeasure * 100.0
        rouge2_f1 += scores["rouge2"].fmeasure * 100.0
        rougeL_f1 += scores["rougeL"].fmeasure * 100.0

        kr = compute_keyword_recall(pred, label, term_map)
        if kr >= 0:
            total_keyword_recall += kr * 100.0

    # 语料级 BLEU-4
    try:
        bleu4 = sacrebleu.corpus_bleu(hyps, [refs], use_effective_order=True).score
    except Exception:
        bleu4 = -1.0

    # 可选：BERTScore F1（需 --use_bert_score 且安装 bert-score）
    bert_f1 = -1.0
    if args.use_bert_score and _has_bert:
        try:
            P, R, F = bert_score(hyps, refs, model_type=args.bert_model, lang="en", verbose=False)
            bert_f1 = float(F.mean().item()) * 100.0
        except Exception:
            bert_f1 = -1.0

    n = len(pairs)
    metrics = {
        "strict_em": round(strict_em / n * 100, 4),
        "case_insensitive_em": round(ci_em / n * 100, 4),
        "normalized_em": round(norm_em / n * 100, 4),
        "rouge1_f1_en": round(rouge1_f1 / n, 4),
        "rouge2_f1_en": round(rouge2_f1 / n, 4),
        "rougeL_f1_en": round(rougeL_f1 / n, 4),
        "bleu4_en": round(bleu4, 4) if bleu4 >= 0 else "n/a",
        "bert_f1_en": round(bert_f1, 4) if bert_f1 >= 0 else "n/a",
        "avg_pred_len": round(total_pred_len / n, 2),
        "avg_label_len": round(total_label_len / n, 2),
        "len_ratio_pred_label": round((total_pred_len / n) / max(1.0, (total_label_len / n)), 4),
        "keyword_recall_percent": round(total_keyword_recall / n, 4) if term_map else "n/a",
        "num_samples": n,
        "source": str(pred_path)
    }

    out_file = pred_path.parent / "em_rouge_en_report.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 新增：可选上报到 SwanLab
    if args.upload_to_swanlab:
        log_to_swanlab(metrics, run_name=args.run_name)

if __name__ == "__main__":
    main()