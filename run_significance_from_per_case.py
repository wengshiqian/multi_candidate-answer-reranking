import os
import json
from pathlib import Path
from itertools import combinations
import importlib.util
import gc

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SCRIPT_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "candidate_generator_distill_v1",
    str(SCRIPT_DIR / "candidate_generator_distill_v1.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DistillCandidateGenerator = mod.DistillCandidateGenerator


class PairwiseReranker:
    def __init__(self, model_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score_pair(self, question: str, candidate_a: str, candidate_b: str) -> float:
        text = (
            f"[QUESTION] {question} "
            f"[CANDIDATE_A] {candidate_a} "
            f"[CANDIDATE_B] {candidate_b}"
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=384,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # label=1 means candidate_a better than candidate_b
        return float(probs[1].item())

    def rank_candidates(self, question: str, candidates):
        n = len(candidates)
        win_scores = [0.0] * n
        detail = []

        for i, j in combinations(range(n), 2):
            p_ij = self.score_pair(question, candidates[i], candidates[j])  # i > j
            p_ji = self.score_pair(question, candidates[j], candidates[i])  # j > i

            # 对称合并
            score_i = (p_ij + (1.0 - p_ji)) / 2.0
            score_j = (p_ji + (1.0 - p_ij)) / 2.0

            win_scores[i] += score_i
            win_scores[j] += score_j

            detail.append({
                "i": i,
                "j": j,
                "p_i_beats_j": score_i,
                "p_j_beats_i": score_j
            })

        ranked = sorted(
            [{"candidate_id": i, "score": win_scores[i], "answer": candidates[i]} for i in range(n)],
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "ranked": ranked,
            "pairwise_detail": detail
        }


def main():
    BASE_MODEL = r"D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5"
    LORA_ADAPTER = r"d:\weng\llama-factory\saves\Qwen2.5-VL-7B-Instruct\lora\exp_dropout_10_full"

    VAL_JSON = r"D:\weng\llama-factory\data\PubMedVision_VALIDATION.json"
    IMAGES_PATH = r"D:\weng\llama-factory\data\images"

    RERANKER_CKPT = r"./distilled_reranker_ckpt"
    OUTPUT_DIR = r"./distilled_reranker_predictions"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 跑第 500~999 条样本（Python 切片右边不包含）
    START_INDEX = 3000
    END_INDEX = 3500

    generator = DistillCandidateGenerator(
        base_model_path=BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER,
        device="cuda"
    )
    reranker = PairwiseReranker(RERANKER_CKPT)

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    samples = all_samples[START_INDEX:END_INDEX]

    print(f"总样本数: {len(all_samples)}")
    print(f"当前分段: [{START_INDEX}, {END_INDEX})")
    print(f"本段样本数: {len(samples)}")

    predictions = []
    oom_samples = []
    failed_samples = []

    for idx, sample in enumerate(samples):
        global_idx = START_INDEX + idx
        sample_id = sample.get("id", f"sample_{global_idx}")
        print(f"处理样本 全局{global_idx + 1} | 分段内{idx + 1}/{len(samples)} | {sample_id}")

        try:
            image_path = generator.resolve_image_path(IMAGES_PATH, sample.get("image"))
            if not image_path:
                print(f"⚠️ 找不到图片: {sample_id}")
                failed_samples.append({
                    "id": sample_id,
                    "reason": "image_not_found"
                })
                continue

            question = sample["conversations"][0]["value"].replace("<image>", "").strip()
            reference = sample["conversations"][1]["value"].replace("<image>", "").strip()
            modality = sample.get("modality", "Unknown")
            body_part = sample.get("body_part", "Unknown")

            gen_result = generator.generate_candidates_for_question(
                image_path=image_path,
                question=question,
                modality=modality,
                body_part=body_part
            )

            candidates = gen_result["candidate_answers"]
            qtype = gen_result["question_type"]

            rerank_result = reranker.rank_candidates(question, candidates)
            best = rerank_result["ranked"][0]
            selected_idx = best["candidate_id"]
            final_answer = best["answer"]

            predictions.append({
                "id": sample_id,
                "question": question,
                "reference": reference,
                "prediction": final_answer,
                "modality": modality,
                "body_part": body_part,
                "candidate_answers": candidates,
                "metadata": {
                    "question_type": qtype,
                    "selector": "reference_distilled_pairwise_reranker_v1",
                    "selected_candidate_id": selected_idx,
                    "reranker_ranked": rerank_result["ranked"],
                    "pairwise_detail": rerank_result["pairwise_detail"]
                }
            })

        except torch.OutOfMemoryError:
            print(f"⚠️ CUDA OOM，跳过样本: {sample_id}")
            oom_samples.append({
                "id": sample_id,
                "reason": "cuda_oom"
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        except Exception as e:
            print(f"⚠️ 样本处理失败，跳过: {sample_id} | {e}")
            failed_samples.append({
                "id": sample_id,
                "reason": str(e)
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    pred_file = os.path.join(OUTPUT_DIR, f"predictions_{START_INDEX}_{END_INDEX}.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    oom_file = os.path.join(OUTPUT_DIR, f"oom_samples_{START_INDEX}_{END_INDEX}.json")
    with open(oom_file, "w", encoding="utf-8") as f:
        json.dump(oom_samples, f, indent=2, ensure_ascii=False)

    failed_file = os.path.join(OUTPUT_DIR, f"failed_samples_{START_INDEX}_{END_INDEX}.json")
    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(failed_samples, f, indent=2, ensure_ascii=False)

    summary = {
        "start_index": START_INDEX,
        "end_index": END_INDEX,
        "requested_samples": len(samples),
        "successful_predictions": len(predictions),
        "oom_count": len(oom_samples),
        "failed_count": len(failed_samples)
    }
    summary_file = os.path.join(OUTPUT_DIR, f"run_summary_{START_INDEX}_{END_INDEX}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ 推理完成")
    print(f"预测文件: {pred_file}")
    print(f"OOM文件: {oom_file}")
    print(f"失败文件: {failed_file}")
    print(f"汇总文件: {summary_file}")


if __name__ == "__main__":
    main()