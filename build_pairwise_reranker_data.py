import os
import json
import random
from pathlib import Path
from itertools import combinations
import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "candidate_generator_distill_v1",
    str(SCRIPT_DIR / "candidate_generator_distill_v1.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DistillCandidateGenerator = mod.DistillCandidateGenerator


def main():
    BASE_MODEL = r"D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5"
    LORA_ADAPTER = r"d:\weng\llama-factory\saves\Qwen2.5-VL-7B-Instruct\lora\exp_dropout_10_full"

    TRAIN_JSON = r"D:\weng\llama-factory\data\PubMedVision_CLEANED.json"
    IMAGES_PATH = r"D:\weng\llama-factory\data\images"

    OUTPUT_DIR = r"./distilled_reranker_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MAX_SAMPLES = 1000
    RANDOM_SEED = 42
    MIN_MARGIN = 0.02  # 两个 candidate 分数差太小就不拿来训

    random.seed(RANDOM_SEED)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    random.shuffle(samples)
    samples = samples[:MAX_SAMPLES]

    generator = DistillCandidateGenerator(
        base_model_path=BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER,
        device="cuda"
    )

    candidate_dump = []
    pairwise_records = []

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx}")
        print(f"处理样本 {idx + 1}/{len(samples)} | {sample_id}")

        image_path = generator.resolve_image_path(IMAGES_PATH, sample.get("image"))
        if not image_path:
            print(f"⚠️ 找不到图片: {sample_id}")
            continue

        question = sample["conversations"][0]["value"].replace("<image>", "").strip()
        reference = sample["conversations"][1]["value"].replace("<image>", "").strip()
        modality = sample.get("modality", "Unknown")
        body_part = sample.get("body_part", "Unknown")

        result = generator.generate_candidates_for_question(
            image_path=image_path,
            question=question,
            modality=modality,
            body_part=body_part
        )

        answers = result["candidate_answers"]
        qtype = result["question_type"]

        scored_candidates = []
        for cid, ans in enumerate(answers):
            metrics = generator.reference_alignment_score(question, ans, reference)
            scored_candidates.append({
                "candidate_id": cid,
                "answer": ans,
                "metrics": metrics
            })

        candidate_dump.append({
            "id": sample_id,
            "question": question,
            "reference": reference,
            "question_type": qtype,
            "modality": modality,
            "body_part": body_part,
            "candidate_answers": answers,
            "candidate_scores": scored_candidates
        })

        # pairwise 构造
        for i, j in combinations(range(len(answers)), 2):
            score_i = scored_candidates[i]["metrics"]["final_score"]
            score_j = scored_candidates[j]["metrics"]["final_score"]

            if abs(score_i - score_j) < MIN_MARGIN:
                continue

            # label=1 表示 candidate_a better than candidate_b
            if score_i > score_j:
                better_idx, worse_idx = i, j
            else:
                better_idx, worse_idx = j, i

            pairwise_records.append({
                "sample_id": sample_id,
                "question_type": qtype,
                "question": question,
                "candidate_a": answers[better_idx],
                "candidate_b": answers[worse_idx],
                "label": 1,
                "score_a": max(score_i, score_j),
                "score_b": min(score_i, score_j),
                "margin": abs(score_i - score_j),
            })

            # 反向样本
            pairwise_records.append({
                "sample_id": sample_id,
                "question_type": qtype,
                "question": question,
                "candidate_a": answers[worse_idx],
                "candidate_b": answers[better_idx],
                "label": 0,
                "score_a": min(score_i, score_j),
                "score_b": max(score_i, score_j),
                "margin": abs(score_i - score_j),
            })

    candidate_dump_file = os.path.join(OUTPUT_DIR, "train_candidate_dump.json")
    with open(candidate_dump_file, "w", encoding="utf-8") as f:
        json.dump(candidate_dump, f, indent=2, ensure_ascii=False)

    pairwise_file = os.path.join(OUTPUT_DIR, "pairwise_train.jsonl")
    with open(pairwise_file, "w", encoding="utf-8") as f:
        for rec in pairwise_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n✅ 数据构造完成")
    print(f"候选集文件: {candidate_dump_file}")
    print(f"Pairwise 训练文件: {pairwise_file}")
    print(f"样本数: {len(candidate_dump)}")
    print(f"Pairwise 条数: {len(pairwise_records)}")


if __name__ == "__main__":
    main()