import os
import json
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "pubmedvision_ablation_candidate_selector_v1",
    str(SCRIPT_DIR / "pubmedvision_ablation_candidate_selector_v1.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PubMedVisionCandidateSelectorStudy = mod.PubMedVisionCandidateSelectorStudy


def main():
    BASE_MODEL = r"D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5"
    LORA_ADAPTER = r"d:\weng\llama-factory\saves\Qwen2.5-VL-7B-Instruct\lora\exp_dropout_10_full"
    IMAGES_PATH = r"D:\weng\llama-factory\data\images"
    JSON_DATA_FILE = r"D:\weng\llama-factory\data\PubMedVision_VALIDATION.json"
    OUTPUT_DIR = r"./pubmedvision_candidate_selector_v1"

    # ===== 分段范围：跑 500~999 =====
    START_INDEX = 3000
    END_INDEX = 3500

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 先把大验证集切出这一段，生成临时 JSON
    with open(JSON_DATA_FILE, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    chunk_samples = all_samples[START_INDEX:END_INDEX]

    chunk_json_file = os.path.join(
        OUTPUT_DIR,
        f"validation_chunk_{START_INDEX}_{END_INDEX}.json"
    )

    with open(chunk_json_file, "w", encoding="utf-8") as f:
        json.dump(chunk_samples, f, ensure_ascii=False, indent=2)

    print(f"总样本数: {len(all_samples)}")
    print(f"当前分段: [{START_INDEX}, {END_INDEX})")
    print(f"本段样本数: {len(chunk_samples)}")
    print(f"临时分段文件: {chunk_json_file}")

    study = PubMedVisionCandidateSelectorStudy(
        base_model_path=BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER,
    )

    # results = study.run_experiment(
    #     json_data_file=chunk_json_file,
    #     images_path=IMAGES_PATH,
    #     output_dir=os.path.join(OUTPUT_DIR, f"chunk_{START_INDEX}_{END_INDEX}"),
    #     max_samples=len(chunk_samples),
    #     stratified_sampling=False,
    #     include_oracle=True
    # )
    results = study.run_experiment(
        json_data_file=chunk_json_file,
        images_path=IMAGES_PATH,
        output_dir=os.path.join(OUTPUT_DIR, f"chunk_{START_INDEX}_{END_INDEX}"),
        max_samples=len(chunk_samples),
        stratified_sampling=False,
        include_oracle=True,
        resume=True

    # results = study.run_experiment(
    #     json_data_file=JSON_DATA_FILE,
    #     images_path=IMAGES_PATH,
    #     output_dir=OUTPUT_DIR,
    #     max_samples=500,            # 先跑 100，看趋势
    #     stratified_sampling=False,
    #     # seed=42,
    #     include_oracle=True         # oracle 只用于上界分析
    )

    # print("\n实验完成。输出目录：")
    # for k in results.keys():
    #     print(os.path.join(OUTPUT_DIR, k))

    print("\n实验完成。输出目录：")
    for k in results.keys():
        print(os.path.join(OUTPUT_DIR, f"chunk_{START_INDEX}_{END_INDEX}", k))


if __name__ == "__main__":
    main()