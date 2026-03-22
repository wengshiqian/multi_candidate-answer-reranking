import os
import json
import time
import random
from typing import Dict, List
from collections import defaultdict
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    "candidate_selector_workflow_v1",
    "candidate_selector_workflow_v1.py"
)
workflow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_module)

CandidateSelectorWorkflow = workflow_module.CandidateSelectorWorkflow
MedicalReportState = workflow_module.MedicalReportState


class PubMedVisionCandidateSelectorStudy:
    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: str,
        device: str = "cuda"
    ):
        self.workflow_obj = CandidateSelectorWorkflow(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            device=device
        )

    def load_pubmedvision_data(
        self,
        json_data_file: str,
        max_samples: int = 100,
        stratified: bool = True,
        seed: int = 42
    ) -> List[dict]:
        with open(json_data_file, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        rng = random.Random(seed)

        if not stratified or len(all_samples) <= max_samples:
            return rng.sample(all_samples, min(max_samples, len(all_samples)))

        groups = defaultdict(list)
        for s in all_samples:
            key = f"{s.get('modality', 'Unknown')}_{s.get('body_part', 'Unknown')}"
            groups[key].append(s)

        total_groups = len(groups)
        base_per_group = max_samples // total_groups
        remainder = max_samples % total_groups

        selected = []
        for i, key in enumerate(groups.keys()):
            group = groups[key]
            k = min(len(group), base_per_group + (1 if i < remainder else 0))
            selected.extend(rng.sample(group, k))

        return selected

    def _make_initial_state(self, sample: dict, image_path: str) -> MedicalReportState:
        question = sample["conversations"][0]["value"].replace("<image>", "").strip()

        return MedicalReportState(
            sample_id=sample.get("id", "unknown"),
            image_path=image_path,
            question=question,
            modality=sample.get("modality", "Unknown"),
            body_part=sample.get("body_part", "Unknown"),
            candidate_answers=[],
            final_report="",
            metadata={}
        )

    def run_experiment(
            self,
            json_data_file: str,
            images_path: str,
            output_dir: str,
            max_samples: int = 100,
            stratified_sampling: bool = True,
            seed: int = 42,
            include_oracle: bool = True,
            resume: bool = True,
    ):
        os.makedirs(output_dir, exist_ok=True)

        samples = self.load_pubmedvision_data(
            json_data_file=json_data_file,
            max_samples=max_samples,
            stratified=stratified_sampling,
            seed=seed
        )

        experiments = {
            "baseline": self._run_baseline,
            "multi_candidate_first": self._run_multi_candidate_first,
            "multi_candidate_llm_selector": self._run_multi_candidate_llm,
        }

        if include_oracle:
            experiments["multi_candidate_oracle"] = self._run_multi_candidate_oracle

        all_results = {}

        for exp_name, runner in experiments.items():
            print("\n" + "#" * 80)
            print(f"🔬 运行实验: {exp_name}")
            print("#" * 80)

            exp_dir = os.path.join(output_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            pred_file = os.path.join(exp_dir, "predictions.jsonl")
            summary_file = os.path.join(exp_dir, "run_summary.json")

            if resume and os.path.exists(pred_file) and os.path.exists(summary_file):
                print(f"✅ {exp_name} 已完成，直接跳过。")
                continue

            res = self._run_single_experiment(
                runner_func=runner,
                samples=samples,
                images_path=images_path
            )
            all_results[exp_name] = res

            with open(pred_file, "w", encoding="utf-8") as f:
                for item in res["predictions"]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_count": res["sample_count"],
                    "avg_time_seconds": res["avg_time_seconds"],
                    "errors": res["errors"],
                }, f, indent=2, ensure_ascii=False)

        return all_results
    # def run_experiment(
    #     self,
    #     json_data_file: str,
    #     images_path: str,
    #     output_dir: str,
    #     max_samples: int = 100,
    #     stratified_sampling: bool = True,
    #     seed: int = 42,
    #     include_oracle: bool = True,
    # ):
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     samples = self.load_pubmedvision_data(
    #         json_data_file=json_data_file,
    #         max_samples=max_samples,
    #         stratified=stratified_sampling,
    #         seed=seed
    #     )
    #
    #     experiments = {
    #         "baseline": self._run_baseline,
    #         "multi_candidate_first": self._run_multi_candidate_first,
    #         # "multi_candidate_heuristic_selector": self._run_multi_candidate_heuristic,
    #         "multi_candidate_llm_selector": self._run_multi_candidate_llm,
    #     }
    #
    #     if include_oracle:
    #         experiments["multi_candidate_oracle"] = self._run_multi_candidate_oracle
    #
    #     all_results = {}
    #
    #     for exp_name, runner in experiments.items():
    #         print("\n" + "#" * 80)
    #         print(f"🔬 运行实验: {exp_name}")
    #         print("#" * 80)
    #
    #         exp_dir = os.path.join(output_dir, exp_name)
    #         os.makedirs(exp_dir, exist_ok=True)
    #
    #         res = self._run_single_experiment(
    #             runner_func=runner,
    #             samples=samples,
    #             images_path=images_path
    #         )
    #         all_results[exp_name] = res
    #
    #         with open(os.path.join(exp_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
    #             for item in res["predictions"]:
    #                 f.write(json.dumps(item, ensure_ascii=False) + "\n")
    #
    #         with open(os.path.join(exp_dir, "run_summary.json"), "w", encoding="utf-8") as f:
    #             json.dump({
    #                 "sample_count": res["sample_count"],
    #                 "avg_time_seconds": res["avg_time_seconds"],
    #                 "errors": res["errors"],
    #             }, f, indent=2, ensure_ascii=False)
    #
    #     return all_results

    def _run_baseline(self, state: MedicalReportState, reference: str) -> MedicalReportState:
        state = self.workflow_obj.generate_baseline_answer(state)
        return state

    def _run_multi_candidate_first(self, state: MedicalReportState, reference: str) -> MedicalReportState:
        state = self.workflow_obj.generate_three_candidates(state)
        state = self.workflow_obj.select_first_candidate(state)
        return state

    # def _run_multi_candidate_heuristic(self, state: MedicalReportState, reference: str) -> MedicalReportState:
    #     state = self.workflow_obj.generate_three_candidates(state)
    #     state = self.workflow_obj.select_by_heuristic(state)
    #     return state

    def _run_multi_candidate_llm(self, state: MedicalReportState, reference: str) -> MedicalReportState:
        state = self.workflow_obj.generate_three_candidates(state)
        state = self.workflow_obj.select_by_llm(state)
        return state

    def _run_multi_candidate_oracle(self, state: MedicalReportState, reference: str) -> MedicalReportState:
        state = self.workflow_obj.generate_three_candidates(state)
        result = self.workflow_obj.oracle_select_with_reference(
            question=state["question"],
            answers=state["candidate_answers"],
            reference=reference
        )
        state["final_report"] = result["selected_answer"]
        state["metadata"] = state.get("metadata", {})
        state["metadata"]["selector"] = "oracle_reference_upper_bound"
        state["metadata"]["selected_candidate_id"] = result["selected_candidate_id"]
        state["metadata"]["oracle_scores"] = result["oracle_scores"]
        return state

    def _run_single_experiment(
        self,
        runner_func,
        samples: List[dict],
        images_path: str
    ) -> Dict:
        predictions = []
        timings = []
        errors = []

        for idx, sample in enumerate(samples):
            print(f"处理样本 {idx + 1}/{len(samples)} (ID: {sample.get('id', f'sample_{idx}')})")

            try:
                image_field = sample.get("image")
                image_relpath = image_field[0] if isinstance(image_field, list) else image_field
                if not image_relpath:
                    continue

                candidates = [
                    os.path.join(images_path, str(image_relpath).replace("/", os.sep).replace("\\", os.sep)),
                    os.path.join(images_path, os.path.basename(str(image_relpath))),
                ]
                image_path = next((p for p in candidates if os.path.exists(p)), None)

                if image_path is None:
                    errors.append({
                        "id": sample.get("id", f"sample_{idx}"),
                        "error": f"image not found: {image_relpath}"
                    })
                    continue

                reference = sample["conversations"][1]["value"].replace("<image>", "").strip()
                state = self._make_initial_state(sample, image_path)

                st = time.time()
                final_state = runner_func(state, reference)
                ed = time.time()
                timings.append(ed - st)

                predictions.append({
                    "id": state["sample_id"],
                    "question": state["question"],
                    "reference": reference,
                    "prediction": final_state.get("final_report", ""),
                    "modality": state["modality"],
                    "body_part": state["body_part"],
                    "candidate_answers": final_state.get("candidate_answers", []),
                    "metadata": final_state.get("metadata", {})
                })

            except Exception as e:
                errors.append({
                    "id": sample.get("id", f"sample_{idx}"),
                    "error": str(e)
                })

        return {
            "predictions": predictions,
            "sample_count": len(predictions),
            "avg_time_seconds": float(np.mean(timings)) if timings else 0.0,
            "errors": errors
        }