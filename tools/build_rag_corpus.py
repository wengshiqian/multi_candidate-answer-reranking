import argparse
import json
from typing import List, Dict, Iterable, Set, Tuple


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_id(sample: Dict) -> str:
    return str(sample.get("id", "")).strip()


def _extract_images(sample: Dict) -> Tuple[str, ...]:
    images = sample.get("image") or []
    if isinstance(images, str):
        images = [images]
    return tuple(str(x).strip() for x in images if x)


def _extract_question(sample: Dict) -> str:
    convs = sample.get("conversations") or []
    if convs:
        return str(convs[0].get("value", "")).strip()
    return ""


def _build_key_sets(samples: Iterable[Dict]) -> Tuple[Set[str], Set[Tuple[str, ...]], Set[str]]:
    id_set: Set[str] = set()
    image_set: Set[Tuple[str, ...]] = set()
    question_set: Set[str] = set()
    for sample in samples:
        sample_id = _extract_id(sample)
        if sample_id:
            id_set.add(sample_id)
        images = _extract_images(sample)
        if images:
            image_set.add(images)
        question = _extract_question(sample)
        if question:
            question_set.add(question)
    return id_set, image_set, question_set


def _filter_samples(
    source_samples: List[Dict],
    drop_ids: Set[str],
    drop_images: Set[Tuple[str, ...]],
    drop_questions: Set[str]
) -> List[Dict]:
    filtered: List[Dict] = []
    for sample in source_samples:
        sample_id = _extract_id(sample)
        if sample_id and sample_id in drop_ids:
            continue
        images = _extract_images(sample)
        if images and images in drop_images:
            continue
        question = _extract_question(sample)
        if question and question in drop_questions:
            continue
        filtered.append(sample)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter PubMedVision InstructionTuning by validation set.")
    parser.add_argument("--instruction", required=True, help="Path to PubMedVision_InstructionTuning_VQA.json")
    parser.add_argument("--validation", required=True, help="Path to PubMedVision_VALIDATION.json (test set)")
    parser.add_argument("--output", required=True, help="Output path for filtered RAG corpus JSON")
    args = parser.parse_args()

    instruction_samples = _load_json(args.instruction)
    validation_samples = _load_json(args.validation)

    drop_ids, drop_images, drop_questions = _build_key_sets(validation_samples)
    filtered_samples = _filter_samples(instruction_samples, drop_ids, drop_images, drop_questions)

    removed = len(instruction_samples) - len(filtered_samples)
    print(f"Loaded instruction samples: {len(instruction_samples)}")
    print(f"Loaded validation samples: {len(validation_samples)}")
    print(f"Removed samples: {removed}")
    print(f"Filtered samples: {len(filtered_samples)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved filtered RAG corpus to: {args.output}")


if __name__ == "__main__":
    main()
