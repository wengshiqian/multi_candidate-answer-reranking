import argparse, json, os, glob
from typing import List, Dict
import easyocr

def resolve_image_path(rel_path: str, roots: List[str]) -> str:
    # Try direct join under each root
    for root in roots:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    # Try searching in PubMedVision images_* buckets by filename
    fname = os.path.basename(rel_path)
    for bucket in glob.glob(r"d:\weng\PubMedVision\images_*"):
        cand = os.path.join(bucket, "images", fname)
        if os.path.exists(cand):
            return cand
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to PubMedVision_VALIDATION.json")
    ap.add_argument("--images-root", default="d:\\weng\\llama-factory\\data", help="Primary images root")
    ap.add_argument("--out", required=True, help="Path to save OCR json")
    ap.add_argument("--lang", default="en", help="EasyOCR language code, e.g., en")
    ap.add_argument("--min-conf", type=float, default=0.5, help="Min confidence to keep OCR tokens")
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)

    roots = [args.images_root, r"d:\weng\llama-factory\data", r"d:\weng\PubMedVision"]
    reader = easyocr.Reader([args.lang], gpu=True)
    ocr_map: Dict[str, List[str]] = {}

    for i, sample in enumerate(ds):
        sid = sample.get("id", str(i))
        ocr_texts: List[str] = []
        for rel in sample.get("image", []):
            img_path = resolve_image_path(rel, roots)
            if not img_path:
                continue
            try:
                results = reader.readtext(img_path)
                # results: list of (bbox, text, conf)
                filtered = [text for (_b, text, conf) in results if conf >= args.min_conf]
                # 把本图的文本合并为一个字符串，也可保留为多个项
                if filtered:
                    ocr_texts.append(" ".join(filtered))
            except Exception as e:
                # 识别失败时跳过该图
                pass
        if ocr_texts:
            ocr_map[sid] = ocr_texts

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ocr_map, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR to: {args.out} (entries: {len(ocr_map)})")

if __name__ == "__main__":
    main()