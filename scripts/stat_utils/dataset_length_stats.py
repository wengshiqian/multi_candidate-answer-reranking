import json
import os
import re
import statistics
from typing import List, Dict, Any

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def approx_token_count(text: str) -> int:
    # 轻量近似：粗略地把英文/数字/符号分词，中文按字符计。经验上对英文 1 token ≈ 4 chars。
    # 为保守起见，使用：tokens ≈ 英文词数 + 中文字符数 + 数字块 + 标点块
    if not text:
        return 0
    # 英文词
    eng_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    # 数字块（包括带单位的）
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?(?:\s?[a-zA-Z%/]+)?", text)
    # 中文字符
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    # 其他标点/符号
    puncts = re.findall(r"[^\w\s\u4e00-\u9fff]", text)
    return len(eng_words) + len(numbers) + len(zh_chars) + len(puncts)

def collect_lengths(ds: List[Dict[str, Any]]):
    img_counts = []
    text_chars = []
    text_tokens = []
    for ex in ds:
        imgs = ex.get("image", []) or []
        img_counts.append(len(imgs))
        convs = ex.get("conversations", []) or []
        # 只统计单轮 Q/A（你的数据基本是这种），拼接 human + gpt 文本
        h = ""
        g = ""
        for c in convs:
            if c.get("from") == "human":
                h += c.get("value", "")
            elif c.get("from") == "gpt":
                g += c.get("value", "")
        text = (h + "\n" + g).replace("<image>", "")  # 去掉占位符，仅计纯文本
        text_chars.append(len(text))
        text_tokens.append(approx_token_count(text))
    return img_counts, text_chars, text_tokens

def percentile(data: List[int], p: float) -> float:
    if not data:
        return 0.0
    return statistics.quantiles(data, n=100, method="inclusive")[int(p)-1]

def report(name: str, img_counts: List[int], text_chars: List[int], text_tokens: List[int]) -> Dict[str, Any]:
    total = len(text_tokens)
    if total == 0:
        return {"dataset": name, "total": 0}
    single_img = sum(1 for x in img_counts if x == 1)
    multi_img = sum(1 for x in img_counts if x >= 2)
    over_2048 = sum(1 for t in text_tokens if t > 2048)
    over_4096 = sum(1 for t in text_tokens if t > 4096)

    return {
        "dataset": name,
        "total_samples": total,
        "images": {
            "single_image_ratio": round(single_img/total, 4),
            "multi_image_ratio": round(multi_img/total, 4),
            "max_images_per_sample": max(img_counts),
        },
        "text_tokens": {
            "p50": int(statistics.median(text_tokens)),
            "p90": int(percentile(text_tokens, 90)),
            "p95": int(percentile(text_tokens, 95)),
            "p99": int(percentile(text_tokens, 99)),
            ">2048_ratio": round(over_2048/total, 4),
            ">4096_ratio": round(over_4096/total, 4),
            "max_tokens": int(max(text_tokens)),
        },
        "text_chars": {
            "p50": int(statistics.median(text_chars)),
            "p90": int(percentile(text_chars, 90)),
            "p95": int(percentile(text_chars, 95)),
            "p99": int(percentile(text_chars, 99)),
            "max_chars": int(max(text_chars)),
        },
    }

def main():
    train_path = r"d:\weng\llama-factory\data\PubMedVision_CLEANED.json"
    valid_path = r"d:\weng\llama-factory\data\PubMedVision_VALIDATION.json"
    assert os.path.exists(train_path), f"Not found: {train_path}"
    assert os.path.exists(valid_path), f"Not found: {valid_path}"

    train = load_json(train_path)
    valid = load_json(valid_path)

    tr_img, tr_chars, tr_tokens = collect_lengths(train)
    va_img, va_chars, va_tokens = collect_lengths(valid)

    train_rpt = report("PubMedVision_CLEANED(train)", tr_img, tr_chars, tr_tokens)
    valid_rpt = report("PubMedVision_VALIDATION(valid)", va_img, va_chars, va_tokens)

    print(json.dumps({"train": train_rpt, "valid": valid_rpt}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()