import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# =========================================================
# 1) 配置区：按你的实际目录改这里
# =========================================================
EVAL_ROOT = Path(r"./evaluation_reranker")

EXPERIMENT_DIRS = {
    "Baseline": "baseline_predictions",
    "Multi-Candidate First": "multi_candidate_first_predictions",
    "Prompt-Based LLM Selector": "multi_candidate_llm_selector_predictions",
    "Distilled Reranker": "distilled_reranker_predictions_predictions",
}

# 你论文里真正要做的三组比较
COMPARISONS = [
    ("Distilled Reranker", "Baseline"),
    ("Distilled Reranker", "Multi-Candidate First"),
    ("Distilled Reranker", "Prompt-Based LLM Selector"),
]

# 连续/准连续指标：Wilcoxon
CONTINUOUS_METRICS = [
    ("Medical Fact Score", "medical_fact_score"),
    ("Spatial Fact Score", "spatial_fact_score"),
    ("Finding F1", "finding_f1"),
    ("Attribute F1", "attribute_f1"),
    ("Laterality F1", "laterality_f1"),
    ("Anatomy F1", "anatomy_f1"),
]

# 二值指标：McNemar
BINARY_METRICS = [
    ("Polarity Match", "polarity_match"),
    ("Uncertainty Match", "uncertainty_match"),
]

OUTPUT_DIR = EVAL_ROOT / "statistical_significance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) 读 per_case_results.json
# =========================================================
def load_per_case_results(exp_name: str) -> list:
    exp_dir = EVAL_ROOT / EXPERIMENT_DIRS[exp_name]
    file_path = exp_dir / "per_case_results.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} is not a list.")
    return data


def build_case_index(cases: list) -> dict:
    out = {}
    for c in cases:
        cid = c.get("id")
        if cid:
            out[cid] = c
    return out


# =========================================================
# 3) 配对工具
# =========================================================
def align_cases(left_cases: list, right_cases: list):
    left_idx = build_case_index(left_cases)
    right_idx = build_case_index(right_cases)

    common_ids = sorted(set(left_idx.keys()) & set(right_idx.keys()))

    aligned_left = [left_idx[cid] for cid in common_ids]
    aligned_right = [right_idx[cid] for cid in common_ids]

    return common_ids, aligned_left, aligned_right


# =========================================================
# 4) Wilcoxon + effect size
# =========================================================
def run_wilcoxon(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    deltas = y - x

    improved = int(np.sum(deltas > 0))
    degraded = int(np.sum(deltas < 0))
    tied = int(np.sum(deltas == 0))

    nonzero = deltas[deltas != 0]

    if len(nonzero) == 0:
        return {
            "N": 0,
            "Mean Δ": float(np.mean(deltas)) if len(deltas) > 0 else np.nan,
            "Median Δ": float(np.median(deltas)) if len(deltas) > 0 else np.nan,
            "Improved": improved,
            "Degraded": degraded,
            "Tied": tied,
            "p-value": 1.0,
            "Effect size r": 0.0,
        }

    stat, p = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")

    abs_vals = np.abs(nonzero)
    ranks = rankdata(abs_vals)
    w_pos = np.sum(ranks[nonzero > 0])

    n = len(nonzero)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    z = 0.0 if sd_w == 0 else (w_pos - mean_w) / sd_w
    r = abs(z) / math.sqrt(n)

    return {
        "N": int(n),
        "Mean Δ": float(np.mean(deltas)),
        "Median Δ": float(np.median(deltas)),
        "Improved": improved,
        "Degraded": degraded,
        "Tied": tied,
        "p-value": float(p),
        "Effect size r": float(r),
    }


# =========================================================
# 5) McNemar
# =========================================================
def run_mcnemar(x, y):
    both_correct = 0
    x_correct_y_wrong = 0
    x_wrong_y_correct = 0
    both_wrong = 0

    for a, b in zip(x, y):
        a = int(a)
        b = int(b)

        if a == 1 and b == 1:
            both_correct += 1
        elif a == 1 and b == 0:
            x_correct_y_wrong += 1
        elif a == 0 and b == 1:
            x_wrong_y_correct += 1
        else:
            both_wrong += 1

    table = np.array([
        [both_correct, x_correct_y_wrong],
        [x_wrong_y_correct, both_wrong]
    ])

    result = mcnemar(table, exact=False, correction=True)

    return {
        "Improved Count": int(x_wrong_y_correct),
        "Degraded Count": int(x_correct_y_wrong),
        "Both Correct": int(both_correct),
        "Both Wrong": int(both_wrong),
        "p-value": float(result.pvalue),
    }


# =========================================================
# 6) 多重比较校正
# =========================================================
def apply_holm(df, p_col="p-value"):
    if df.empty:
        df["Corrected p"] = []
        df["Significant"] = []
        return df

    reject, corrected_p, _, _ = multipletests(df[p_col].values, alpha=0.05, method="holm")
    df["Corrected p"] = corrected_p
    df["Significant"] = ["Yes" if p < 0.05 else "No" for p in corrected_p]
    return df


# =========================================================
# 7) 主分析
# =========================================================
def analyze():
    all_cases = {}
    for exp_name in EXPERIMENT_DIRS:
        all_cases[exp_name] = load_per_case_results(exp_name)

    continuous_rows = []
    binary_rows = []

    for right_name, left_name in COMPARISONS:
        # 这里按“right - left”算提升
        # 即 Distilled - Baseline / Distilled - First / Distilled - LLM
        left_cases = all_cases[left_name]
        right_cases = all_cases[right_name]

        common_ids, aligned_left, aligned_right = align_cases(left_cases, right_cases)

        comparison_label = f"{right_name} vs {left_name}"
        print(f"[INFO] {comparison_label}: overlap = {len(common_ids)}")

        # 连续指标
        for metric_name, key in CONTINUOUS_METRICS:
            valid_pairs = [
                (l, r) for l, r in zip(aligned_left, aligned_right)
                if key in l and key in r
            ]
            if not valid_pairs:
                continue

            x = [l[key] for l, r in valid_pairs]
            y = [r[key] for l, r in valid_pairs]

            res = run_wilcoxon(x, y)
            continuous_rows.append({
                "Comparison": comparison_label,
                "Metric": metric_name,
                **res
            })

        # 二值指标
        for metric_name, key in BINARY_METRICS:
            valid_pairs = [
                (l, r) for l, r in zip(aligned_left, aligned_right)
                if key in l and key in r
            ]
            if not valid_pairs:
                continue

            x = [l[key] for l, r in valid_pairs]
            y = [r[key] for l, r in valid_pairs]

            res = run_mcnemar(x, y)
            binary_rows.append({
                "Comparison": comparison_label,
                "Metric": metric_name,
                **res
            })

    continuous_df = pd.DataFrame(continuous_rows)
    binary_df = pd.DataFrame(binary_rows)

    continuous_df = apply_holm(continuous_df, p_col="p-value")
    binary_df = apply_holm(binary_df, p_col="p-value")

    if not continuous_df.empty:
        continuous_df = continuous_df[
            [
                "Comparison", "Metric", "N",
                "Mean Δ", "Median Δ",
                "Improved", "Degraded", "Tied",
                "p-value", "Corrected p", "Effect size r", "Significant"
            ]
        ]
        continuous_df.to_csv(
            OUTPUT_DIR / "continuous_metrics_significance.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if not binary_df.empty:
        binary_df = binary_df[
            [
                "Comparison", "Metric",
                "Improved Count", "Degraded Count",
                "Both Correct", "Both Wrong",
                "p-value", "Corrected p", "Significant"
            ]
        ]
        binary_df.to_csv(
            OUTPUT_DIR / "binary_metrics_significance.csv",
            index=False,
            encoding="utf-8-sig"
        )

    write_report(continuous_df, binary_df)

    print("\n=== Continuous Metrics Significance ===")
    if not continuous_df.empty:
        print(continuous_df.to_string(index=False))
    else:
        print("No continuous metrics found.")

    print("\n=== Binary Metrics Significance ===")
    if not binary_df.empty:
        print(binary_df.to_string(index=False))
    else:
        print("No binary metrics found.")

    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


# =========================================================
# 8) 文本报告
# =========================================================
def write_report(continuous_df: pd.DataFrame, binary_df: pd.DataFrame):
    report_file = OUTPUT_DIR / "significance_report.txt"
    lines = []

    lines.append("=" * 140)
    lines.append("Statistical Significance Analysis Report")
    lines.append("=" * 140)
    lines.append("")

    lines.append("[Continuous / Quasi-continuous Metrics: Wilcoxon signed-rank test]")
    if continuous_df.empty:
        lines.append("No continuous metrics results.")
    else:
        header = (
            f"{'Comparison':<55}"
            f"{'Metric':<22}"
            f"{'N':<8}"
            f"{'MeanΔ':<12}"
            f"{'p-value':<14}"
            f"{'Corrected p':<14}"
            f"{'Effect r':<12}"
            f"{'Sig':<8}"
        )
        lines.append(header)
        lines.append("-" * 140)

        for _, row in continuous_df.iterrows():
            lines.append(
                f"{row['Comparison']:<55}"
                f"{row['Metric']:<22}"
                f"{int(row['N']):<8}"
                f"{row['Mean Δ']:<12.6f}"
                f"{row['p-value']:<14.6g}"
                f"{row['Corrected p']:<14.6g}"
                f"{row['Effect size r']:<12.4f}"
                f"{row['Significant']:<8}"
            )

    lines.append("")
    lines.append("[Binary Metrics: McNemar test]")
    if binary_df.empty:
        lines.append("No binary metrics results.")
    else:
        header = (
            f"{'Comparison':<55}"
            f"{'Metric':<22}"
            f"{'Improved':<12}"
            f"{'Degraded':<12}"
            f"{'p-value':<14}"
            f"{'Corrected p':<14}"
            f"{'Sig':<8}"
        )
        lines.append(header)
        lines.append("-" * 140)

        for _, row in binary_df.iterrows():
            lines.append(
                f"{row['Comparison']:<55}"
                f"{row['Metric']:<22}"
                f"{int(row['Improved Count']):<12}"
                f"{int(row['Degraded Count']):<12}"
                f"{row['p-value']:<14.6g}"
                f"{row['Corrected p']:<14.6g}"
                f"{row['Significant']:<8}"
            )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    analyze()