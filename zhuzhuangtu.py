import matplotlib.pyplot as plt
import numpy as np

# ===== 1. 你的真实数据改这里 =====
methods = [
    "Baseline",
    "MC-First",
    "LLM Selector",
    "Distilled\nReranker",
    "Oracle"
]

factscore = [0.548, 0.561, 0.544, 0.579, 0.619]
spatial = [0.563, 0.589, 0.554, 0.627, 0.616]
findingf1 = [0.474, 0.513, 0.482, 0.527, 0.615]

# ===== 2. 基础设置 =====
x = np.arange(len(methods))
width = 0.23

fig, ax = plt.subplots(figsize=(10, 6))

# ===== 3. 画柱子 =====
bars1 = ax.bar(x - width, factscore, width, label='FactScore')
bars2 = ax.bar(x, spatial, width, label='Spatial')
bars3 = ax.bar(x + width, findingf1, width, label='FindingF1')

# ===== 4. 添加数值标注 =====
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 向上偏移3个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# ===== 5. 坐标轴和标题 =====
ax.set_xlabel("Methods", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Comparison of Core Metrics Across Different Methods", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylim(0, 0.85)

# ===== 6. 图例和网格 =====
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)

# ===== 7. 自动调整布局并保存 =====
plt.tight_layout()
plt.savefig("main_metric_bar_chart.png", dpi=300, bbox_inches='tight')
plt.show()