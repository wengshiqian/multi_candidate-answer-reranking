import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# 样本规模
sample_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500]
sample_labels = ['0.5K', '1.0K', '1.5K', '2.0K', '2.5K', '3.0K', '3.5K']

# 数据
baseline = [0.5450, 0.5458, 0.5474, 0.5585, 0.5625, 0.5630, 0.5625]
multi_candidate_first = [0.5648, 0.5692, 0.5789, 0.5894, 0.5926, 0.5922, 0.5893]
llm_selector = [0.5322, 0.5287, 0.5343,  0.5466,  0.5533, 0.5554, 0.5544]
distilled_reranker = [0.6100, 0.6063,  0.6123, 0.6213, 0.6205, 0.6235, 0.6271]

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

ax.plot(sample_sizes, baseline, marker='o', linewidth=1.8, markersize=5.5, label='Baseline')
ax.plot(sample_sizes, multi_candidate_first, marker='s', linewidth=1.8, markersize=5.5, label='Multi-Candidate First')
ax.plot(sample_sizes, llm_selector, marker='^', linewidth=1.8, markersize=5.5, label='LLM-Selector')
ax.plot(sample_sizes, distilled_reranker, marker='D', linewidth=1.8, markersize=5.5, label='Distilled Reranker')

# 标题和坐标轴
ax.set_title("Spatial Fact Score", fontsize=10, pad=6)
ax.set_xlabel("Sample Size", fontsize=8, labelpad=6)
ax.set_ylabel("Score", fontsize=8, labelpad=6)

# 横轴改成 0.5K 形式
ax.set_xticks(sample_sizes)
ax.set_xticklabels(sample_labels, fontsize=8)

# 纵轴保留 0.xxx 的标准写法，不要去掉 0
ax.tick_params(axis='y', labelsize=8)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# 网格
ax.grid(True, linestyle='--', alpha=0.35)

# 自动设置 y 轴范围
all_values = baseline + multi_candidate_first + llm_selector + distilled_reranker
y_min, y_max = min(all_values), max(all_values)
margin = max(0.008, (y_max - y_min) * 0.25)
ax.set_ylim(y_min - margin, y_max + margin)

# 图例放到底部，并进一步下移，避免遮挡
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.33),
    ncol=2,
    fontsize=8,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.2
)

# 关键：增大左边和底部留白
fig.subplots_adjust(
    top=0.88,
    bottom=0.40,
    left=0.20,
    right=0.97
)

plt.savefig("medical_fact_score_clean.png", dpi=300, bbox_inches='tight')
plt.show()