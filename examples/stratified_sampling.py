
"""
分层采样脚本：从验证集中按 modality-body_part 比例采样
用途：快速验证阶段（Phase 2）快速评估系统性能
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_sampling(validation_path, sample_size=1000, random_seed=42):
    """
    分层采样：保证采样数据中各 modality-body_part 组合的比例与原数据一致
    
    Args:
        validation_path: 验证集路径
        sample_size: 目标采样大小（默认1000）
        random_seed: 随机种子（保证可重现）
    
    Returns:
        采样后的数据列表
    """
    random.seed(random_seed)
    
    with open(validation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按 modality_body_part 分组
    groups = defaultdict(list)
    for idx, item in enumerate(data):
        key = f"{item['modality']}_{item['body_part']}"
        groups[key].append(idx)  # 保存索引，方便后续快速定位
    
    # 各组按比例采样
    sampled_indices = []
    total_items = len(data)
    
    for key, indices in groups.items():
        group_ratio = len(indices) / total_items
        group_sample_size = max(1, int(sample_size * group_ratio))  # 该组应该采样的数量
        
        # 从该组中随机采样
        sampled_group = random.sample(indices, min(group_sample_size, len(indices)))
        sampled_indices.extend(sampled_group)
    
    # 如果采样数量不足，补充随机采样
    if len(sampled_indices) < sample_size:
        remaining_needed = sample_size - len(sampled_indices)
        all_indices = set(range(total_items))
        available = list(all_indices - set(sampled_indices))
        sampled_indices.extend(random.sample(available, min(remaining_needed, len(available))))
    
    # 构建采样后的数据
    sampled_indices = sampled_indices[:sample_size]
    sampled_data = [data[i] for i in sampled_indices]
    
    # 统计信息
    print(f"\n【分层采样完成】")
    print(f"原始数据量: {len(data)}")
    print(f"采样数据量: {len(sampled_data)}")
    print(f"采样比例: {len(sampled_data)/len(data)*100:.1f}%")
    
    print(f"\n采样后各场景分布:")
    sampled_groups = defaultdict(int)
    for item in sampled_data:
        key = f"{item['modality']}_{item['body_part']}"
        sampled_groups[key] += 1
    
    for key in sorted(sampled_groups.keys()):
        original_count = len(groups[key])
        sampled_count = sampled_groups[key]
        print(f"  {key:<50} {sampled_count:>3} / {original_count:>4} ({sampled_count/original_count*100:>5.1f}%)")
    
    return sampled_data


def save_sampled_data(sampled_data, output_path):
    """保存采样数据到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    print(f"\n采样数据已保存到: {output_path}")


if __name__ == '__main__':
    # 配置（基于脚本位置解析仓库根目录）
    repo_root = Path(__file__).resolve().parents[1]
    validation_path = repo_root / 'data' / 'PubMedVision_VALIDATION.json'
    sample_size = 1000
    output_path = repo_root / 'data' / 'PubMedVision_VALIDATION_SAMPLED_1k.json'

    # 执行采样
    sampled_data = stratified_sampling(validation_path, sample_size=sample_size)

    # 保存采样数据
    save_sampled_data(sampled_data, output_path)

    print(f"\n✅ Phase 2 快速验证的采样数据已准备好！")