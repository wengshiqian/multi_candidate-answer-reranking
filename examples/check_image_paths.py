#!/usr/bin/env python3
"""
检查验证集中的图像路径是否存在
"""

import json
import os
from pathlib import Path

validation_path = 'data/PubMedVision_VALIDATION.json'

print("=" * 80)
print("🔍 检查验证集图像路径")
print("=" * 80)

# 加载验证集
with open(validation_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\n验证集包含 {len(data)} 个样本\n")

# 检查前20个样本的图像路径
missing_count = 0
found_count = 0

for idx, item in enumerate(data[:20]):
    image_list = item['image'] if isinstance(item['image'], list) else [item['image']]
    
    for img_path in image_list:
        # 尝试多个可能的路径
        possible_paths = [
            img_path,  # 原始路径
            os.path.join('..', img_path),  # 上一级目录
            os.path.join(os.path.dirname(__file__), '..', 'data', img_path.split('/')[-1]),  # data目录
            os.path.join(os.path.dirname(__file__), img_path),  # examples目录
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 样本 {idx}: {img_path}")
                print(f"   找到: {os.path.abspath(path)}")
                found_count += 1
                found = True
                break
        
        if not found:
            print(f"❌ 样本 {idx}: {img_path}")
            print(f"   尝试过的路径:")
            for path in possible_paths:
                print(f"     - {os.path.abspath(path)}")
            missing_count += 1

print(f"\n" + "=" * 80)
print(f"找到: {found_count} / {found_count + missing_count}")
print(f"缺失: {missing_count} / {found_count + missing_count}")
print("=" * 80)

if missing_count > 0:
    print("\n⚠️  建议:")
    print("1. 检查 data/ 目录中是否存在 images/ 文件夹")
    print("2. 确保图像文件在正确的位置")
    print("3. 运行: ls -la data/images/ (在data目录下检查)")
else:
    print("\n✅ 所有图像路径都存在，可以开始评估！")
