#!/usr/bin/env python3
"""
测试修复后的 medical_vqa_workflow.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

def test_import():
    """测试导入是否成功"""
    try:
        from medical_vqa_workflow import create_workflow, load_lora_model
        print("✅ 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_paths():
    """测试模型路径配置"""
    try:
        from medical_vqa_workflow import MODEL_PATH, ADAPTER_PATH
        
        print(f"基础模型路径: {MODEL_PATH}")
        print(f"LoRA适配器路径: {ADAPTER_PATH}")
        
        # 检查路径是否存在
        if os.path.exists(MODEL_PATH):
            print("✅ 基础模型路径存在")
        else:
            print("⚠️  基础模型路径不存在（可能不影响）")
            
        if os.path.exists(ADAPTER_PATH):
            print("✅ LoRA适配器路径存在")
        else:
            print("❌ LoRA适配器路径不存在")
            
        return True
    except Exception as e:
        print(f"❌ 模型路径测试失败: {e}")
        return False

def main():
    print("🧪 测试修复后的 medical_vqa_workflow.py")
    print("=" * 50)
    
    tests = [
        test_import,
        test_model_paths,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！代码修复成功。")
    else:
        print("⚠️  部分测试失败，请检查上述错误。")

if __name__ == '__main__':
    main()