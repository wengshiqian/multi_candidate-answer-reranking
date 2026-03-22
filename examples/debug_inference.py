#!/usr/bin/env python3
"""
调试推理质量问题
检查模型是否真的在进行推理
"""

import json
import torch

try:
    # 在 examples 目录内运行
    from medical_vqa_workflow import load_lora_model, create_workflow
except ImportError:
    # 从仓库根目录运行（需要 Python 3.3+ 的命名空间包支持）
    from examples.medical_vqa_workflow import load_lora_model, create_workflow

def test_model_inference():
    """测试模型推理"""
    print("="*80)
    print("🔍 诊断：模型推理质量检查")
    print("="*80)
    
    # 1. 测试基础模型加载
    print("\n【Test 1】加载模型...")
    try:
        model, tokenizer = load_lora_model()
        print("✅ 模型加载成功")
        print(f"   模型类型: {type(model)}")
        print(f"   模型设备: {model.device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 测试简单推理
    print("\n【Test 2】简单推理测试...")
    test_prompt = "What is the capital of France?"

    try:
        # 使用 Qwen2.5-VL 的 chat 模板（纯文本消息）
        messages = [
            {"role": "user", "content": [{"type": "text", "text": test_prompt}]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 推理成功")
        print(f"   输入: {test_prompt}")
        print(f"   输出: {response}")
    except Exception as e:
        print(f"❌ 简单推理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 测试 LangGraph 工作流
    print("\n【Test 3】LangGraph 工作流测试...")
    
    try:
        app = create_workflow()
        print("✅ LangGraph 创建成功")
        
        # 使用模拟图像路径测试
        initial_state = {
            'image_path': 'images/test.jpg',  # 模拟路径
            'question': 'What are the findings in this image?',
            'modality': 'CT',
            'body_part': 'Abdomen',
        }
        
        print("\n【运行工作流】")
        result = app.invoke(initial_state)
        
        print("✅ 工作流运行成功")
        print(f"\n【工作流输出】")
        print(f"  Findings: {result['findings'][:100]}...")
        print(f"  Draft Report: {result['draft_report'][:100]}...")
        print(f"  Final Report: {result['final_report'][:100]}...")
        print(f"  Confidence: {result['confidence_score']}")
        
        # 检查输出长度
        if len(result['final_report']) < 20:
            print("\n⚠️  警告：最终报告过短！")
            print("   可能原因：")
            print("   1. 模型未正确推理")
            print("   2. 生成的 tokens 太少")
            print("   3. 模型在回退到模拟输出")
        
    except Exception as e:
        print(f"❌ LangGraph 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 诊断建议
    print("\n【诊断建议】")
    print("""
    指标差（BLEU-4 ≈ 0）的可能原因：

    1️⃣  模型推理不工作
        → 检查 GPU 是否真的被使用（nvidia-smi）
        → 检查是否在回退到模拟输出
        → 确保 apply_chat_template 正确

    2️⃣  生成长度太短
        → 增加 max_new_tokens
        → 检查生成参数（temperature、top_p）

    3️⃣  格式问题
        → 推理输出可能不符合格式期望
        → BLEU 对精确匹配很严格

    4️⃣  验证集不匹配
        → 参考答案和生成答案格式不同
        → 参考答案可能使用了不同的语言

    建议：
    ✅ 运行此脚本，检查具体输出内容
    ✅ 如果模型输出是模拟文本，说明推理确实失败了
    ✅ 检查 langgraph.py 的异常处理日志
    """)

if __name__ == '__main__':
    test_model_inference()
