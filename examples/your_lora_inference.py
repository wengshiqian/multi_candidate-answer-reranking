"""
使用 LoRA 适配器进行推理
直接对接 langgraph.py 中的 Perceiver 和 Writer 节点
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import yaml

# ==================== 全局模型缓存 ====================
_model_cache = None
_tokenizer_cache = None


def _resolve_dtype(infer_dtype: Optional[str]) -> torch.dtype:
    if not infer_dtype:
        return torch.float16
    infer_dtype = str(infer_dtype).lower()
    if infer_dtype in ("fp16", "float16"):
        return torch.float16
    if infer_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _load_yaml_config(path: str = r'd:\weng\llama-factory\examples\inference\qwen2_5vl.yaml') -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception as e:
        print(f"⚠️ 警告：读取 YAML 失败（{e}），使用内置默认值")
        return {}


def load_model_and_tokenizer(force_reload=False):
    """
    加载基础模型 + LoRA 适配器
    使用缓存避免重复加载（节省内存和时间）
    """
    global _model_cache, _tokenizer_cache
    
    if _model_cache is not None and not force_reload:
        return _model_cache, _tokenizer_cache
    
    print("🔄 正在加载基础模型和 LoRA 适配器...")
    
    # 从 YAML 读取配置
    cfg = _load_yaml_config()
    model_path = cfg.get('model_name_or_path') or r'D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5'
    adapter_path = cfg.get('adapter_name_or_path')  # 允许为 None（无 LoRA）
    trust_remote_code = bool(cfg.get('trust_remote_code', True))
    dtype = _resolve_dtype(cfg.get('infer_dtype', 'float16'))
    
    # 1. 加载基础模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )
    
    # 2. 加载 LoRA 适配器（若配置了）
    if adapter_path:
        print(f"🔗 应用 LoRA 适配器：{adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=dtype,
            device_map="auto"
        )
    else:
        print("ℹ️ 未配置 adapter_name_or_path，使用基础模型进行推理。")
    
    model.eval()  # 推理模式
    
    _model_cache = model
    _tokenizer_cache = tokenizer
    
    print("✅ 模型加载完成！")
    print(f"   基础模型: {model_path}")
    if adapter_path:
        print(f"   LoRA 适配器: {adapter_path}")
    print(f"   推理精度 dtype: {dtype}")
    
    return model, tokenizer


def generate_findings(
    image_path: str,
    prompt: str,
    model: Optional[object] = None
) -> str:
    """
    Perceiver 节点：生成影像发现
    
    Args:
        image_path: 图像路径
        prompt: 提示词
        model: 如果传入 None，使用全局缓存模型
    
    Returns:
        影像发现字符串
    """
    if model is None:
        model, tokenizer = load_model_and_tokenizer()
    else:
        _, tokenizer = load_model_and_tokenizer()
    
    # Qwen2.5-VL 的对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # 构建输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # 解码
    response = tokenizer.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return response


def generate_report(
    image_path: str,
    prompt: str,
    model: Optional[object] = None
) -> str:
    """
    Writer 节点：生成医学影像报告
    
    Args:
        image_path: 图像路径
        prompt: 包含 Findings 和 RAG 知识的提示词
        model: 如果传入 None，使用全局缓存模型
    
    Returns:
        医学影像报告字符串
    """
    if model is None:
        model, tokenizer = load_model_and_tokenizer()
    else:
        _, tokenizer = load_model_and_tokenizer()
    
    # 与 generate_findings 逻辑相同，但用于报告生成
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # 报告可能更长
            temperature=0.5,  # 报告应该更严谨
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return response


# ==================== 快速测试 ====================
if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    
    # 测试 generate_findings
    test_prompt = "请描述这张医学影像的客观发现，不要给诊断结论。"
    result = generate_findings(
        image_path="images/pmc_129954_0.jpg",
        prompt=test_prompt
    )
    print(f"\n测试输出:\n{result}")
