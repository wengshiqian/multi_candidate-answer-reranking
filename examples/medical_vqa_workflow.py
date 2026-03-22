# medical_vqa_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForVision2Seq, AutoTokenizer
from peft import PeftModel

# ================== 直接使用 Transformers + PEFT 加载 LoRA ==================
# 你的模型配置 - 更新为正确的路径
MODEL_PATH = r'D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5'
ADAPTER_PATH = r'e:\llama-factory\saves\Qwen2.5-VL-7B-Instruct\lora\eval_2025-12-11-00-10-26-drout10'

# 全局模型缓存（避免重复加载）
_model_cache = None
_tokenizer_cache = None

def load_lora_model():
    """
    加载基础模型 + LoRA 适配器
    使用缓存避免重复加载

    注意：Qwen2.5-VL 是多模态视觉-语言模型，需要用 AutoModelForVision2Seq 加载
    """
    global _model_cache, _tokenizer_cache

    if _model_cache is not None and _tokenizer_cache is not None:
        return _model_cache, _tokenizer_cache

    print("🔄 正在加载基础模型和 LoRA 适配器...")

    try:
        # 1. 加载 tokenizer
        print(f"📂 加载 tokenizer: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )

        # 2. 加载基础模型（Qwen2.5-VL 是多模态模型，用 AutoModelForVision2Seq）
        print(f"🏗️  加载基础模型 (Vision2Seq): {MODEL_PATH}")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. 加载 LoRA 适配器
        print(f"🔌 加载 LoRA 适配器: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model.eval()  # 推理模式

        _model_cache = model
        _tokenizer_cache = tokenizer

        print("✅ 模型加载完成！")
        print(f"   基础模型: {MODEL_PATH}")
        print(f"   LoRA 适配器: {ADAPTER_PATH}")
        print(f"   推理精度: float16")

        return model, tokenizer

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise


class WorkflowState(TypedDict):
    """工作流状态"""
    image_path: str
    question: str
    modality: str
    body_part: str

    # 任务理解与控制信号
    diagnosis_terms: List[str]
    needs_pathophysiology: bool

    # 各节点输出
    findings: str  # Perceiver输出
    retrieved_knowledge: List[dict]  # Retriever输出
    draft_report: str  # Writer输出
    hallucination_check: dict  # Critic输出
    final_report: str  # Finalizer输出
    confidence_score: float


# ============== 节点1：视觉感知 (Perceiver) ==============
def perceiver_node(state: WorkflowState) -> WorkflowState:
    """
    只负责客观描述影像发现，不做诊断判断
    使用 Transformers + PEFT 加载的 LoRA 模型
    """
    try:
        import os
        # ✅ 修复：处理图像路径（支持绝对路径和相对路径）
        image_path = state['image_path']
        if not os.path.isabs(image_path):
            # 相对路径：尝试多个可能的位置
            possible_paths = [
                image_path,  # 当前目录
                os.path.join('..', image_path),  # 上一级目录
                os.path.join(os.path.dirname(__file__), '..', 'data', image_path),  # examples/../data/images/xxx
                # 新增：指向原始 llama-factory 目录的图像
                os.path.join(r'D:\weng\llama-factory\data', image_path),  # 原始 llama-factory 目录
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break

        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"⚠️  警告：找不到图像 {state['image_path']}")
            print(f"   尝试过的路径:")
            for p in possible_paths:
                print(f"     - {os.path.abspath(p)}")
            findings = f"Findings from {state['modality']} imaging at {state['body_part']}: [Mock analysis - image not found]"
        else:
            model, tokenizer = load_lora_model()

            prompt = f"""请仔细观察这张{state['modality']}影像（部位：{state['body_part']}），
描述你看到的客观影像学发现（Findings），不要给出诊断结论。

问题：{state['question']}

请按以下格式回答：
Findings: [客观描述你看到的影像特征]
            """

            # 使用 Qwen2.5-VL 的对话格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,  # 使用验证后的路径
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

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            # 解码：只保留 assistant 的回答部分（跳过对话历史）
            full_response = tokenizer.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            # 提取真实答案（移除对话模板前缀）
            # 格式: "system\n...\nuser\n...\nassistant\nACTUAL_ANSWER"
            if 'assistant' in full_response:
                findings = full_response.split('assistant')[-1].strip()
            else:
                findings = full_response.strip()

    except Exception as e:
        print(f"⚠️  警告：Perceiver 推理失败 ({str(e)})，使用模拟输出进行演示")
        findings = f"Findings from {state['modality']} imaging at {state['body_part']}: [Mock analysis - {str(e)[:50]}]"

    # 初诊医生（Diagnostician）的结构化输出
    state['findings'] = findings

    # 从初诊描述中粗略提取病变名词，并判断是否为病理生理学问题
    state['diagnosis_terms'] = _extract_diagnosis_terms(findings)
    state['needs_pathophysiology'] = _detect_pathophysiology_question(state['question'])

    return state


# ============== 节点2：知识检索 (Retriever) ==============
def retriever_node(state: WorkflowState) -> WorkflowState:
    """
    基于Findings检索相关医学知识
    """
    try:
        # 加载知识库
        import os
        rag_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_guidelines_rag.txt')
        if not os.path.exists(rag_path):
            # 备选路径
            rag_path = 'data/medical_guidelines_rag.txt'

        print(f"[住信] 读取 RAG 知识库: {rag_path}")
        with open(rag_path, 'r', encoding='utf-8') as f:
            knowledge_base = f.read()
    except FileNotFoundError:
        print(f"⚠️  警告: RAG 知识库不存在, 使用空知识库处理")
        knowledge_base = ""

    # 使用语义相似度检索
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 决定是否触发 RAG：仅在有明确病变名词或用户询问病理生理学时触发
    diagnosis_terms = state.get('diagnosis_terms', [])
    needs_pathophysiology = state.get('needs_pathophysiology', False)

    if not diagnosis_terms and not needs_pathophysiology:
        # 不需要外部知识，返回空检索结果占位
        state['retrieved_knowledge'] = []
        return state

    # 构建查询
    if diagnosis_terms:
        # 例如: "pathophysiological changes of hydronephrosis"
        base_term = diagnosis_terms[0]
        query = f"pathophysiological changes of {base_term}"
    else:
        query = state['question']

    # 可选：拼接部分影像发现，提供额外上下文
    query = f"{query} {state['findings'][:256]}"

    query_embedding = model.encode(query, convert_to_tensor=True)

    # 分块检索知识库（简化示例）
    chunks = knowledge_base.split('\n\n')
    if len(chunks) > 0 and chunks[0]:  # 检查是否有效
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

        # 计算相似度，取Top-3
        similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
        top_k = 3
        top_indices = similarities.argsort(descending=True)[:top_k]

        retrieved = [
            {
                'content': chunks[idx],
                'similarity': float(similarities[idx]),
                'source': 'RAG Knowledge Base'
            }
            for idx in top_indices
        ]
    else:
        # 空知识库的情况
        print("⚠️  知识库为空，使用空检索结果")
        retrieved = [
            {
                'content': "",
                'similarity': 0.0,
                'source': 'RAG Knowledge Base (Empty)'
            }
        ]

    state['retrieved_knowledge'] = retrieved
    return state


# ============== 节点3：报告撰写 (Writer) ==============
def writer_node(state: WorkflowState) -> WorkflowState:
    """
    结合Findings和检索知识生成初稿，包含引用
    使用 Transformers + PEFT 加载的 LoRA 模型
    """
    # 构建带引用的提示词
    references = "\n".join([
        f"[{i + 1}] {k['content'][:200]}... (Similarity: {k['similarity']:.2f})"
        for i, k in enumerate(state['retrieved_knowledge'])
    ])

    prompt = f"""你是一名经验丰富的影像科与临床医生，当前任务是回答病人的病理生理学问题。

请结合以下信息，给出一段结构化、因果链条清晰的解释性回答：
- 影像发现（Findings）：
{state['findings']}

- 检索到的医学知识（可作为背景参考，可在需要时用[1][2]标注引用）：
{references}

- 用户问题：
{state['question']}

回答要求：
1. 先用 1–2 句话概括图像中的主要异常特征。
2. 接着按照“影像改变 → 器官/组织功能改变 → 症状或风险”的顺序，解释病理生理学机制。
3. 若存在多种可能机制，指出主要假设及不确定性来源。
4. 避免做出与当前影像模态（如 Ultrasound / CT / MRI）不一致的过度推断。
    """

    try:
        import os
        # ✅ 修复：处理图像路径（支持绝对路径和相对路径）
        image_path = state['image_path']
        if not os.path.isabs(image_path):
            possible_paths = [
                image_path,
                os.path.join('..', image_path),
                os.path.join(os.path.dirname(__file__), '..', 'data', image_path),
                os.path.join(r'D:\weng\llama-factory\data', image_path),  # 原始 llama-factory 目录
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break

        if not os.path.exists(image_path):
            print(f"⚠️  警告：找不到图像 {state['image_path']}，使用模拟输出")
            draft = f"Draft report for {state['modality']} at {state['body_part']}: [Mock report - image not found]"
        else:
            model, tokenizer = load_lora_model()

            # 使用 Qwen2.5-VL 的对话格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,  # 使用验证后的路径
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

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            # 解码：只保留 assistant 的回答部分
            full_response = tokenizer.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            # 提取真实答案
            if 'assistant' in full_response:
                draft = full_response.split('assistant')[-1].strip()
            else:
                draft = full_response.strip()

    except Exception as e:
        # 如果推理失败，使用模拟输出
        print(f"⚠️  警告：Writer 推理失败 ({str(e)})，使用模拟输出进行演示")
        draft = f"""【医学影像报告】
Findings: {state['findings']}
Impression: 基于以上影像发现和相关医学知识，该患者的诊断为...
Recommendations: 建议进一步检查"""

    state['draft_report'] = draft
    return state


# ============== 节点4：主任审查 (Critic) ==============
def critic_node(state: WorkflowState) -> WorkflowState:
    """
    幻觉检测和医学正确性核验
    """
    hallucinations = []
    confidence = 1.0

    # 检测1：Findings与draft的一致性
    if not _check_consistency(state['findings'], state['draft_report']):
        hallucinations.append({
            'type': 'FINDINGS_MISMATCH',
            'description': '报告结论与影像发现不一致',
            'severity': 'HIGH'
        })
        confidence -= 0.3

    # 检测2：与RAG知识库的偏离度
    if state['retrieved_knowledge']:
        best_similarity = max([k['similarity'] for k in state['retrieved_knowledge']])
        if best_similarity < 0.3:  # 相似度过低
            hallucinations.append({
                'type': 'KNOWLEDGE_DEVIATION',
                'description': f'与标准答案相似度过低（{best_similarity:.2f}）',
                'severity': 'MEDIUM'
            })
            confidence -= 0.2

    # 检测3：医学术语合理性（基于modality和body_part）
    if not _check_terminology(state['draft_report'], state['modality'], state['body_part']):
        hallucinations.append({
            'type': 'TERMINOLOGY_ERROR',
            'description': f'检测到与{state["body_part"]}不匹配的医学术语',
            'severity': 'HIGH'
        })
        confidence -= 0.3

    state['hallucination_check'] = {
        'has_hallucination': len(hallucinations) > 0,
        'hallucinations': hallucinations,
        'confidence': max(0.0, confidence)
    }
    return state


# ============== 节点5：最终定稿 (Finalizer) ==============
def finalizer_node(state: WorkflowState) -> WorkflowState:
    """
    根据审查意见修正报告
    """
    if not state['hallucination_check']['has_hallucination']:
        # 无幻觉，直接通过
        state['final_report'] = state['draft_report']
        state['confidence_score'] = state['hallucination_check']['confidence']
    else:
        # 有幻觉，需要修正
        corrections = "\n".join([
            f"- {h['type']}: {h['description']}"
            for h in state['hallucination_check']['hallucinations']
        ])

        prompt = f"""以下报告存在问题，请修正：

原始报告：
{state['draft_report']}

检测到的问题：
{corrections}

影像发现：
{state['findings']}

请生成修正后的报告。
"""

        try:
            import os
            # ✅ 修复：处理图像路径（支持绝对路径和相对路径）
            image_path = state['image_path']
            if not os.path.isabs(image_path):
                possible_paths = [
                    image_path,
                    os.path.join('..', image_path),
                    os.path.join(os.path.dirname(__file__), '..', 'data', image_path),
                    os.path.join(r'D:\weng\llama-factory\data', image_path),  # 原始 llama-factory 目录
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break

            if not os.path.exists(image_path):
                print(f"⚠️  警告：找不到图像 {state['image_path']}，使用模拟输出")
                corrected = state['draft_report']
            else:
                model, tokenizer = load_lora_model()

                # 使用 Qwen2.5-VL 的对话格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,  # 使用验证后的路径
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

                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )

                # 解码
                full_response = tokenizer.batch_decode(
                    generated_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0]

                # 提取真实答案
                if 'assistant' in full_response:
                    corrected = full_response.split('assistant')[-1].strip()
                else:
                    corrected = full_response.strip()

        except Exception as e:
            # 如果修正失败，使用原始草稿
            print(f"⚠️  警告：Finalizer 修正失败 ({str(e)})，使用原始草稿")
            corrected = state['draft_report']

        state['final_report'] = corrected
        state['confidence_score'] = state['hallucination_check']['confidence']

    return state


# ============== 辅助函数 ==============
def _check_consistency(findings: str, report: str) -> bool:
    """检查Findings与报告的一致性"""
    # 简化实现：检查关键词是否保留
    findings_keywords = set(findings.lower().split())
    report_keywords = set(report.lower().split())
    overlap = len(findings_keywords & report_keywords) / len(findings_keywords)
    return overlap > 0.5


def _check_terminology(report: str, modality: str, body_part: str) -> bool:
    """检查医学术语合理性"""
    # 定义不合理的组合
    invalid_combinations = {
        'Brain': ['costophrenic angle', 'cardiac', 'pneumothorax'],
        'Chest': ['hydronephrosis', 'brain lesion'],
        'Abdomen': ['brain', 'pulmonary'],
    }

    report_lower = report.lower()
    for invalid_term in invalid_combinations.get(body_part, []):
        if invalid_term.lower() in report_lower:
            return False
    return True


def _extract_diagnosis_terms(text: str) -> List[str]:
    """ 
    从初诊描述中粗略抽取可能的病变名词，用于构造 RAG 查询。
    这里使用一个可扩展的短词表，实际项目中可以替换为 NER 或专业词表。
    """
    candidates = [
        "hydronephrosis",
        "aneurysm",
        "embolism",
        "infarction",
        "tumor",
        "mass",
        "lesion",
        "edema",
        "fibrosis",
    ]
    text_lower = text.lower()
    return [term for term in candidates if term in text_lower]


def _detect_pathophysiology_question(question: str) -> bool:
    """ 
    判断用户问题是否主要关注病理生理学（Why / Pathophysiology）。
    """
    q_lower = question.lower()
    triggers = [
        "why",
        "pathophysiology",
        "pathophysiological",
        "mechanism",
        "how does",
        "病理生理",
        "为什么",
    ]
    return any(trigger in q_lower for trigger in triggers)


# ============== 构建LangGraph工作流 ==============
def create_workflow():
    workflow = StateGraph(WorkflowState)

    # 添加节点
    workflow.add_node("perceiver", perceiver_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalizer", finalizer_node)

    # 定义边（顺序执行）
    workflow.set_entry_point("perceiver")
    workflow.add_edge("perceiver", "retriever")
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile()


# ============== 运行示例 ==============
if __name__ == '__main__':
    app = create_workflow()

    # 测试案例
    initial_state = {
        'image_path': 'images/pmc_129954_0.jpg',
        'question': 'What are the likely pathophysiological changes occurring in this kidney?',
        'modality': 'Ultrasound',
        'body_part': 'Abdomen'

    }

    result = app.invoke(initial_state)

    print("=== 最终报告 ===")
    print(result['final_report'])
    print(f"\n置信度：{result['confidence_score']:.2f}")
    print(f"幻觉检测：{result['hallucination_check']}")