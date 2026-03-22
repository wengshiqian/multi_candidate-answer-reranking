#!/usr/bin/env python3
"""
医学VQA系统 + RAG + LangGraph集成
整合微调的Qwen2.5-VL与医学知识库的高级框架
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Annotated
from dataclasses import dataclass

import requests
from enum import Enum

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    LangGraph_Available = True
except ImportError:
    LangGraph_Available = False
    print("警告: langgraph未安装。请运行: pip install langgraph")


# ============================================================================
# 第一部分：数据结构定义
# ============================================================================

@dataclass
class MedicalDocument:
    """医学知识库文档"""
    id: str
    title: str
    content: str
    modality: str  # 影像类型: MRI, CT, X-Ray, Ultrasound等
    body_part: str  # 身体部位
    keywords: List[str]


class ProcessingState(str, Enum):
    """处理状态枚举"""
    ANALYZING = "analyzing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class VQARequest:
    """VQA请求"""
    image_paths: List[str]
    question: str
    modality: Optional[str] = None
    body_part: Optional[str] = None


@dataclass
class VQAResponse:
    """VQA响应"""
    question: str
    answer: str
    confidence: float
    retrieved_knowledge: List[str]
    processing_steps: List[str]


# ============================================================================
# 第二部分：医学知识库（模拟版）
# ============================================================================

class MedicalKnowledgeBase:
    """医学知识库管理器"""
    
    def __init__(self, knowledge_file: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            knowledge_file: 知识库JSON文件路径（可选）
        """
        self.documents: Dict[str, MedicalDocument] = {}
        self.modality_index: Dict[str, List[str]] = {}  # 按影像类型索引
        self.body_part_index: Dict[str, List[str]] = {}  # 按身体部位索引
        
        if knowledge_file and Path(knowledge_file).exists():
            self._load_from_file(knowledge_file)
        else:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认医学知识"""
        default_docs = [
            MedicalDocument(
                id="mri_001",
                title="MRI基础原理",
                content="MRI（磁共振成像）是一种无辐射的医学成像技术...",
                modality="MRI",
                body_part="General",
                keywords=["MRI", "磁共振", "成像"]
            ),
            MedicalDocument(
                id="ct_001",
                title="CT扫描临床应用",
                content="CT（计算机断层扫描）广泛用于诊断各种疾病...",
                modality="CT",
                body_part="General",
                keywords=["CT", "断层扫描", "诊断"]
            ),
            MedicalDocument(
                id="ultrasound_001",
                title="超声成像原理",
                content="超声成像利用声波反射原理进行医学诊断...",
                modality="Ultrasound",
                body_part="General",
                keywords=["超声", "成像", "诊断"]
            ),
        ]
        
        for doc in default_docs:
            self.add_document(doc)
    
    def add_document(self, doc: MedicalDocument):
        """添加文档到知识库"""
        self.documents[doc.id] = doc
        
        # 更新索引
        if doc.modality not in self.modality_index:
            self.modality_index[doc.modality] = []
        self.modality_index[doc.modality].append(doc.id)
        
        if doc.body_part not in self.body_part_index:
            self.body_part_index[doc.body_part] = []
        self.body_part_index[doc.body_part].append(doc.id)
    
    def retrieve_by_modality(self, modality: str, top_k: int = 3) -> List[MedicalDocument]:
        """按影像类型检索知识"""
        doc_ids = self.modality_index.get(modality, [])[:top_k]
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def retrieve_by_body_part(self, body_part: str, top_k: int = 3) -> List[MedicalDocument]:
        """按身体部位检索知识"""
        doc_ids = self.body_part_index.get(body_part, [])[:top_k]
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def retrieve_by_keywords(self, keywords: List[str], top_k: int = 3) -> List[MedicalDocument]:
        """按关键词检索知识"""
        scored_docs = []
        for doc in self.documents.values():
            score = sum(1 for kw in keywords if kw.lower() in doc.content.lower())
            if score > 0:
                scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def _load_from_file(self, filepath: str):
        """从文件加载知识库（可选功能）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    doc = MedicalDocument(**item)
                    self.add_document(doc)
        except Exception as e:
            print(f"加载知识库失败: {e}，使用默认知识库")
            self._initialize_default_knowledge()


# ============================================================================
# 第三部分：RAG检索器
# ============================================================================

class RAGRetriever:
    """检索增强生成（RAG）检索器"""
    
    def __init__(self, kb: MedicalKnowledgeBase):
        self.kb = kb
    
    def retrieve_context(self, 
                        question: str,
                        modality: Optional[str] = None,
                        body_part: Optional[str] = None,
                        top_k: int = 3) -> List[MedicalDocument]:
        """
        根据问题、影像类型、身体部位检索相关知识
        
        Returns:
            相关的医学文档列表
        """
        all_docs = []
        
        # 策略1：按身体部位检索
        if body_part:
            all_docs.extend(self.kb.retrieve_by_body_part(body_part, top_k // 2))
        
        # 策略2：按影像类型检索
        if modality:
            all_docs.extend(self.kb.retrieve_by_modality(modality, top_k // 2))
        
        # 策略3：按关键词检索
        keywords = question.split()[:5]  # 简单的关键词提取
        all_docs.extend(self.kb.retrieve_by_keywords(keywords, top_k // 2))
        
        # 去重并返回
        unique_docs = {doc.id: doc for doc in all_docs}
        return list(unique_docs.values())[:top_k]


# ============================================================================
# 第四部分：医学图像分析Agent（调用微调模型）
# ============================================================================

class MedicalImageAnalysisAgent:
    """医学图像分析Agent - 调用微调的Qwen2.5-VL"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        初始化Agent
        
        Args:
            api_base_url: 微调模型API地址
        """
        self.api_url = f"{api_base_url}/v1/chat/completions"
        self.processing_steps: List[str] = []
    
    def analyze_image(self, image_paths: List[str], question: str) -> Dict[str, Any]:
        """
        分析医学图像
        
        Args:
            image_paths: 医学图像路径列表
            question: 医学问题
        
        Returns:
            分析结果
        """
        self.processing_steps = []
        self.processing_steps.append("开始编码图像...")
        
        # 编码图像
        encoded_images = []
        for img_path in image_paths:
            try:
                with open(img_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    encoded_images.append(encoded)
                    self.processing_steps.append(f"✓ 编码图像: {Path(img_path).name}")
            except FileNotFoundError:
                self.processing_steps.append(f"✗ 未找到图像: {img_path}")
                continue
        
        if not encoded_images:
            return {
                "error": "未能加载任何图像",
                "steps": self.processing_steps
            }
        
        # 准备请求体
        self.processing_steps.append("准备API请求...")
        content = [{"type": "text", "text": question}]
        
        # 添加图像到请求
        for encoded_img in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}
            })
        
        payload = {
            "model": "qwen2.5-vl-7b-instruct",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
            "temperature": 0.3,  # 降低温度以提高一致性
            "top_p": 0.8
        }
        
        # 调用API
        self.processing_steps.append("调用微调模型...")
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            api_response = response.json()
            analysis = api_response['choices'][0]['message']['content']
            
            self.processing_steps.append("✓ 模型分析完成")
            return {
                "analysis": analysis,
                "confidence": 0.85,  # 可根据模型输出调整
                "steps": self.processing_steps
            }
        
        except requests.exceptions.RequestException as e:
            self.processing_steps.append(f"✗ API请求失败: {str(e)}")
            return {
                "error": f"API请求失败: {e}",
                "steps": self.processing_steps
            }


# ============================================================================
# 第五部分：LangGraph工作流编排
# ============================================================================

class MedicalVQAPipeline:
    """医学VQA完整流水线（使用LangGraph编排）"""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 knowledge_file: Optional[str] = None):
        """初始化流水线"""
        self.image_agent = MedicalImageAnalysisAgent(api_base_url)
        self.kb = MedicalKnowledgeBase(knowledge_file)
        self.rag_retriever = RAGRetriever(self.kb)
        self.processing_steps: List[str] = []
        
        if LangGraph_Available:
            self._build_graph()
        else:
            print("LangGraph不可用，使用简化流程")
    
    def _build_graph(self):
        """构建LangGraph工作流图"""
        graph = StateGraph(dict)
        
        # 定义节点
        def image_analysis_node(state):
            """步骤1: 图像分析"""
            state["status"] = ProcessingState.ANALYZING.value
            state["processing_steps"].append("【步骤1】分析医学图像...")
            
            result = self.image_agent.analyze_image(
                state["image_paths"],
                state["question"]
            )
            
            if "error" not in result:
                state["image_analysis"] = result["analysis"]
                state["processing_steps"].extend(result["steps"])
            else:
                state["error"] = result["error"]
                state["status"] = ProcessingState.ERROR.value
            
            return state
        
        def rag_retrieval_node(state):
            """步骤2: RAG检索"""
            if state.get("error"):
                return state
            
            state["status"] = ProcessingState.RETRIEVING.value
            state["processing_steps"].append("【步骤2】检索医学知识库...")
            
            docs = self.rag_retriever.retrieve_context(
                state["question"],
                modality=state.get("modality"),
                body_part=state.get("body_part"),
                top_k=3
            )
            
            state["retrieved_documents"] = docs
            state["processing_steps"].extend([
                f"✓ 检索到 {len(docs)} 篇相关文献",
                f"  - {', '.join([doc.title for doc in docs])}"
            ])
            
            return state
        
        def answer_generation_node(state):
            """步骤3: 生成最终答案"""
            if state.get("error"):
                return state
            
            state["status"] = ProcessingState.GENERATING.value
            state["processing_steps"].append("【步骤3】整合知识生成答案...")
            
            # 整合图像分析和检索到的知识
            image_analysis = state.get("image_analysis", "")
            docs = state.get("retrieved_documents", [])
            
            knowledge_context = "\n".join([
                f"- {doc.title}: {doc.content[:200]}..." 
                for doc in docs
            ])
            
            # 这里可以调用第二个LLM进行答案整合（可选）
            final_answer = f"""
基于医学图像分析和知识库：

【图像分析结果】
{image_analysis}

【相关医学知识】
{knowledge_context}

【综合诊断建议】
根据上述分析，建议：...
"""
            
            state["final_answer"] = final_answer
            state["status"] = ProcessingState.COMPLETED.value
            state["processing_steps"].append("✓ 答案生成完成")
            
            return state
        
        # 添加节点到图
        graph.add_node("image_analysis", image_analysis_node)
        graph.add_node("rag_retrieval", rag_retrieval_node)
        graph.add_node("answer_generation", answer_generation_node)
        
        # 定义边（工作流顺序）
        graph.add_edge(START, "image_analysis")
        graph.add_edge("image_analysis", "rag_retrieval")
        graph.add_edge("rag_retrieval", "answer_generation")
        graph.add_edge("answer_generation", END)
        
        self.graph = graph.compile()
    
    def process_vqa_request(self, request: VQARequest) -> VQAResponse:
        """
        处理VQA请求
        
        Returns:
            完整的VQA响应
        """
        self.processing_steps = []
        
        # 初始化状态
        initial_state = {
            "image_paths": request.image_paths,
            "question": request.question,
            "modality": request.modality,
            "body_part": request.body_part,
            "processing_steps": [],
            "status": ProcessingState.ANALYZING.value
        }
        
        if LangGraph_Available:
            # 使用LangGraph执行流水线
            self.processing_steps.append("=== 启动LangGraph流水线 ===")
            result = self.graph.invoke(initial_state)
        else:
            # 简化流程（不用LangGraph）
            self.processing_steps.append("=== 使用简化流程 ===")
            result = self._simple_pipeline(initial_state)
        
        # 返回结构化响应
        return VQAResponse(
            question=request.question,
            answer=result.get("final_answer", "处理失败"),
            confidence=result.get("confidence", 0.0),
            retrieved_knowledge=[
                doc.title for doc in result.get("retrieved_documents", [])
            ],
            processing_steps=result.get("processing_steps", [])
        )
    
    def _simple_pipeline(self, state: dict) -> dict:
        """不使用LangGraph的简化流程"""
        # 步骤1: 分析图像
        state["processing_steps"].append("【步骤1】分析医学图像...")
        result = self.image_agent.analyze_image(
            state["image_paths"],
            state["question"]
        )
        
        if "error" in result:
            state["error"] = result["error"]
            return state
        
        state["image_analysis"] = result["analysis"]
        state["processing_steps"].extend(result["steps"])
        
        # 步骤2: 检索知识库
        state["processing_steps"].append("【步骤2】检索医学知识库...")
        docs = self.rag_retriever.retrieve_context(
            state["question"],
            state.get("modality"),
            state.get("body_part")
        )
        state["retrieved_documents"] = docs
        state["processing_steps"].append(f"✓ 检索到 {len(docs)} 篇相关文献")
        
        # 步骤3: 生成答案
        state["processing_steps"].append("【步骤3】整合知识生成答案...")
        state["final_answer"] = self._synthesize_answer(
            state["image_analysis"],
            docs
        )
        state["processing_steps"].append("✓ 答案生成完成")
        state["confidence"] = 0.85
        
        return state
    
    @staticmethod
    def _synthesize_answer(image_analysis: str, docs: List[MedicalDocument]) -> str:
        """合成最终答案"""
        knowledge_context = "\n".join([
            f"- {doc.title}: {doc.content[:150]}..." 
            for doc in docs
        ])
        
        return f"""
【医学影像分析结果】
{image_analysis}

【相关医学知识】
{knowledge_context}

【综合评估】
根据上述影像分析和医学知识库的检索结果，该患者的诊断和治疗方案应当...
注：此分析仅供参考，具体诊疗方案需由医疗专业人士最终确定。
"""


# ============================================================================
# 第六部分：主程序
# ============================================================================

def main():
    """主程序示例"""
    
    print("=" * 70)
    print("医学VQA + RAG + LangGraph 集成系统")
    print("=" * 70)
    
    # 初始化流水线
    print("\n正在初始化系统...")
    pipeline = MedicalVQAPipeline(
        api_base_url="http://localhost:8000",  # 改为你的微调模型API地址
        # knowledge_file="path/to/knowledge_base.json"  # 可选
    )
    
    # 示例VQA请求
    dummy_image = "test_image.jpg"
    try:
        # 创建虚拟医学图像用于测试
        import struct
        jpeg_header = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01
        ])
        with open(dummy_image, "wb") as f:
            f.write(jpeg_header + b'\x00' * 100)
    except:
        pass
    
    request = VQARequest(
        image_paths=[dummy_image],
        question="这张医学影像显示了什么？可能的诊断是什么？",
        modality="MRI",
        body_part="Brain"
    )
    
    # 处理请求
    print("\n处理VQA请求...\n")
    response = pipeline.process_vqa_request(request)
    
    # 输出结果
    print("=" * 70)
    print("【处理步骤】")
    for i, step in enumerate(response.processing_steps, 1):
        print(f"{i}. {step}")
    
    print("\n【问题】")
    print(response.question)
    
    print("\n【答案】")
    print(response.answer)
    
    print("\n【检索到的知识】")
    for knowledge in response.retrieved_knowledge:
        print(f"  - {knowledge}")
    
    print(f"\n【置信度】{response.confidence:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
