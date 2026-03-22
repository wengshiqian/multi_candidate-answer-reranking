"""
医学 VQA 系统评估框架
功能：计算 BLEU, ROUGE-L, 医学正确性, 幻觉率等指标
与 langgraph.py 的 LangGraph 工作流对接，自动评估系统性能
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import os

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️ 请先安装依赖: pip install nltk rouge-score")


class MedicalVQAEvaluator:
    """医学 VQA 系统评估框架"""
    
    def __init__(self, langgraph_app, validation_data_path: str):
        """
        初始化评估器
        
        Args:
            langgraph_app: LangGraph 应用实例 (从 langgraph.py 导入)
            validation_data_path: 验证数据路径
        """
        self.app = langgraph_app
        self.validation_data = self._load_data(validation_data_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.results = []
    
    def _load_data(self, path: str) -> List[Dict]:
        """加载验证数据"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_full_dataset(self, max_samples=None) -> Dict:
        """
        评估完整或部分数据集
        
        Args:
            max_samples: 最多评估多少条数据（如果为None则评估全部）
        
        Returns:
            包含各项指标的评估结果字典
        """
        data_to_eval = self.validation_data
        
        if max_samples:
            data_to_eval = data_to_eval[:max_samples]
        
        print(f"\n【开始评估】")
        print(f"评估样本数: {len(data_to_eval)}")
        
        # 逐样本评估
        for idx, item in enumerate(data_to_eval):
            if (idx + 1) % 100 == 0:
                print(f"  进度: {idx + 1}/{len(data_to_eval)}")
            
            try:
                system_output = self._run_langgraph(item)
                sample_metrics = self._evaluate_single_sample(item, system_output)
                self.results.append(sample_metrics)
            except Exception as e:
                print(f"  ❌ 样本 {idx} 处理失败: {e}")
                continue
        
        summary = self._compute_summary_statistics()
        return summary
    
    def _run_langgraph(self, item: Dict) -> Dict:
        """运行 LangGraph 工作流"""
        initial_state = {
            'image_path': item['image'][0] if isinstance(item['image'], list) else item['image'],
            'question': item['conversations'][0]['value'].replace('<image>', '').strip(),
            'modality': item['modality'],
            'body_part': item['body_part'],
        }
        
        result = self.app.invoke(initial_state)
        return result
    
    def _evaluate_single_sample(self, item: Dict, system_output: Dict) -> Dict:
        """评估单个样本"""
        reference = item['conversations'][1]['value']
        prediction = system_output['final_report']
        
        metrics = {
            'sample_id': item['id'],
            'modality': item['modality'],
            'body_part': item['body_part'],
            
            # 文本生成指标
            'bleu_4': self._compute_bleu(reference, prediction),
            'rouge_l': self._compute_rouge_l(reference, prediction),
            
            # 医学相关指标
            'medical_fact_accuracy': self._compute_medical_accuracy(prediction, item),
            'terminology_score': self._compute_terminology_score(prediction, item),
            
            # 幻觉控制指标
            'has_hallucination': system_output['hallucination_check']['has_hallucination'],
            'hallucination_count': len(system_output['hallucination_check']['hallucinations']),
            'hallucination_types': [h['type'] for h in system_output['hallucination_check']['hallucinations']],
            
            # 系统可靠性指标
            'confidence_score': system_output['confidence_score'],
            'rag_similarity': max([k['similarity'] for k in system_output['retrieved_knowledge']], default=0),
        }
        
        return metrics
    
    def _compute_bleu(self, reference: str, hypothesis: str) -> float:
        """计算 BLEU-4 分数"""
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            
            smoothing_function = SmoothingFunction().method1
            bleu = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing_function
            )
            return float(bleu)
        except:
            return 0.0
    
    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """计算 ROUGE-L 分数"""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return float(scores['rougeL'].fmeasure)
        except:
            return 0.0
    
    def _compute_medical_accuracy(self, prediction: str, item: Dict) -> float:
        """计算医学事实准确性"""
        reference = item['conversations'][1]['value']
        medical_keywords = ['diagnosis', 'finding', 'abnormal', 'normal', 'lesion', 
                           'tumor', 'inflammation', 'edema', 'consolidation']
        
        ref_keywords = [k for k in medical_keywords if k in reference.lower()]
        pred_keywords = [k for k in medical_keywords if k in prediction.lower()]
        
        if not ref_keywords:
            return 1.0
        
        overlap = len(set(ref_keywords) & set(pred_keywords))
        accuracy = overlap / len(ref_keywords)
        return min(accuracy, 1.0)
    
    def _compute_terminology_score(self, prediction: str, item: Dict) -> float:
        """计算术语规范性分数"""
        modality_terms = {
            'Computed Tomography': ['CT', 'scan', 'attenuation', 'contrast'],
            'Magnetic Resonance Imaging': ['MRI', 'T1', 'T2', 'FLAIR', 'signal'],
            'Ultrasound': ['ultrasound', 'echogenicity', 'doppler'],
            'Microscopy Images': ['microscopy', 'cellular', 'histology'],
            'Digital Photography': ['photography', 'lesion', 'surface'],
            'Endoscopy': ['endoscopy', 'mucosal', 'visualization'],
        }
        
        modality = item['modality']
        expected_terms = modality_terms.get(modality, [])
        
        if not expected_terms:
            return 1.0
        
        pred_lower = prediction.lower()
        found_terms = [t for t in expected_terms if t.lower() in pred_lower]
        
        score = len(found_terms) / len(expected_terms) if expected_terms else 1.0
        return min(score, 1.0)
    
    def _compute_summary_statistics(self) -> Dict:
        """计算汇总统计"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_samples': len(self.results),
            'evaluation_timestamp': datetime.now().isoformat(),
            
            # 文本生成指标
            'avg_bleu_4': float(df['bleu_4'].mean()),
            'avg_rouge_l': float(df['rouge_l'].mean()),
            
            # 医学相关指标
            'avg_medical_fact_accuracy': float(df['medical_fact_accuracy'].mean()),
            'avg_terminology_score': float(df['terminology_score'].mean()),
            
            # 幻觉控制指标
            'hallucination_rate': float(df['has_hallucination'].sum() / len(df)),
            'avg_hallucination_count': float(df['hallucination_count'].mean()),
            'hallucination_type_distribution': dict(pd.Series(
                [h for hlist in df['hallucination_types'] for h in hlist]
            ).value_counts()),
            
            # 系统可靠性指标
            'avg_confidence_score': float(df['confidence_score'].mean()),
            'avg_rag_similarity': float(df['rag_similarity'].mean()),
        }
        
        return summary
    
    def generate_comparison_report(self, baseline_metrics: Dict, system_metrics: Dict) -> pd.DataFrame:
        """生成 Baseline vs System 对比报告"""
        comparison_data = {
            'Metric': [
                'BLEU-4',
                'ROUGE-L',
                'Medical Fact Accuracy (%)',
                'Terminology Score (%)',
                'Hallucination Rate (%)',
                'Avg Confidence Score',
                'Avg RAG Similarity'
            ],
            'Baseline': [
                baseline_metrics.get('avg_bleu_4', 0),
                baseline_metrics.get('avg_rouge_l', 0),
                baseline_metrics.get('avg_medical_fact_accuracy', 0) * 100,
                baseline_metrics.get('avg_terminology_score', 0) * 100,
                baseline_metrics.get('hallucination_rate', 0) * 100,
                baseline_metrics.get('avg_confidence_score', 0),
                baseline_metrics.get('avg_rag_similarity', 0),
            ],
            'LangGraph System': [
                system_metrics.get('avg_bleu_4', 0),
                system_metrics.get('avg_rouge_l', 0),
                system_metrics.get('avg_medical_fact_accuracy', 0) * 100,
                system_metrics.get('avg_terminology_score', 0) * 100,
                system_metrics.get('hallucination_rate', 0) * 100,
                system_metrics.get('avg_confidence_score', 0),
                system_metrics.get('avg_rag_similarity', 0),
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # 计算提升幅度
        df['Improvement (%)'] = (
            (df['LangGraph System'] - df['Baseline']) / (df['Baseline'].abs() + 1e-6) * 100
        ).round(2)
        
        return df
    
    def export_results_to_csv(self, output_path: str):
        """导出详细结果"""
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ 详细评估结果已导出到: {output_path}")
    
    def print_summary(self, summary: Dict):
        """打印汇总统计"""
        print("\n" + "="*80)
        print("【评估汇总】")
        print("="*80)
        
        print(f"\n样本总数: {summary['total_samples']}")
        print(f"\n文本生成指标:")
        print(f"  BLEU-4:        {summary['avg_bleu_4']:.4f}")
        print(f"  ROUGE-L:       {summary['avg_rouge_l']:.4f}")
        
        print(f"\n医学相关指标:")
        print(f"  医学事实准确率: {summary['avg_medical_fact_accuracy']*100:.2f}%")
        print(f"  术语规范性:     {summary['avg_terminology_score']*100:.2f}%")
        
        print(f"\n幻觉控制效果:")
        print(f"  幻觉发生率:     {summary['hallucination_rate']*100:.2f}%")
        print(f"  平均幻觉数:     {summary['avg_hallucination_count']:.2f}")
        print(f"  幻觉类型分布:   {summary['hallucination_type_distribution']}")
        
        print(f"\n系统可靠性:")
        print(f"  平均置信度:     {summary['avg_confidence_score']:.4f}")
        print(f"  平均RAG相似度:  {summary['avg_rag_similarity']:.4f}")
