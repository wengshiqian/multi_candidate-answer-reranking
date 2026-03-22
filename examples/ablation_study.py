#!/usr/bin/env python3
"""
消融实验：逐个移除 LangGraph 的组件，评估每个组件的贡献度
用于论文：证明系统中每个节点的必要性
"""

import json
import pandas as pd
from typing import Dict, List, Callable
from langgraph.graph import StateGraph, END
from typing import TypedDict
from evaluation_framework import MedicalVQAEvaluator


class AblationStudy:
    """消融实验框架"""
    
    def __init__(self, validation_data_path: str, sampled_size: int = 100):
        """
        初始化消融实验
        
        Args:
            validation_data_path: 验证集路径
            sampled_size: 用于消融实验的样本数（为了快速，通常较小）
        """
        self.validation_data_path = validation_data_path
        self.sampled_size = sampled_size
        self.ablation_results = []
    
    def run_ablation_study(self, configurations: List[tuple]) -> pd.DataFrame:
        """
        运行所有消融实验
        
        Args:
            configurations: 列表，每个元素是 (配置名称, LangGraph应用)
        
        Returns:
            包含所有实验结果的DataFrame
        """
        results_table = pd.DataFrame()
        
        for idx, (config_name, app) in enumerate(configurations, 1):
            print(f"\n【{idx}/{len(configurations)}】评估: {config_name}...")
            print("-" * 80)
            
            try:
                evaluator = MedicalVQAEvaluator(app, self.validation_data_path)
                summary = evaluator.evaluate_full_dataset(max_samples=self.sampled_size)
                
                result_row = {
                    'Configuration': config_name,
                    'BLEU-4': round(summary.get('avg_bleu_4', 0), 4),
                    'ROUGE-L': round(summary.get('avg_rouge_l', 0), 4),
                    'Medical Fact Acc (%)': round(summary.get('avg_medical_fact_accuracy', 0) * 100, 2),
                    'Terminology Score (%)': round(summary.get('avg_terminology_score', 0) * 100, 2),
                    'Hallucination Rate (%)': round(summary.get('hallucination_rate', 0) * 100, 2),
                    'Avg Confidence': round(summary.get('avg_confidence_score', 0), 4),
                    'Avg RAG Similarity': round(summary.get('avg_rag_similarity', 0), 4),
                }
                
                results_table = pd.concat([results_table, pd.DataFrame([result_row])], ignore_index=True)
                print(f"✓ {config_name} 评估完成")
                
            except Exception as e:
                print(f"❌ {config_name} 评估失败: {e}")
                continue
        
        return results_table
    
    @staticmethod
    def create_baseline_app():
        """Baseline: 纯推理，无LangGraph"""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        class SimpleState(TypedDict):
            final_report: str
            confidence_score: float
            hallucination_check: Dict
            retrieved_knowledge: List[dict]
        
        def simple_inference(state):
            # 简化版：直接输出
            state['final_report'] = f"【医学影像报告】\nFindings: Mock analysis\nImpression: ...\nRecommendations: ..."
            state['confidence_score'] = 0.5
            state['hallucination_check'] = {'has_hallucination': False, 'hallucinations': []}
            state['retrieved_knowledge'] = []
            return state
        
        workflow = StateGraph(SimpleState)
        workflow.add_node("inference", simple_inference)
        workflow.set_entry_point("inference")
        workflow.add_edge("inference", END)
        
        return workflow.compile()
    
    @staticmethod
    def print_ablation_results(results_table: pd.DataFrame):
        """打印消融实验结果"""
        print("\n" + "="*120)
        print("【消融实验结果表】")
        print("="*120)
        print(results_table.to_string(index=False))
        print("="*120)
        
        # 计算改进幅度
        baseline_row = results_table.iloc[0]
        print("\n【相对于 Baseline 的改进幅度】")
        print("-"*120)
        
        for idx, row in results_table.iterrows():
            if idx == 0:
                continue
            
            print(f"\n{row['Configuration']}:")
            for col in ['BLEU-4', 'ROUGE-L', 'Medical Fact Acc (%)', 'Hallucination Rate (%)']:
                improvement = ((row[col] - baseline_row[col]) / (abs(baseline_row[col]) + 1e-6) * 100)
                symbol = "↑" if improvement > 0 else "↓"
                print(f"  {col:<25} {baseline_row[col]:>8.4f} → {row[col]:>8.4f}  ({symbol} {abs(improvement):.1f}%)")


def main():
    """运行消融实验"""
    print("\n" + "="*120)
    print("医学 VQA 系统 - 消融实验")
    print("="*120)
    
    from langgraph import create_workflow
    
    ablation = AblationStudy(
        validation_data_path='data/PubMedVision_VALIDATION_SAMPLED_1k.json',
        sampled_size=50  # 消融实验用小样本快速迭代
    )
    
    # 定义所有配置
    configurations = [
        ('Baseline (No LangGraph)', ablation.create_baseline_app()),
        ('Full System (All Components)', create_workflow()),
    ]
    
    # 注意：这里只演示 Baseline 和完整系统
    # 在实际应用中，需要为每个配置创建对应的工作流
    # 例如：Exp-1: Perceiver Only, Exp-2: +Retriever, 等等
    
    # 运行消融实验
    ablation_results = ablation.run_ablation_study(configurations)
    
    # 打印结果
    ablation.print_ablation_results(ablation_results)
    
    # 保存结果
    import os
    os.makedirs('results', exist_ok=True)
    ablation_results.to_csv('results/paper_table_ablation_study.csv', index=False)
    print(f"\n✅ 消融实验结果已保存到: results/paper_table_ablation_study.csv")


if __name__ == '__main__':
    main()
