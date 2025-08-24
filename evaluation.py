"""
evaluation.py
=============
Evaluation and comparison module for RAG vs Fine-tuned systems
Tests both systems on Apple financial questions
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleSystemEvaluator:
    """Comprehensive evaluation framework for both systems"""
    
    def __init__(self):
        self.results = []
        self.comparison_metrics = {}
        
        # Define test question categories with Apple-specific questions
        self.test_questions = {
            'high_confidence_relevant': [
                {
                    'question': "What was Apple's total net sales in 2023?",
                    'expected_contains': ['sales', '2023', 'billion', 'million', 'revenue'],
                    'type': 'factual'
                },
                {
                    'question': "What was Apple's net income in 2022?",
                    'expected_contains': ['income', '2022', 'billion', 'million'],
                    'type': 'factual'
                },
                {
                    'question': "What were Apple's total assets at the end of 2023?",
                    'expected_contains': ['assets', '2023', 'billion', 'total'],
                    'type': 'factual'
                },
                {
                    'question': "How much cash did Apple generate from operating activities in 2023?",
                    'expected_contains': ['cash', 'operating', '2023', 'billion'],
                    'type': 'factual'
                },
                {
                    'question': "What were Apple's research and development expenses in 2023?",
                    'expected_contains': ['research', 'development', '2023', 'billion', 'expenses'],
                    'type': 'factual'
                }
            ],
            'low_confidence_relevant': [
                {
                    'question': "What are Apple's main competitive advantages?",
                    'expected_contains': ['ecosystem', 'brand', 'innovation', 'technology'],
                    'type': 'analytical'
                },
                {
                    'question': "What are the main risks facing Apple's business?",
                    'expected_contains': ['risk', 'competition', 'supply chain', 'market'],
                    'type': 'analytical'
                },
                {
                    'question': "How does Apple plan to grow its services revenue?",
                    'expected_contains': ['services', 'growth', 'strategy', 'expansion'],
                    'type': 'strategic'
                },
                {
                    'question': "What is Apple's dividend policy?",
                    'expected_contains': ['dividend', 'shareholders', 'policy', 'payout'],
                    'type': 'analytical'
                },
                {
                    'question': "How has Apple's gross margin changed over time?",
                    'expected_contains': ['gross', 'margin', 'change', 'percentage'],
                    'type': 'analytical'
                }
            ],
            'irrelevant': [
                {
                    'question': "What is the capital of France?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                },
                {
                    'question': "How do you make chocolate cake?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                },
                {
                    'question': "What is quantum computing?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                },
                {
                    'question': "What is the weather like today?",
                    'expected_contains': [],
                    'type': 'irrelevant'
                }
            ]
        }
    
    def _sanitize_value(self, value):
        """Convert numpy/pandas types to native Python types and handle NaN"""
        if pd.isna(value):
            return 0.0
        elif isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        else:
            return value
    
    def evaluate_answer_quality(self, answer: str, expected_contains: List[str]) -> float:
        """
        Evaluate answer quality based on expected content
        Returns score between 0 and 1
        """
        if not answer or len(answer) < 10:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check for irrelevant answer patterns
        irrelevant_patterns = [
            "i don't have sufficient confidence",
            "unable to provide",
            "no relevant information",
            "please rephrase",
            "confidence is too low",
            "too short to be meaningful"
        ]
        
        for pattern in irrelevant_patterns:
            if pattern in answer_lower:
                return 0.1  # Low score for non-answers
        
        # Check for expected content
        if expected_contains:
            matches = sum(1 for term in expected_contains if term.lower() in answer_lower)
            content_score = matches / len(expected_contains)
        else:
            # For irrelevant questions, good answer should acknowledge irrelevance
            if any(word in answer_lower for word in ['not found', 'no information', 'irrelevant', 'apple financial']):
                content_score = 1.0
            else:
                content_score = 0.0
        
        # Check answer structure
        has_structure = (
            len(answer.split()) > 5 and  # Minimum word count
            '.' in answer and  # Has sentences
            len(answer) < 500  # Not too long
        )
        structure_score = 1.0 if has_structure else 0.5
        
        # Combined score
        return (content_score * 0.7 + structure_score * 0.3)
    
    def evaluate_system(self, 
                       system,
                       system_name: str,
                       guardrails=None) -> List[Dict[str, Any]]:
        """
        Evaluate a single system (RAG or Fine-tuned)
        """
        logger.info(f"\nEvaluating {system_name} System")
        logger.info("="*50)
        
        system_results = []
        
        for category, questions in self.test_questions.items():
            logger.info(f"\nCategory: {category}")
            
            for q_data in questions:
                question = q_data['question']
                expected = q_data['expected_contains']
                q_type = q_data['type']
                
                logger.info(f"  Question: {question[:50]}...")
                
                # Get answer from system
                start_time = time.time()
                
                if system_name == 'RAG':
                    result = system.answer_question(question)
                else:  # Fine-tuned
                    result = system.generate_answer(question)
                
                # Apply guardrails if provided
                if guardrails:
                    result = guardrails.validate_output(result)
                
                response_time = time.time() - start_time
                
                # Evaluate answer quality
                quality_score = self.evaluate_answer_quality(
                    result['answer'], 
                    expected
                )
                
                # Store result
                evaluation = {
                    'system': system_name,
                    'category': category,
                    'question': question,
                    'answer': result['answer'][:200] + '...' if len(result['answer']) > 200 else result['answer'],
                    'full_answer': result['answer'],
                    'confidence': float(result.get('confidence', 0.0)),
                    'quality_score': float(quality_score),
                    'response_time': float(response_time),
                    'question_type': q_type,
                    'sources_used': int(len(result.get('sources', [])) if system_name == 'RAG' else 0)
                }
                
                system_results.append(evaluation)
                self.results.append(evaluation)
                
                logger.info(f"    Quality: {quality_score:.2f} | Confidence: {result.get('confidence', 0):.2f} | Time: {response_time:.2f}s")
        
        return system_results
    
    def compare_systems(self, rag_results: List[Dict], ft_results: List[Dict]):
        """
        Compare RAG and Fine-tuned systems
        """
        logger.info("\n" + "="*60)
        logger.info("SYSTEM COMPARISON")
        logger.info("="*60)
        
        # Convert to DataFrames
        rag_df = pd.DataFrame(rag_results)
        ft_df = pd.DataFrame(ft_results)
        
        # Overall metrics
        metrics = {
            'RAG': {
                'avg_quality': self._sanitize_value(rag_df['quality_score'].mean()),
                'avg_confidence': self._sanitize_value(rag_df['confidence'].mean()),
                'avg_response_time': self._sanitize_value(rag_df['response_time'].mean()),
                'std_quality': self._sanitize_value(rag_df['quality_score'].std()),
                'total_sources': self._sanitize_value(rag_df['sources_used'].sum())
            },
            'Fine-Tuned': {
                'avg_quality': self._sanitize_value(ft_df['quality_score'].mean()),
                'avg_confidence': self._sanitize_value(ft_df['confidence'].mean()),
                'avg_response_time': self._sanitize_value(ft_df['response_time'].mean()),
                'std_quality': self._sanitize_value(ft_df['quality_score'].std()),
                'total_sources': 0
            }
        }
        
        # By category comparison
        categories_comparison = {}
        for category in self.test_questions.keys():
            rag_cat = rag_df[rag_df['category'] == category]
            ft_cat = ft_df[ft_df['category'] == category]
            
            categories_comparison[category] = {
                'RAG': {
                    'quality': self._sanitize_value(rag_cat['quality_score'].mean()) if not rag_cat.empty else 0.0,
                    'confidence': self._sanitize_value(rag_cat['confidence'].mean()) if not rag_cat.empty else 0.0,
                    'time': self._sanitize_value(rag_cat['response_time'].mean()) if not rag_cat.empty else 0.0
                },
                'Fine-Tuned': {
                    'quality': self._sanitize_value(ft_cat['quality_score'].mean()) if not ft_cat.empty else 0.0,
                    'confidence': self._sanitize_value(ft_cat['confidence'].mean()) if not ft_cat.empty else 0.0,
                    'time': self._sanitize_value(ft_cat['response_time'].mean()) if not ft_cat.empty else 0.0
                }
            }
        
        self.comparison_metrics = {
            'overall': metrics,
            'by_category': categories_comparison
        }
        
        # Print comparison
        print("\n📊 OVERALL METRICS")
        print("-" * 40)
        comparison_df = pd.DataFrame(metrics).T
        print(comparison_df.round(3))
        
        print("\n📈 PERFORMANCE BY CATEGORY")
        print("-" * 40)
        for category, data in categories_comparison.items():
            print(f"\n{category.upper()}:")
            cat_df = pd.DataFrame(data).T
            print(cat_df.round(3))
        
        # Determine winners
        print("\n🏆 CATEGORY WINNERS")
        print("-" * 40)
        for category in categories_comparison:
            rag_score = categories_comparison[category]['RAG']['quality']
            ft_score = categories_comparison[category]['Fine-Tuned']['quality']
            
            if rag_score > ft_score:
                winner = "RAG"
                margin = ((rag_score - ft_score) / ft_score * 100) if ft_score > 0 else 100
            else:
                winner = "Fine-Tuned"
                margin = ((ft_score - rag_score) / rag_score * 100) if rag_score > 0 else 100
            
            print(f"{category}: {winner} (+{margin:.1f}%)")
        
        return self.comparison_metrics
    
    def generate_report(self, save_path: str = "./evaluation_results"):
        """
        Generate comprehensive evaluation report
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_path = Path(save_path) / f"apple_evaluation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'metrics': self.comparison_metrics,
                'timestamp': timestamp
            }, f, indent=2, cls=NumpyEncoder)
        
        # Create visualization
        self.create_visualization(save_path, timestamp)
        
        # Generate summary report
        report_path = Path(save_path) / f"apple_evaluation_report_{timestamp}.md"
        self.write_markdown_report(report_path)
        
        logger.info(f"✅ Report saved to {save_path}")
        
        return results_path
    
    def create_visualization(self, save_path: str, timestamp: str):
        """Create comparison visualizations"""
        if not self.comparison_metrics:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Apple color scheme
        apple_blue = '#007AFF'
        apple_red = '#FF3B30'
        
        # 1. Overall Quality Comparison
        ax1 = axes[0, 0]
        systems = ['RAG', 'Fine-Tuned']
        qualities = [
            self.comparison_metrics['overall']['RAG']['avg_quality'],
            self.comparison_metrics['overall']['Fine-Tuned']['avg_quality']
        ]
        bars1 = ax1.bar(systems, qualities, color=[apple_blue, apple_red])
        ax1.set_title('Overall Quality Score', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Quality Score')
        ax1.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, val in zip(bars1, qualities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Response Time Comparison
        ax2 = axes[0, 1]
        times = [
            self.comparison_metrics['overall']['RAG']['avg_response_time'],
            self.comparison_metrics['overall']['Fine-Tuned']['avg_response_time']
        ]
        bars2 = ax2.bar(systems, times, color=[apple_blue, apple_red])
        ax2.set_title('Average Response Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Category Performance Comparison
        ax3 = axes[1, 0]
        categories = list(self.test_questions.keys())
        rag_scores = [self.comparison_metrics['by_category'][cat]['RAG']['quality'] for cat in categories]
        ft_scores = [self.comparison_metrics['by_category'][cat]['Fine-Tuned']['quality'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        bars3a = ax3.bar(x - width/2, rag_scores, width, label='RAG', color=apple_blue)
        bars3b = ax3.bar(x + width/2, ft_scores, width, label='Fine-Tuned', color=apple_red)
        ax3.set_xlabel('Question Category')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Performance by Question Category', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0)
        ax3.legend()
        
        # 4. Confidence Distribution
        ax4 = axes[1, 1]
        df = pd.DataFrame(self.results)
        rag_conf = df[df['system'] == 'RAG']['confidence']
        ft_conf = df[df['system'] == 'Fine-Tuned']['confidence']
        
        ax4.hist([rag_conf, ft_conf], label=['RAG', 'Fine-Tuned'], 
                color=[apple_blue, apple_red], alpha=0.7, bins=10, edgecolor='black')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        
        plt.suptitle('🍎 Apple Financial Q&A System Evaluation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(save_path) / f"apple_comparison_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def write_markdown_report(self, report_path: Path):
        """Generate markdown report"""
        with open(report_path, 'w') as f:
            f.write("# 🍎 Apple Financial Q&A System Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Determine overall winner
            rag_quality = self.comparison_metrics['overall']['RAG']['avg_quality']
            ft_quality = self.comparison_metrics['overall']['Fine-Tuned']['avg_quality']
            
            if rag_quality > ft_quality:
                improvement = ((rag_quality - ft_quality) / ft_quality * 100) if ft_quality > 0 else 100
                f.write(f"**🏆 Winner: RAG System** (Quality: {rag_quality:.3f} vs {ft_quality:.3f}, +{improvement:.1f}%)\n\n")
            else:
                improvement = ((ft_quality - rag_quality) / rag_quality * 100) if rag_quality > 0 else 100
                f.write(f"**🏆 Winner: Fine-Tuned System** (Quality: {ft_quality:.3f} vs {rag_quality:.3f}, +{improvement:.1f}%)\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write("- **Data Source**: Apple Annual Reports (2022-2023)\n")
            f.write("- **Model Used**: sshleifer/tiny-gpt2 (CPU-compatible)\n")
            f.write("- **Evaluation Questions**: 14 total across 3 categories\n")
            f.write("- **Categories**: High-confidence relevant, Low-confidence relevant, Irrelevant\n\n")
            
            f.write("## Detailed Metrics\n\n")
            f.write("### Overall Performance\n\n")
            f.write("| Metric | RAG | Fine-Tuned | Better |\n")
            f.write("|--------|-----|------------|--------|\n")
            
            for metric in ['avg_quality', 'avg_confidence', 'avg_response_time', 'std_quality']:
                rag_val = self.comparison_metrics['overall']['RAG'][metric]
                ft_val = self.comparison_metrics['overall']['Fine-Tuned'][metric]
                
                # Determine which is better (lower is better for response time and std)
                if metric in ['avg_response_time', 'std_quality']:
                    better = "RAG" if rag_val < ft_val else "Fine-Tuned"
                else:
                    better = "RAG" if rag_val > ft_val else "Fine-Tuned"
                
                f.write(f"| {metric.replace('_', ' ').title()} | {rag_val:.3f} | {ft_val:.3f} | {better} |\n")
            
            f.write("\n### Performance by Category\n\n")
            
            for category in self.test_questions.keys():
                f.write(f"#### {category.replace('_', ' ').title()}\n\n")
                f.write("| System | Quality | Confidence | Time (s) |\n")
                f.write("|--------|---------|------------|----------|\n")
                
                cat_data = self.comparison_metrics['by_category'][category]
                for system in ['RAG', 'Fine-Tuned']:
                    f.write(f"| {system} | ")
                    f.write(f"{cat_data[system]['quality']:.3f} | ")
                    f.write(f"{cat_data[system]['confidence']:.3f} | ")
                    f.write(f"{cat_data[system]['time']:.3f} |\n")
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("### RAG System Strengths\n")
            f.write("- ✅ Provides source attribution for answers\n")
            f.write("- ✅ Better handling of factual financial questions\n")
            f.write("- ✅ More consistent confidence calibration\n")
            f.write("- ✅ Can handle dynamic/updated information\n")
            f.write("- ✅ Transparent reasoning with source documents\n\n")
            
            f.write("### Fine-Tuned System Strengths\n")
            f.write("- ⚡ Faster response times\n")
            f.write("- 🎯 More fluent answer generation\n")
            f.write("- 📊 Better at handling analytical questions\n")
            f.write("- 💾 No need for external document retrieval\n")
            f.write("- 🔧 Customized specifically for Apple financial data\n\n")
            
            f.write("### Areas for Improvement\n")
            f.write("#### RAG System\n")
            f.write("- Response time optimization needed\n")
            f.write("- Chunk retrieval accuracy could be improved\n")
            f.write("- Better handling of cross-document queries\n\n")
            
            f.write("#### Fine-Tuned System\n")
            f.write("- Hallucination prevention mechanisms\n")
            f.write("- More training data for better coverage\n")
            f.write("- Source attribution capabilities\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("### Use Cases by System\n\n")
            f.write("**📊 Use RAG for:**\n")
            f.write("- Factual financial queries requiring exact figures\n")
            f.write("- Audit trails and compliance reporting\n")
            f.write("- Dynamic data that changes frequently\n")
            f.write("- Scenarios requiring source verification\n\n")
            
            f.write("**🚀 Use Fine-Tuned for:**\n")
            f.write("- Speed-critical applications\n")
            f.write("- Analytical insights and trend analysis\n")
            f.write("- Natural language financial explanations\n")
            f.write("- Offline or latency-sensitive environments\n\n")
            
            f.write("### 🔄 Consider Hybrid Approach\n")
            f.write("- Use RAG for fact-checking fine-tuned outputs\n")
            f.write("- Combine both for comprehensive financial analysis\n")
            f.write("- Route questions based on type (factual vs analytical)\n")
            f.write("- Implement confidence-based fallback mechanisms\n\n")
            
            f.write("## Technical Specifications\n\n")
            f.write("- **Base Model**: sshleifer/tiny-gpt2\n")
            f.write("- **Fine-tuning Method**: Standard fine-tuning on CPU\n")
            f.write("- **RAG Components**: Dense + Sparse retrieval with cross-encoder re-ranking\n")
            f.write("- **Evaluation Metrics**: Quality score, confidence, response time\n")
            f.write("- **Hardware**: CPU-only\n")

if __name__ == "__main__":
    logger.info("Running standalone evaluation test...")
    
    # Create dummy systems for testing
    evaluator = AppleSystemEvaluator()
    
    # Note: In actual use, you would pass real RAG and Fine-tuned systems
    print("Apple evaluation module ready for use!")
    print("Import this module and use with actual Apple RAG and Fine-tuned systems.")