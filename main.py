"""
main.py
=======
Main execution script for Apple Financial Q&A System
Orchestrates data processing, RAG, fine-tuning, and evaluation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processor import AppleReportProcessor
from rag_system import AppleRAGSystem, RAGGuardrails, load_and_initialize_rag
from finetune_system import AppleFineTunedModel, FineTuneGuardrails
from evaluation import AppleSystemEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AppleQAOrchestrator:
    """Main orchestrator for the complete pipeline"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'data_dir': './financial_data',
            'processed_dir': './processed_data',
            'models_dir': './models',
            'results_dir': './evaluation_results',
            'report_2022': 'aapl-20220924.pdf', # Corrected filename
            'report_2023': 'aapl-20230930.pdf'
        }
        
        # Create directories
        for dir_key in ['data_dir', 'processed_dir', 'models_dir', 'results_dir']:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        # System components
        self.processor = None
        self.rag_system = None
        self.ft_system = None
        self.evaluator = None
        
    def step1_process_documents(self):
        """Step 1: Process Apple annual reports"""
        print("\n" + "="*60)
        print("STEP 1: PROCESSING APPLE DOCUMENTS")
        print("="*60)
        
        # Initialize processor
        self.processor = AppleReportProcessor(self.config['data_dir'])
        
        # Check if reports exist
        report_2022_path = Path(self.config['data_dir']) / self.config['report_2022']
        report_2023_path = Path(self.config['data_dir']) / self.config['report_2023']
        
        if not report_2022_path.exists():
            logger.error(f"❌ 2022 report not found: {report_2022_path}")
            return False
        
        if not report_2023_path.exists():
            logger.error(f"❌ 2023 report not found: {report_2023_path}")
            return False
        
        # Process reports
        try:
            results = self.processor.process_reports(
                str(report_2022_path),
                str(report_2023_path)
            )
            
            print(f"✅ Processing complete!")
            print(f"   - Created {len(results['chunks'])} text chunks")
            print(f"   - Generated {len(results['qa_pairs'])} Q&A pairs")
            print(f"   - Saved to: {self.config['processed_dir']}/apple_processed_data.json")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing documents: {e}")
            return False
    
    def step2_build_rag_system(self):
        """Step 2: Build and initialize RAG system"""
        print("\n" + "="*60)
        print("STEP 2: BUILDING RAG SYSTEM")
        print("="*60)
        
        try:
            # Load processed data and initialize RAG
            self.rag_system, self.rag_guardrails, self.processed_data = load_and_initialize_rag(
                f"{self.config['processed_dir']}/apple_processed_data.json"
            )
            
            print("✅ RAG system initialized!")
            print(f"   - Indexed {len(self.processed_data['chunks'])} chunks")
            print("   - Dense + Sparse retrieval ready")
            print("   - Cross-encoder re-ranking enabled")
            
            # Test RAG with a sample question
            test_q = "What was Apple's total net sales in 2023?"
            print(f"\n🧪 Testing RAG with: '{test_q}'")
            result = self.rag_system.answer_question(test_q)
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Confidence: {result['confidence']:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building RAG system: {e}")
            return False
    
    def step3_train_finetuned_model(self, force_retrain: bool = False):
        """Step 3: Train or load fine-tuned model"""
        print("\n" + "="*60)
        print("STEP 3: FINE-TUNING MODEL (CPU)")
        print("="*60)
        
        model_path = Path(self.config['models_dir']) / "apple_tiny_finetuned"
        
        try:
            # Check if model already exists
            if model_path.exists() and not force_retrain:
                print("📂 Found existing fine-tuned model. Loading...")
                self.ft_system = AppleFineTunedModel()
                self.ft_system.load_model(str(model_path))
                print("✅ Model loaded from disk")
            else:
                print("🔧 Training new fine-tuned model on CPU...")
                print("   This may take several minutes...")
                
                # Load processed data for training
                processed_data_path = Path(self.config['processed_dir']) / "apple_processed_data.json"
                if not processed_data_path.exists():
                    logger.error(f"Processed data not found at {processed_data_path}. Run step 1 first.")
                    return False

                with open(processed_data_path, 'r') as f:
                    data = json.load(f)
                
                self.ft_system = AppleFineTunedModel()
                self.ft_system.train(data['qa_pairs'])
                self.ft_system.save_model(str(model_path))
                print("✅ Fine-tuning complete!")
            
            self.ft_guardrails = FineTuneGuardrails()
            
            # Test fine-tuned model
            test_q = "What was Apple's total net sales in 2023?"
            print(f"\n🧪 Testing Fine-tuned with: '{test_q}'")
            result = self.ft_system.generate_answer(test_q)
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Confidence: {result['confidence']:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error with fine-tuned model: {e}", exc_info=True)
            return False
    
    def step4_evaluate_systems(self):
        """Step 4: Comprehensive evaluation of both systems"""
        print("\n" + "="*60)
        print("STEP 4: EVALUATING BOTH SYSTEMS")
        print("="*60)
        
        if not self.rag_system or not self.ft_system:
            logger.error("❌ Both systems must be initialized before evaluation")
            return False
        
        try:
            # Initialize evaluator
            self.evaluator = AppleSystemEvaluator()
            
            # Evaluate RAG system
            print("\n📊 Evaluating RAG System...")
            rag_results = self.evaluator.evaluate_system(
                self.rag_system,
                'RAG',
                self.rag_guardrails
            )
            
            # Evaluate Fine-tuned system
            print("\n📊 Evaluating Fine-Tuned System...")
            ft_results = self.evaluator.evaluate_system(
                self.ft_system,
                'Fine-Tuned',
                self.ft_guardrails
            )
            
            # Compare systems
            print("\n📊 Comparing Systems...")
            self.evaluator.compare_systems(rag_results, ft_results)
            
            # Generate report
            report_path = self.evaluator.generate_report(self.config['results_dir'])
            print(f"\n✅ Evaluation complete! Report saved to: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}", exc_info=True)
            return False
    
    def run_complete_pipeline(self, skip_processing: bool = False, 
                            skip_training: bool = False):
        """Run the complete pipeline"""
        print("\n" + "🍎 "*20)
        print("APPLE FINANCIAL Q&A SYSTEM - COMPLETE PIPELINE")
        print("🍎 "*20)
        
        start_time = datetime.now()
        
        # Step 1: Process documents (unless skipped)
        if not skip_processing:
            if not self.step1_process_documents():
                print("⚠️  Document processing failed. Attempting to use existing processed data...")
        else:
            print("⏭️  Skipping document processing (using existing data)")
        
        # Step 2: Build RAG system
        if not self.step2_build_rag_system():
            print("❌ Failed to build RAG system")
            return False
        
        # Step 3: Train/Load fine-tuned model
        if not self.step3_train_finetuned_model(force_retrain=not skip_training):
            print("❌ Failed to prepare fine-tuned model")
            return False
        
        # Step 4: Evaluate both systems
        if not self.step4_evaluate_systems():
            print("❌ Failed to evaluate systems")
            return False
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print(f"Total time: {duration:.1f} seconds")
        print(f"\nNext steps:")
        print("1. Review evaluation report in ./evaluation_results/")
        print("2. Run interface: streamlit run interface.py")
        print("3. Test with custom questions using the systems")
        
        return True
    
    def interactive_qa(self):
        """Interactive Q&A mode"""
        if not self.rag_system or not self.ft_system:
            print("❌ Systems not initialized. Run the pipeline first.")
            return
        
        print("\n" + "="*60)
        print("INTERACTIVE Q&A MODE")
        print("="*60)
        print("Type 'quit' to exit, 'compare' to see both answers")
        print("Type 'rag' or 'ft' to use specific system")
        
        mode = 'compare'
        
        while True:
            print(f"\nMode: {mode.upper()}")
            question = input("❓ Your question: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'rag':
                mode = 'rag'
                print("Switched to RAG mode")
                continue
            elif question.lower() == 'ft':
                mode = 'ft'
                print("Switched to Fine-Tuned mode")
                continue
            elif question.lower() == 'compare':
                mode = 'compare'
                print("Switched to Compare mode")
                continue
            
            if not question:
                continue
            
            # Get answers based on mode
            if mode in ['rag', 'compare']:
                print("\n🔍 RAG System:")
                rag_result = self.rag_system.answer_question(question)
                rag_result = self.rag_guardrails.validate_output(rag_result)
                print(f"Answer: {rag_result['answer']}")
                print(f"Confidence: {rag_result['confidence']:.2%} | Time: {rag_result['time']:.2f}s")
            
            if mode in ['ft', 'compare']:
                print("\n🧠 Fine-Tuned System:")
                ft_result = self.ft_system.generate_answer(question)
                ft_result = self.ft_guardrails.validate_output(ft_result)
                print(f"Answer: {ft_result['answer']}")
                print(f"Confidence: {ft_result['confidence']:.2%} | Time: {ft_result['time']:.2f}s")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Apple Financial Q&A System')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip document processing (use existing)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive Q&A after pipeline')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (requires existing systems)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = AppleQAOrchestrator()
    
    if args.eval_only:
        # Load existing systems and evaluate
        orchestrator.step2_build_rag_system()
        orchestrator.step3_train_finetuned_model(force_retrain=False)
        orchestrator.step4_evaluate_systems()
    else:
        # Run complete pipeline
        success = orchestrator.run_complete_pipeline(
            skip_processing=args.skip_processing,
            skip_training=args.skip_training
        )
        
        if success and args.interactive:
            orchestrator.interactive_qa()

if __name__ == "__main__":
    main()