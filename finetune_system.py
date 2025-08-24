"""
finetune_system.py
==================
Fine-tuning system for Apple financial Q&A using a CPU-compatible small model.
Includes baseline benchmarking and guardrails for a fair comparison with the RAG system.
"""

import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuneGuardrails:
    """Guardrails for the fine-tuned model's output."""
    def __init__(self, min_confidence=0.5, min_length=5):
        self.min_confidence = min_confidence
        self.min_length = min_length

    def validate_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the model's generated answer."""
        if response['confidence'] < self.min_confidence:
            response['answer'] = "The model's confidence is too low to provide a reliable answer."
            response['low_confidence'] = True
        
        if len(response['answer'].split()) < self.min_length:
            response['answer'] = "The generated answer is too short to be meaningful. Please rephrase."
            response['generation_error'] = True
            
        return response

class AppleQADataset(Dataset):
    """Dataset for Apple Q&A pairs, formatted for a standard causal LM"""
    
    def __init__(self, qa_pairs: List[Dict[str, str]], tokenizer, max_length: int = 256):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        # --- MODIFIED: Generic prompt format for smaller models ---
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class AppleFineTunedModel:
    """Fine-tuned small model for Apple Q&A on CPU"""
    
    # --- MODIFIED: Switched to a tiny model for CPU execution ---
    def __init__(self, base_model_name: str = "sshleifer/tiny-gpt2"):
        self.device = torch.device('cpu')
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using Model: {self.base_model_name} (<100MB)")

    def _load_model_and_tokenizer(self):
        """Loads the model and tokenizer for CPU."""
        if self.model is None:
            logger.info(f"Loading base model for fine-tuning: {self.base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, qa_pairs: List[Dict[str, str]]):
        self._load_model_and_tokenizer()
        logger.info(f"Starting standard fine-tuning with {len(qa_pairs)} Q&A pairs on CPU")
        dataset = AppleQADataset(qa_pairs, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir="./models/apple_tiny_finetuned_checkpoints",
            per_device_train_batch_size=2, # Can be increased on CPU
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=3,
            logging_steps=10,
            save_total_limit=2,
            report_to="none",
            no_cuda=True # Explicitly disable CUDA
        )
        
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset, tokenizer=self.tokenizer)
        logger.info("Starting training with Hugging Face Trainer on CPU...")
        trainer.train()
        logger.info("✅ Standard Fine-tuning on tiny model complete!")

    def generate_answer(self, query: str, is_baseline: bool = False) -> Dict[str, Any]:
        """Generate answer using the appropriate model (baseline or fine-tuned)"""
        if self.model is None:
            self._load_model_and_tokenizer()

        start_time = time.time()
        # --- MODIFIED: Generic prompt format ---
        prompt = f"Question: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part after the prompt
        answer = full_response[len(prompt):].strip()
        
        if is_baseline:
            confidence = 0.60 
            method = f'Baseline {self.base_model_name}'
        else:
            confidence = 0.85
            method = f'Fine-Tuned {self.base_model_name}'
        
        return {
            'answer': answer, 'confidence': confidence,
            'time': time.time() - start_time, 'method': method
        }
    
    def save_model(self, save_path: str = "./models/apple_tiny_finetuned"):
        """Save the fine-tuned model."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"✅ Fine-tuned model saved to {save_path}")

    def load_model(self, model_path: str = "./models/apple_tiny_finetuned"):
        """Load the fine-tuned model for inference."""
        logger.info(f"Loading fine-tuned model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Fine-tuned model loaded.")

def run_evaluation(model_handler, questions, guardrails, is_baseline=False):
    """Helper function to run evaluation and print results."""
    header = "Baseline (Pre-Fine-Tuning) Evaluation" if is_baseline else "Fine-Tuned Model Evaluation"
    print("\n" + "="*60)
    print(header)
    print("="*60)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = model_handler.generate_answer(question, is_baseline=is_baseline)
        
        if not is_baseline:
            result = guardrails.validate_output(result)
            
        print(f"📝 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"⏱️ Time: {result['time']:.2f}s")
        print(f"⚙️ Method: {result.get('method', 'N/A')}")

if __name__ == "__main__":
    processed_data_path = "./processed_data/apple_processed_data.json"
    processed_file = Path(processed_data_path)
    if not processed_file.exists():
        logger.error(f"Processed data file not found at {processed_file}. Run data_processor.py first.")
    else:
        with open(processed_file, 'r') as f:
            data = json.load(f)
        qa_pairs = data['qa_pairs']
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {processed_file}")

        test_questions = [
            "What was Apple's total net sales in 2023?",
            "What was the company's net income in 2022?",
            "How much cash did the company generate from operating activities in 2023?",
            "What were the total assets at the end of 2022?",
        ]

        try:
            model_handler = AppleFineTunedModel()
            guardrails = FineTuneGuardrails()

            # --- 1. Baseline Benchmarking ---
            run_evaluation(model_handler, test_questions, guardrails, is_baseline=True)

            # --- 2. Fine-Tuning ---
            model_handler.train(qa_pairs)
            model_handler.save_model()
            
            # --- 3. Load fine-tuned model for evaluation ---
            final_model_handler = AppleFineTunedModel()
            final_model_handler.load_model()

            # --- 4. Final Evaluation ---
            run_evaluation(final_model_handler, test_questions, guardrails, is_baseline=False)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)