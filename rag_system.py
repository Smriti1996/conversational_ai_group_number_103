"""
rag_system.py
=============
RAG (Retrieval-Augmented Generation) system for Apple reports
Implements hybrid retrieval, re-ranking, and answer generation using Llama 2.
"""

import json
import time
import logging
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import streamlit as st  # Import Streamlit
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- MODIFIED: Authentication using Streamlit Secrets ---
def setup_huggingface_auth():
    """Setup Hugging Face authentication using Streamlit secrets or environment variables."""
    token = None
    # Try to get token from Streamlit secrets first (for deployed app)
    try:
        if 'HUGGINGFACE_HUB_TOKEN' in st.secrets:
            token = st.secrets['HUGGINGFACE_HUB_TOKEN']
            print("🔑 Authenticating with token from Streamlit secrets.")
    except Exception:
        # This will fail if not run in a Streamlit context, which is fine.
        pass

    # Fallback to environment variable (for local testing with .env file)
    if not token:
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token:
            print("🔑 Authenticating with token from local environment.")

    if token:
        os.environ['HUGGINGFACE_HUB_TOKEN'] = token
        try:
            from huggingface_hub import login
            # Set add_to_git_credential to False for deployment environments
            login(token=token, add_to_git_credential=False)
            print("✅ Authenticated with Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Failed to authenticate with Hugging Face: {e}")
            return False
    else:
        # This message will show if no token is found anywhere
        print("❌ HUGGINGFACE_HUB_TOKEN not found in Streamlit secrets or environment variables.")
        return False

# Authenticate before any model loading
setup_huggingface_auth()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleRAGSystem:
    """RAG system specifically tuned for Apple financial reports"""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info("Loading cross-encoder...")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        logger.info(f"Loading generator model: {generator_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model, use_fast=True)

        if self.device.type == 'cuda':
            logger.info("CUDA detected. Loading model with 4-bit quantization.")
            self.generator = AutoModelForCausalLM.from_pretrained(
                generator_model,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
            )
        else:
            model_dtype = torch.float16 if self.device.type == 'mps' else torch.float32
            logger.warning(f"Loading model on {self.device.type.upper()} in {model_dtype}. This will be slow and memory-intensive.")
            self.generator = AutoModelForCausalLM.from_pretrained(
                generator_model,
                torch_dtype=model_dtype,
            ).to(self.device)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.dense_index = None
        self.sparse_matrix = None
        self.chunks = []
        
        self.top_k_dense = 5
        self.top_k_sparse = 5
        self.rerank_top_k = 3
        
    def build_indices(self, chunks: List[Dict[str, Any]]):
        logger.info(f"Building indices for {len(chunks)} chunks...")
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info("Creating dense embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        self.dense_index.add(embeddings)
        
        logger.info("Creating sparse index...")
        self.sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"✅ Indices built successfully!")
        
    def hybrid_retrieve(self, query: str, filter_year: str = None) -> List[Dict[str, Any]]:
        expanded_query = self._expand_query(query)
        query_embedding = self.embedding_model.encode([expanded_query])
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.dense_index.search(query_embedding, min(self.top_k_dense * 2, len(self.chunks)))
        query_vector = self.tfidf_vectorizer.transform([expanded_query])
        sparse_scores = cosine_similarity(query_vector, self.sparse_matrix).flatten()
        sparse_top_indices = np.argsort(sparse_scores)[::-1][:self.top_k_sparse * 2]
        
        results = {}
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            chunk = self.chunks[idx]
            if filter_year and chunk['metadata'].get('year') != filter_year:
                continue
            chunk_id = chunk['chunk_id']
            boost = self._calculate_relevance_boost(query, chunk['text'])
            results[chunk_id] = {'chunk': chunk, 'dense_score': float(score), 'sparse_score': 0.0, 'combined_score': float(score) * 0.7 + boost}
        
        for idx in sparse_top_indices:
            chunk = self.chunks[idx]
            if filter_year and chunk['metadata'].get('year') != filter_year:
                continue
            chunk_id = chunk['chunk_id']
            score = float(sparse_scores[idx])
            boost = self._calculate_relevance_boost(query, chunk['text'])
            if chunk_id in results:
                results[chunk_id]['sparse_score'] = score
                results[chunk_id]['combined_score'] += score * 0.3 + boost
            else:
                results[chunk_id] = {'chunk': chunk, 'dense_score': 0.0, 'sparse_score': score, 'combined_score': score * 0.3 + boost}
        
        return sorted(results.values(), key=lambda x: x['combined_score'], reverse=True)[:self.top_k_dense + self.top_k_sparse]

    def _expand_query(self, query: str) -> str:
        query_lower = query.lower()
        expansions = []
        if 'revenue' in query_lower: expansions.extend(['total revenue', 'net sales', 'revenue growth', 'product revenue', 'services revenue'])
        if 'income' in query_lower: expansions.extend(['net income', 'operating income', 'income statement'])
        if 'iphone' in query_lower: expansions.extend(['iPhone sales', 'iPhone revenue', 'product sales'])
        if 'services' in query_lower: expansions.extend(['services revenue', 'App Store', 'iCloud', 'Apple Care'])
        if 'cash' in query_lower: expansions.extend(['cash flow', 'operating cash flow', 'cash equivalents'])
        if 'assets' in query_lower: expansions.extend(['total assets', 'current assets', 'balance sheet'])
        if 'equity' in query_lower: expansions.extend(['shareholders equity', 'stockholders equity', 'total equity'])
        return query + " " + " ".join(expansions) if expansions else query

    def _calculate_relevance_boost(self, query: str, text: str) -> float:
        boost = 0.0
        query_lower = query.lower()
        text_lower = text.lower()
        if 'revenue' in query_lower:
            if any(term in text_lower for term in ['billion', 'million', 'net sales']): boost += 0.3
            if 'product' in text_lower and 'revenue' in text_lower: boost += 0.2
            if 'services' in text_lower and 'revenue' in text_lower: boost += 0.2
        if 'income' in query_lower and any(term in text_lower for term in ['net income', 'operating income']): boost += 0.3
        if any(p in query_lower for p in ['iphone', 'ipad', 'mac', 'apple watch']) and any(p in text_lower for p in ['iphone', 'ipad', 'mac', 'watch']): boost += 0.4
        if any(m in query_lower for m in ['cash', 'assets', 'liabilities', 'equity']) and any(m in text_lower for m in ['cash', 'assets', 'liabilities', 'equity']): boost += 0.2
        if ('research' in query_lower or 'r&d' in query_lower) and any(t in text_lower for t in ['research', 'development', 'r&d']): boost += 0.3
        return boost

    def rerank_with_cross_encoder(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results: return []
        pairs = [[query, result['chunk']['text']] for result in results]
        ce_scores = self.cross_encoder.predict(pairs)
        for i, result in enumerate(results):
            result['ce_score'] = float(ce_scores[i])
            result['final_score'] = result['combined_score'] * 0.3 + result['ce_score'] * 0.7
        return sorted(results, key=lambda x: x['final_score'], reverse=True)[:self.rerank_top_k]

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()
        context_text = "\n\n".join([chunk['chunk']['text'] for chunk in retrieved_chunks])
        system_prompt = "You are a precise financial analyst. Answer the user's question based *only* on the provided context. Be direct and concise. If the answer is not in the context, state that the information is not available in the provided documents."
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context_text}

Question: {query} [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("[/INST]")[-1].strip()
        confidence = self._calculate_improved_confidence(query, answer, retrieved_chunks)
        sources = [{'year': c['chunk']['metadata'].get('year'), 'section': c['chunk']['metadata'].get('section'), 'score': c.get('final_score', 0.0)} for c in retrieved_chunks]
        
        return {
            'answer': answer,
            'confidence': float(confidence),
            'time': time.time() - start_time,
            'sources': sources,
            'num_chunks_used': len(retrieved_chunks),
            'method': 'Llama2-RAG'
        }

    def answer_question(self, query: str, year_filter: str = None) -> Dict[str, Any]:
        logger.info(f"Processing query: {query}")
        retrieved = self.hybrid_retrieve(query, filter_year=year_filter)
        
        if not retrieved:
            return {
                'answer': "No relevant information found in the Apple reports.",
                'confidence': 0.0,
                'time': 0.0,
                'sources': [],
                'method': 'Llama2-RAG'
            }
        
        reranked = self.rerank_with_cross_encoder(query, retrieved)
        result = self.generate_answer(query, reranked)
        
        logger.info(f"Answer generated in {result['time']:.2f}s with confidence {result['confidence']:.2f} via {result.get('method', 'N/A')}")
        return result

    def _calculate_improved_confidence(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        if not retrieved_chunks or "not available" in answer.lower():
            return 0.0
        scores = [max(0.0, r.get('final_score', 0.0)) for r in retrieved_chunks]
        avg_retrieval_score = np.mean(scores) if scores else 0.0
        has_number = 1.0 if re.search(r'\d', answer) else 0.0
        confidence = avg_retrieval_score * 0.7 + has_number * 0.3
        return max(0.0, min(confidence, 0.99))

class RAGGuardrails:
    def __init__(self):
        self.min_confidence_threshold = 0.05
        self.max_answer_length = 500
        self.valid_topics = ['revenue', 'income', 'profit', 'loss', 'expense', 'sales', 'assets', 'liabilities', 'cash', 'equity', 'shares', 'iphone', 'ipad', 'mac', 'apple watch', 'services', 'app store', 'icloud', 'apple', 'product', 'segment', 'research', 'development', 'r&d', 'operating']
        self.invalid_patterns = ['password', 'hack', 'exploit', 'confidential', 'secret', 'private key', 'api key']
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        query_lower = query.lower()
        if any(p in query_lower for p in self.invalid_patterns): return False, f"Query contains restricted term: {p}"
        if len(query) < 5: return False, "Query too short"
        if len(query) > 500: return False, "Query too long"
        return True, "Valid query"
    
    def validate_output(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if response['confidence'] < self.min_confidence_threshold:
            response['answer'] = "I don't have sufficient confidence to answer this question based on the available Apple report data. Please rephrase or ask about specific financial metrics."
            response['low_confidence'] = True
        if len(response['answer']) > self.max_answer_length:
            response['answer'] = response['answer'][:self.max_answer_length] + "..."
            response['truncated'] = True
        if len(response['answer']) < 10 or response['answer'].count(' ') < 2:
            response['answer'] = "Unable to generate a meaningful answer. Please try rephrasing your question."
            response['generation_error'] = True
        return response

def load_and_initialize_rag(processed_data_path: str = "./processed_data/apple_processed_data.json"):
    logger.info("Initializing RAG system...")
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    rag = AppleRAGSystem()
    rag.build_indices(data['chunks'])
    guardrails = RAGGuardrails()
    logger.info("✅ RAG system ready!")
    return rag, guardrails, data

if __name__ == "__main__":
    rag, guardrails, data = load_and_initialize_rag()
    test_questions = [
        "What was Apple's total net sales in 2023?",
        "What was Apple's revenue in 2022?",
        "How much cash did Apple generate from operating activities in 2023?",
        "What were Apple's research and development expenses in 2022?",
        "What was Apple's net income in 2023?",
        "What were Apple's total assets at the end of 2023?"
    ]
    
    print("\n" + "="*60)
    print("Testing RAG System with Apple Reports")
    print("="*60)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        is_valid, message = guardrails.validate_input(question)
        if not is_valid:
            print(f"❌ Invalid query: {message}")
            continue
        result = rag.answer_question(question)
        result = guardrails.validate_output(result)
        print(f"📝 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"⏱️ Time: {result['time']:.2f}s")
        print(f"📚 Sources: {len(result['sources'])} chunks used")
        print(f"⚙️ Method: {result.get('method', 'N/A')}")