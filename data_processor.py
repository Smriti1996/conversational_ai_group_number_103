"""
data_processor.py
=================
Module for processing Apple financial reports (2023 & 2024)
Extracts text, cleans, segments, and generates Q&A pairs
"""

import os
import re
import json
import logging
from typing import List, Dict
from pathlib import Path
import pdfplumber
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleReportProcessor:
    """Process Apple financial reports for 2023 and 2024"""
    
    def __init__(self, data_dir: str = "./financial_data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("./processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
        # Section identification patterns for Apple financial statements
        self.section_patterns = {
            'income_statement': r'(?i)(consolidated\s+statements?\s+of\s+operations|income\s+statement)',
            'balance_sheet': r'(?i)(consolidated\s+balance\s+sheets?|statement\s+of\s+financial\s+position)',
            'cash_flows': r'(?i)(consolidated\s+statements?\s+of\s+cash\s+flows?)',
            'shareholders_equity': r'(?i)(consolidated\s+statements?\s+of\s+shareholders?\s+equity)',
            'comprehensive_income': r'(?i)(consolidated\s+statements?\s+of\s+comprehensive\s+income)'
        }
    
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, str]:
        """Extract text from Apple PDF report"""
        logger.info(f"Extracting text from: {pdf_path}")
        pages_text = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text[f"page_{i+1}"] = text
            
            logger.info(f"Extracted {len(pages_text)} pages from {pdf_path}")
            return pages_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        # Remove headers and footers
        text = re.sub(r'Apple\s+Inc\.|APPLE\s+INC\.', '', text, flags=re.IGNORECASE)
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)\:\;\&]', '', text)
        return text.strip()
    
    def segment_report(self, pages_text: Dict[str, str]) -> Dict[str, str]:
        """Segment Apple report into logical sections"""
        sections = {key: '' for key in self.section_patterns.keys()}
        full_text = '\n\n'.join(pages_text.values())
        
        for section_name, pattern in self.section_patterns.items():
            matches = re.split(pattern, full_text, maxsplit=1)
            if len(matches) > 1:
                section_content = matches[-1][:15000]
                sections[section_name] = self.clean_text(section_content)
                logger.info(f"Found section: {section_name} ({len(section_content)} chars)")
        
        # If no sections found, store full text in income_statement
        if not any(sections.values()):
            sections['income_statement'] = self.clean_text(full_text[:15000])
        
        return sections
    
    def generate_qa_pairs(self, sections: Dict[str, str], year: str) -> List[Dict[str, str]]:
        """Generate all 27 Q&A pairs based on Apple financial data"""
        qa_pairs = [
            # Income Statement (10 Q/A pairs)
            {
                'question': f"What was the company's total net sales in {year}?",
                'answer': f"The company's total net sales in {year} were extracted from the income statement: {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's revenue in {year}?",
                'answer': f"The company's revenue in {year} was {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's gross margin in {year}?",
                'answer': f"The company's gross margin in {year} was {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's operating income in {year}?",
                'answer': f"The company's operating income in {year} was {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's net income in {year}?",
                'answer': f"The company's net income in {year} was {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's provision for income taxes in {year}?",
                'answer': f"The company's provision for income taxes in {year} was {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's research and development expenses in {year}?",
                'answer': f"The company's research and development expenses in {year} were {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's selling, general and administrative expenses in {year}?",
                'answer': f"The company's selling, general and administrative expenses in {year} were {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's diluted earnings per share in {year}?",
                'answer': f"The company's diluted earnings per share in {year} were {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            {
                'question': f"What was the company's basic earnings per share in {year}?",
                'answer': f"The company's basic earnings per share in {year} were {sections.get('income_statement', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'income_statement'
            },
            # Balance Sheet (7 Q/A pairs)
            {
                'question': f"What were the company's total assets at the end of {year}?",
                'answer': f"The company's total assets at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What were the company's total liabilities at the end of {year}?",
                'answer': f"The company's total liabilities at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What was the company's total shareholders' equity at the end of {year}?",
                'answer': f"The company's total shareholders' equity at the end of {year} was {sections.get('shareholders_equity', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What were the company's total current assets at the end of {year}?",
                'answer': f"The company's total current assets at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What were the company's total current liabilities at the end of {year}?",
                'answer': f"The company's total current liabilities at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What was the company's cash and cash equivalents at the end of {year}?",
                'answer': f"The company's cash and cash equivalents at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            {
                'question': f"What was the company's property, plant and equipment, net at the end of {year}?",
                'answer': f"The company's property, plant and equipment, net at the end of {year} were {sections.get('balance_sheet', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'balance_sheet'
            },
            # Cash Flows (6 Q/A pairs)
            {
                'question': f"How much cash did the company generate from operating activities in {year}?",
                'answer': f"The company generated {sections.get('cash_flows', 'not found')} from operating activities in {year}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            {
                'question': f"How much cash did the company use in investing activities in {year}?",
                'answer': f"The company used {sections.get('cash_flows', 'not found')} in investing activities in {year}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            {
                'question': f"How much cash did the company use in financing activities in {year}?",
                'answer': f"The company used {sections.get('cash_flows', 'not found')} in financing activities in {year}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            {
                'question': f"What were the company's payments for acquisition of property, plant and equipment in {year}?",
                'answer': f"The company's payments for acquisition of property, plant and equipment in {year} were {sections.get('cash_flows', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            {
                'question': f"What was the ending cash, cash equivalents and restricted cash in {year}?",
                'answer': f"The ending cash, cash equivalents and restricted cash in {year} were {sections.get('cash_flows', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            {
                'question': f"How much dividends did the company pay in {year}?",
                'answer': f"The company paid {sections.get('cash_flows', 'not found')} in dividends in {year}.",
                'year': year,
                'type': 'factual',
                'section': 'cash_flows'
            },
            # Shareholders' Equity (2 Q/A pairs)
            {
                'question': f"What was the company's total shareholders' equity ending balance in {year}?",
                'answer': f"The company's total shareholders' equity ending balance in {year} was {sections.get('shareholders_equity', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'shareholders_equity'
            },
            {
                'question': f"How much common stock did the company repurchase in {year}?",
                'answer': f"The company repurchased {sections.get('shareholders_equity', 'not found')} in common stock in {year}.",
                'year': year,
                'type': 'factual',
                'section': 'shareholders_equity'
            },
            # Comprehensive Income (2 Q/A pairs)
            {
                'question': f"What was the company's total comprehensive income in {year}?",
                'answer': f"The company's total comprehensive income in {year} was {sections.get('comprehensive_income', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'comprehensive_income'
            },
            {
                'question': f"What was the company's other comprehensive income or loss in {year}?",
                'answer': f"The company's other comprehensive income or loss in {year} was {sections.get('comprehensive_income', 'not found')}.",
                'year': year,
                'type': 'factual',
                'section': 'comprehensive_income'
            }
        ]
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs for {year}")
        return qa_pairs
    
    def create_chunks(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, any]]:
        """Create overlapping text chunks for retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) > 50:  # Minimum chunk size
                chunks.append({
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': min(i + chunk_size, len(words)),
                    'chunk_id': f"chunk_{len(chunks)}",
                    'metadata': {
                        'chunk_size': chunk_size,
                        'word_count': len(chunk_words),
                        'overlap': overlap
                    }
                })
        
        return chunks
    
    def process_reports(self, report_paths: List[str]) -> Dict:
        """Main processing function for Apple financial reports"""
        logger.info("Starting Apple financial report processing...")
        
        all_sections = {}
        all_qa_pairs = []
        all_chunks = []
        
        for report_path in report_paths:
            # Extract year from filename (assuming format like aapl-20230930.pdf)
            year_match = re.search(r'(\d{4})', report_path)
            year = year_match.group(1) if year_match else 'unknown'
            
            # Extract and process text
            pages_text = self.extract_pdf_text(report_path)
            sections = self.segment_report(pages_text)
            all_sections[year] = sections
            
            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(sections, year)
            all_qa_pairs.extend(qa_pairs)
            
            # Create chunks with metadata
            for section_name, section_text in sections.items():
                if section_text:
                    chunks = self.create_chunks(section_text, chunk_size=400, overlap=50)
                    for chunk in chunks:
                        chunk['metadata']['year'] = year
                        chunk['metadata']['section'] = section_name
                        chunk['metadata']['source'] = report_path
                    all_chunks.extend(chunks)
        
        # Save processed data
        output = {
            'sections': all_sections,
            'qa_pairs': all_qa_pairs,
            'chunks': all_chunks,
            'metadata': {
                'processed_date': datetime.now().isoformat(),
                'num_chunks': len(all_chunks),
                'num_qa_pairs': len(all_qa_pairs),
                'chunk_size': 400,
                'overlap': 50,
                'total_sections': sum(len(sections) for sections in all_sections.values())
            }
        }
        
        output_path = self.processed_dir / "apple_processed_data.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Processing complete! Saved to {output_path}")
        logger.info(f"Created {len(all_chunks)} chunks and {len(all_qa_pairs)} Q&A pairs")
        
        return output

if __name__ == "__main__":
    # Process Apple financial reports
    processor = AppleReportProcessor()
    
    # Update these paths to process BOTH files
    results = processor.process_reports([
        "./financial_data/aapl-20230930.pdf",
        "./financial_data/aapl-20220924.pdf"  # Add the 2022 file
    ])
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated {len(results['chunks'])} chunks")
    print(f"❓ Created {len(results['qa_pairs'])} Q&A pairs")
    print(f"📄 Processed {results['metadata']['total_sections']} sections")