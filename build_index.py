"""
PDF Parser and Embedding Builder
=================================
Extract text from PDF files and build a local vector index.
This allows testing with real data without the challenge database.

Usage:
    python build_index.py
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent / "data" / "enzymes"
INDEX_FILE = Path(__file__).parent / "vector_index.pkl"

# Model config (must match challenge requirements)
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # Smaller chunks for better precision
CHUNK_OVERLAP = 100  # More overlap for context


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,;:()%°/-]', '', text)
    
    # Fix common OCR issues
    text = text.replace('®', '').replace('™', '')
    
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        
        return clean_text(text)
    
    except ImportError:
        logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
        raise
    except Exception as e:
        logger.error(f"Error reading {pdf_path.name}: {e}")
        return ""


def chunk_text_smart(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Smart chunking that tries to split on sentence boundaries.
    Includes source name in chunk for better context.
    """
    if not text:
        return []
    
    # Add source context to beginning
    product_name = source_name.replace('.pdf', '').replace('TDS', '').strip()
    
    chunks = []
    sentences = text.replace('. ', '.|').split('|')
    
    current_chunk = f"{product_name}: "
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip() and len(current_chunk) > 50:
                chunks.append(current_chunk.strip())
            current_chunk = f"{product_name}: {sentence} "
    
    # Add last chunk
    if current_chunk.strip() and len(current_chunk) > 50:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks (fallback method)."""
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk and len(chunk) > 30:  # Only meaningful chunks
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def process_pdfs(data_dir: Path) -> Tuple[List[str], List[str]]:
    """Process all PDFs and extract chunks."""
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return [], []
    
    pdf_files = list(data_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    all_sources = []
    
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            continue
        
        # Chunk text with smart splitting
        chunks = chunk_text_smart(text, pdf_path.name)
        
        # Fallback to simple chunking if smart chunking fails
        if not chunks:
            chunks = chunk_text(text)
        
        logger.info(f"  -> {len(chunks)} chunks extracted")
        
        all_chunks.extend(chunks)
        all_sources.extend([pdf_path.name] * len(chunks))
    
    return all_chunks, all_sources


def build_embeddings(chunks: List[str]) -> np.ndarray:
    """Generate embeddings for all chunks."""
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed")
        raise
    
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    return embeddings


def save_index(chunks: List[str], sources: List[str], embeddings: np.ndarray, output_path: Path):
    """Save the index to a pickle file."""
    
    index_data = {
        "fragments": chunks,
        "sources": sources,
        "vectors": embeddings,
        "model": MODEL_NAME
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)
    
    logger.info(f"Index saved to: {output_path}")
    logger.info(f"Total fragments: {len(chunks)}")


def main():
    """Main entry point."""
    
    print("=" * 60)
    print("   PDF INDEX BUILDER - Rose Blanche Group")
    print("=" * 60)
    
    # Process PDFs
    chunks, sources = process_pdfs(DATA_DIR)
    
    if not chunks:
        logger.error("No chunks extracted. Check your PDF files.")
        return
    
    # Build embeddings
    embeddings = build_embeddings(chunks)
    
    # Save index
    save_index(chunks, sources, embeddings, INDEX_FILE)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! Index built with {len(chunks)} fragments")
    print(f"Now run: python search_local.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
