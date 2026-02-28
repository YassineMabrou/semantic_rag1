"""
Test Script - Verify Semantic Search Logic Without Database
============================================================
This script tests the core functionality using local data.
Run this to verify everything works before connecting to the database.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    HAVE_MODEL = True
except ImportError:
    HAVE_MODEL = False
    print("[WARNING] sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")


# ==================== TEST DATA ====================

TEST_FRAGMENTS = [
    "Dosage recommandé d'alpha-amylase entre 5 et 20 ppm selon la farine",
    "La xylanase améliore l'extensibilité de la pâte et le volume du pain",
    "L'acide ascorbique agit comme agent oxydant pour renforcer le réseau gluten",
    "Dosage recommandé : 0.005% à 0.02% du poids de farine pour l'alpha-amylase",
    "La glucose oxydase (GOX) peut remplacer l'acide ascorbique comme agent oxydant",
    "Les transglutaminases améliorent la texture et la structure de la mie",
    "Température optimale d'activité enzymatique : 50-60°C pour la plupart des amylases",
    "La lipase améliore le volume du pain et la structure de la mie",
    "Dosage de xylanase recommandé : 10-50 ppm selon le type de farine",
    "L'acide ascorbique : dosage typique de 20-100 ppm pour la panification"
]

TEST_QUESTION = "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?"


# ==================== CORE FUNCTIONS ====================

def cosine_similarity(query_vec, corpus_vecs):
    """Compute cosine similarity between query and corpus vectors."""
    # Normalize
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    corpus_norm = corpus_vecs / np.clip(norms, 1e-10, None)
    query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-10)
    
    # Dot product = cosine similarity for normalized vectors
    return np.dot(corpus_norm, query_norm)


def search(query, fragments, vectors, model, top_k=3):
    """Perform semantic search."""
    # Generate query embedding
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    
    # Compute similarities
    scores = cosine_similarity(query_vec, vectors)
    
    # Sort descending
    sorted_idx = np.argsort(scores)[::-1]
    
    # Get top K
    results = []
    seen = set()
    for idx in sorted_idx:
        frag = fragments[idx]
        if frag in seen:
            continue
        seen.add(frag)
        results.append({"texte": frag, "score": float(scores[idx])})
        if len(results) >= top_k:
            break
    
    return results


def display_results(results, query):
    """Display results in challenge format."""
    print("\n" + "=" * 70)
    print("QUESTION:", query[:60] + "..." if len(query) > 60 else query)
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        print(f"\nRésultat {i}")
        print(f"  Texte: \"{r['texte']}\"")
        print(f"  Score: {r['score']:.2f}")
    
    print("\n" + "=" * 70)


# ==================== MAIN ====================

def main():
    if not HAVE_MODEL:
        print("\n[ERROR] Cannot run without sentence-transformers")
        print("Please install: pip install sentence-transformers")
        return
    
    print("=" * 70)
    print("   SEMANTIC RAG TEST - Local Mode (No Database)")
    print("=" * 70)
    
    # Load model
    print("\n[INFO] Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[INFO] Model loaded!\n")
    
    # Generate embeddings for test data
    print(f"[INFO] Generating embeddings for {len(TEST_FRAGMENTS)} fragments...")
    vectors = model.encode(TEST_FRAGMENTS, normalize_embeddings=True)
    print("[INFO] Embeddings generated!\n")
    
    # Run test search
    print("[INFO] Running test search...")
    results = search(TEST_QUESTION, TEST_FRAGMENTS, vectors, model)
    
    # Display results
    display_results(results, TEST_QUESTION)
    
    # Interactive mode
    print("\nEnter your own questions (or 'quit' to exit):\n")
    
    while True:
        try:
            query = input(">> ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            
            results = search(query, TEST_FRAGMENTS, vectors, model)
            display_results(results, query)
            
        except KeyboardInterrupt:
            break
    
    print("\nDone!")


if __name__ == "__main__":
    main()
