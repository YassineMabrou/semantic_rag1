"""
Local Semantic Search - Using PDF Index
========================================
Search through your PDF documents using the locally built index.
This matches the challenge requirements exactly.

Usage:
    python search_local.py
"""

import pickle
import numpy as np
from pathlib import Path

# Paths
INDEX_FILE = Path(__file__).parent / "vector_index.pkl"

# Challenge parameters
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


def load_index():
    """Load the pre-built index."""
    
    if not INDEX_FILE.exists():
        print("[ERROR] Index file not found!")
        print("Please run first: python build_index.py")
        return None
    
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def cosine_similarity_search(query: str, index: dict, model, top_k: int = TOP_K) -> list:
    """
    Perform semantic search using cosine similarity.
    
    Challenge requirements:
    1. Receive user question
    2. Generate query embedding  
    3. Compute cosine similarity
    4. Sort by descending score
    5. Return top 3 results
    6. Display text + score
    """
    
    fragments = index["fragments"]
    vectors = index["vectors"]
    sources = index.get("sources", ["unknown"] * len(fragments))
    
    # Step 1 & 2: Generate query embedding
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    
    # Step 3: Compute cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / np.clip(norms, 1e-10, None)
    scores = np.dot(vectors_norm, query_vec)
    
    # Step 4: Sort descending
    sorted_idx = np.argsort(scores)[::-1]
    
    # Step 5: Get top K unique results
    results = []
    seen = set()
    
    for idx in sorted_idx:
        frag = fragments[idx]
        if frag in seen:
            continue
        seen.add(frag)
        
        results.append({
            "texte": frag,
            "score": float(scores[idx]),
            "source": sources[idx]
        })
        
        if len(results) >= top_k:
            break
    
    return results


def display_results(results: list, query: str):
    """Display results in exact challenge format."""
    
    print("\n" + "=" * 70)
    print(f"QUESTION: {query}")
    print("=" * 70)
    
    if not results:
        print("\nAucun résultat trouvé.")
        return
    
    for i, r in enumerate(results, 1):
        print(f"\nRésultat {i}")
        print(f"  Texte: \"{r['texte'][:200]}{'...' if len(r['texte']) > 200 else ''}\"")
        print(f"  Score: {r['score']:.2f}")
        if r.get('source'):
            print(f"  Source: {r['source']}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    
    print("=" * 70)
    print("   SEMANTIC RAG - Rose Blanche Group")
    print("   Module de Recherche Sémantique (Challenge RoboKids)")
    print("=" * 70)
    
    # Load index
    print("\n[INFO] Loading index...")
    index = load_index()
    
    if index is None:
        return
    
    print(f"[INFO] Loaded {len(index['fragments'])} fragments")
    
    # Load model
    print(f"[INFO] Loading model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print("[INFO] Ready!\n")
    
    # Challenge example question
    example_question = "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?"
    
    print(f"Exemple de question: {example_question[:60]}...")
    print("\nTapez votre question (ou 'quit' pour quitter):")
    print("Tapez 'example' pour tester avec la question d'exemple\n")
    
    while True:
        try:
            query = input(">> Question: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nAu revoir!")
                break
            
            if query.lower() == 'example':
                query = example_question
            
            if not query:
                continue
            
            # Search
            results = cosine_similarity_search(query, index, model)
            
            # Display (Challenge format)
            display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break


if __name__ == "__main__":
    main()
