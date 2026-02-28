"""
Simple Semantic Search - Challenge Version
============================================
Minimal implementation that matches the exact challenge requirements.
Use this for the RoboKids challenge demo.

Requirements:
- Model: all-MiniLM-L6-v2
- Similarity: Cosine
- Top K: 3
- Output: Text + Score
"""

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer


# ==================== CONFIGURATION ====================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "database",
    "user": "user",
    "password": "password"
}

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


# ==================== INITIALIZATION ====================

print("=" * 60)
print("   SEMANTIC RAG MODULE - Rose Blanche Group")
print("=" * 60)
print(f"\n[INFO] Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("[INFO] Model loaded successfully!\n")


# ==================== FUNCTIONS ====================

def load_embeddings_from_db():
    """Load all embeddings from PostgreSQL database."""
    
    print("[INFO] Connecting to PostgreSQL database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, id_document, texte_fragment, vecteur
            FROM embeddings
        """)
        
        rows = cur.fetchall()
        
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return [], [], np.array([])
        
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()
    
    ids, doc_ids, fragments, vectors = [], [], [], []
    
    for row in rows:
        ids.append(row[0])
        doc_ids.append(row[1])
        fragments.append(row[2])
        
        # Parse vector (handle both string and array formats)
        vec = row[3]
        if isinstance(vec, str):
            import ast
            vec = ast.literal_eval(vec)
        vectors.append(np.array(vec, dtype=np.float32))
    
    print(f"[INFO] Loaded {len(fragments)} fragments from database\n")
    
    return fragments, np.array(vectors) if vectors else np.array([])


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for the user query."""
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


def cosine_similarity_search(query: str, fragments: list, vectors: np.ndarray, top_k: int = TOP_K) -> list:
    """
    Perform semantic search using cosine similarity.
    
    Steps:
    1. Generate query embedding
    2. Compute cosine similarity with all stored vectors
    3. Sort by descending score
    4. Return top K results
    """
    
    if len(vectors) == 0:
        return []
    
    # Step 1: Generate query embedding
    query_vector = generate_query_embedding(query)
    
    # Step 2: Compute cosine similarity
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normalized = vectors / np.clip(norms, 1e-10, None)
    
    # Cosine similarity = dot product of normalized vectors
    scores = np.dot(vectors_normalized, query_vector)
    
    # Step 3: Sort by descending score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Step 4: Get top K unique results
    results = []
    seen = set()
    
    for idx in sorted_indices:
        fragment = fragments[idx]
        
        # Skip duplicates
        if fragment in seen:
            continue
        seen.add(fragment)
        
        results.append({
            "texte": fragment,
            "score": float(scores[idx])
        })
        
        if len(results) >= top_k:
            break
    
    return results


def display_results(results: list, query: str):
    """Display search results in the required format."""
    
    print("\n" + "=" * 60)
    print(f"QUESTION: {query}")
    print("=" * 60)
    print(f"\n>>> TOP {len(results)} RÉSULTATS <<<\n")
    
    for i, result in enumerate(results, 1):
        print(f"Résultat {i}")
        print(f"  Texte: \"{result['texte']}\"")
        print(f"  Score: {result['score']:.2f}")
        print()
    
    print("=" * 60)


# ==================== MAIN ====================

def main():
    """Main entry point."""
    
    # Load data from database
    fragments, vectors = load_embeddings_from_db()
    
    if len(fragments) == 0:
        print("[ERROR] No embeddings found in database!")
        return
    
    # Interactive loop
    print("Tapez votre question (ou 'quit' pour quitter):\n")
    
    while True:
        try:
            query = input(">> Question: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nAu revoir!")
                break
            
            if not query:
                continue
            
            # Perform search
            results = cosine_similarity_search(query, fragments, vectors)
            
            # Display results
            display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break


# Example usage for demo
def demo():
    """Run a demo with the example question from the challenge."""
    
    fragments, vectors = load_embeddings_from_db()
    
    if len(fragments) == 0:
        print("[ERROR] No embeddings found!")
        return
    
    # Challenge example question
    question = "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?"
    
    results = cosine_similarity_search(question, fragments, vectors)
    display_results(results, question)


if __name__ == "__main__":
    main()
