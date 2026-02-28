"""
Module de Recherche Sémantique (RAG)
=====================================
Challenge RoboKids - Rose Blanche Group

Ce module interroge une base vectorielle PostgreSQL contenant les embeddings
des fiches techniques d'ingrédients et d'additifs pour la boulangerie/pâtisserie.

Spécifications:
- Modèle: all-MiniLM-L6-v2 (dimension 384)
- Similarité: Cosine Similarity
- Top K: 3 résultats
- Table: embeddings (id, id_document, texte_fragment, vecteur)
"""

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer


# ==================== CONFIGURATION ====================

# Paramètres de connexion PostgreSQL (fournis par les organisateurs)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "database",
    "user": "user",
    "password": "password"
}

# Paramètres imposés par le challenge
MODEL_NAME = "all-MiniLM-L6-v2"  # Modèle obligatoire
EMBEDDING_DIM = 384              # Dimension des vecteurs
TOP_K = 3                        # Nombre de résultats à retourner


# ==================== INITIALISATION ====================

print("=" * 65)
print("   MODULE DE RECHERCHE SÉMANTIQUE (RAG)")
print("   Rose Blanche Group - Challenge RoboKids")
print("=" * 65)

print(f"\n[INFO] Chargement du modèle: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("[INFO] Modèle chargé avec succès!")


# ==================== FONCTIONS ====================

def connect_database():
    """Établir la connexion à la base PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"[ERREUR] Connexion à la base de données échouée: {e}")
        return None


def load_embeddings_from_db():
    """
    Charger les embeddings depuis la table PostgreSQL.
    
    Table: embeddings
    Structure:
        - id (Primary Key)
        - id_document (int)
        - texte_fragment (text)
        - vecteur (VECTOR(384))
    """
    print("\n[INFO] Connexion à la base PostgreSQL...")
    
    conn = connect_database()
    if conn is None:
        return [], [], np.array([])
    
    try:
        cur = conn.cursor()
        
        # Requête pour récupérer tous les fragments et leurs vecteurs
        cur.execute("""
            SELECT id, id_document, texte_fragment, vecteur
            FROM embeddings
            ORDER BY id
        """)
        
        rows = cur.fetchall()
        
        if not rows:
            print("[ATTENTION] Aucun embedding trouvé dans la base.")
            return [], [], np.array([])
        
        # Extraction des données
        ids = []
        fragments = []
        vectors = []
        
        for row in rows:
            ids.append(row[0])
            fragments.append(row[2])  # texte_fragment
            
            # Parsing du vecteur (pgvector retourne une string ou liste)
            vec = row[3]
            if isinstance(vec, str):
                # Format string: "[0.1, 0.2, ...]"
                import ast
                vec = ast.literal_eval(vec)
            vectors.append(np.array(vec, dtype=np.float32))
        
        print(f"[INFO] {len(fragments)} fragments chargés depuis la base.")
        
        return fragments, np.array(vectors), ids
        
    except Exception as e:
        print(f"[ERREUR] Erreur lors du chargement: {e}")
        return [], [], np.array([])
        
    finally:
        if 'cur' in locals():
            cur.close()
        conn.close()


def generate_query_embedding(question: str) -> np.ndarray:
    """
    Étape 2: Générer l'embedding de la question utilisateur.
    
    Utilise le modèle all-MiniLM-L6-v2 pour transformer
    la question en vecteur de dimension 384.
    """
    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return embedding


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculer la similarité cosinus entre deux vecteurs.
    
    Formule: cos(θ) = (A · B) / (||A|| × ||B||)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def semantic_search(question: str, fragments: list, vectors: np.ndarray, top_k: int = TOP_K) -> list:
    """
    Recherche sémantique complète.
    
    Étapes:
    1. Recevoir la question utilisateur ✓
    2. Générer l'embedding de la question
    3. Calculer la similarité cosinus avec tous les fragments
    4. Classer les résultats par score décroissant
    5. Retourner les 3 fragments les plus pertinents
    """
    
    if len(vectors) == 0:
        return []
    
    # Étape 2: Générer l'embedding de la question
    query_embedding = generate_query_embedding(question)
    
    # Étape 3: Calculer la similarité cosinus
    scores = []
    for vec in vectors:
        score = cosine_similarity(query_embedding, vec)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Étape 4: Classer par score décroissant
    sorted_indices = np.argsort(scores)[::-1]
    
    # Étape 5: Retourner les top K résultats
    results = []
    seen_fragments = set()
    
    for idx in sorted_indices:
        fragment = fragments[idx]
        
        # Éviter les doublons
        if fragment in seen_fragments:
            continue
        seen_fragments.add(fragment)
        
        results.append({
            "texte": fragment,
            "score": float(scores[idx])
        })
        
        if len(results) >= top_k:
            break
    
    return results


def display_results(results: list):
    """
    Étape 6: Afficher les résultats.
    
    Format attendu:
        Résultat 1
        Texte : "..."
        Score : 0.XX
    """
    
    if not results:
        print("\nAucun résultat trouvé.")
        return
    
    print()
    for i, result in enumerate(results, 1):
        print(f"Résultat {i}")
        print(f"Texte : \"{result['texte']}\"")
        print(f"Score : {result['score']:.2f}")
        print()


# ==================== PROGRAMME PRINCIPAL ====================

def main():
    """Point d'entrée principal du module."""
    
    # Charger les embeddings depuis PostgreSQL
    fragments, vectors, ids = load_embeddings_from_db()
    
    if len(fragments) == 0:
        print("\n[ERREUR] Impossible de charger les données.")
        print("Vérifiez les paramètres de connexion dans DB_CONFIG.")
        return
    
    # Boucle interactive
    print("\n" + "-" * 65)
    print("Entrez votre question (ou 'quit' pour quitter)")
    print("-" * 65)
    
    while True:
        try:
            # Étape 1: Recevoir la question utilisateur
            question = input("\n>> Question: ").strip()
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\nAu revoir!")
                break
            
            if not question:
                continue
            
            # Recherche sémantique (Étapes 2-5)
            results = semantic_search(question, fragments, vectors)
            
            # Affichage (Étape 6)
            display_results(results)
            
        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break


# ==================== EXEMPLE D'UTILISATION ====================

if __name__ == "__main__":
    main()


# Exemple de question du challenge:
# "Améliorant de panification : quelles sont les quantités recommandées 
#  d'alpha-amylase, xylanase et d'Acide ascorbique ?"
#
# Résultat attendu:
# Résultat 1
# Texte : "Dosage recommandé : 0.005% à 0.02% du poids de farine."
# Score : 0.91
#
# Résultat 2
# Texte : "Alpha-amylase : utilisation entre 5 et 20 ppm selon la farine."
# Score : 0.87
#
# Résultat 3
# Texte : "Xylanase : améliore l'extensibilité de la pâte…"
# Score : 0.82
