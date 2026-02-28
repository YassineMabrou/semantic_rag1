"""
Semantic RAG Module - Rose Blanche Group
=========================================
A semantic search module for querying a vector database of bakery/pastry 
technical documentation using RAG (Retrieval-Augmented Generation).

Challenge: RoboKids 84 Explorers
Organization: STE AGRO MELANGE TECHNOLOGIE - ROSE BLANCHE Group

Features:
- Semantic search using all-MiniLM-L6-v2 embeddings
- Cosine similarity ranking
- PostgreSQL/pgvector integration
- Beautiful CLI interface with rich formatting
- Caching for improved performance
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

from config import db_config, model_config, search_config


# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==================== CONSTANTS ====================

CACHE_FILE = Path(__file__).parent / "vector_index.pkl"

# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# ==================== DATA STRUCTURES ====================

@dataclass
class SearchResult:
    """Represents a single search result."""
    rank: int
    fragment: str
    score: float
    document_id: Optional[int] = None

    def __str__(self) -> str:
        return f"[Score: {self.score:.2f}] {self.fragment[:100]}..."


# ==================== EMBEDDING MODEL ====================

class EmbeddingModel:
    """Singleton wrapper for the sentence transformer model."""
    
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if EmbeddingModel._model is None:
            logger.info(f"Loading embedding model: {model_config.model_name}")
            EmbeddingModel._model = SentenceTransformer(model_config.model_name)
            logger.info("Model loaded successfully")

    @property
    def model(self) -> SentenceTransformer:
        return EmbeddingModel._model

    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=model_config.normalize_embeddings
        )[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=model_config.normalize_embeddings
        )


# ==================== DATABASE OPERATIONS ====================

class VectorDatabase:
    """Handles all database operations for the vector store."""

    def __init__(self):
        self.ids: List[int] = []
        self.fragments: List[str] = []
        self.document_ids: List[int] = []
        self.vectors: np.ndarray = np.array([])
        self._loaded = False

    def connect(self) -> psycopg2.extensions.connection:
        """Create a database connection."""
        try:
            conn = psycopg2.connect(**db_config.to_dict())
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise ConnectionError(f"Could not connect to database: {e}")

    def load_from_db(self) -> bool:
        """Load all embeddings from the database."""
        logger.info("Connecting to PostgreSQL database...")
        
        try:
            conn = self.connect()
            cur = conn.cursor()

            cur.execute("""
                SELECT id, id_document, texte_fragment, vecteur
                FROM embeddings
                ORDER BY id
            """)

            rows = cur.fetchall()
            
            if not rows:
                logger.warning("No embeddings found in database")
                return False

            self.ids = []
            self.document_ids = []
            self.fragments = []
            vectors_list = []

            for row in rows:
                self.ids.append(row[0])
                self.document_ids.append(row[1])
                self.fragments.append(row[2])

                # Handle different vector formats from pgvector
                vec = row[3]
                if isinstance(vec, str):
                    # Parse string representation
                    import ast
                    vec = ast.literal_eval(vec)
                vectors_list.append(np.array(vec, dtype=np.float32))

            self.vectors = np.array(vectors_list)
            self._loaded = True
            
            logger.info(f"Loaded {len(self.fragments)} fragments from database")
            return True

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def save_cache(self) -> bool:
        """Save embeddings to local cache file."""
        try:
            cache_data = {
                "ids": self.ids,
                "document_ids": self.document_ids,
                "fragments": self.fragments,
                "vectors": self.vectors
            }
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved to {CACHE_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False

    def load_cache(self) -> bool:
        """Load embeddings from local cache file."""
        if not CACHE_FILE.exists():
            return False

        try:
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            
            self.ids = cache_data["ids"]
            self.document_ids = cache_data.get("document_ids", [None] * len(self.ids))
            self.fragments = cache_data["fragments"]
            self.vectors = cache_data["vectors"]
            self._loaded = True
            
            logger.info(f"Loaded {len(self.fragments)} fragments from cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def load(self, use_cache: bool = True) -> bool:
        """Load embeddings from cache or database."""
        if use_cache and self.load_cache():
            return True
        
        success = self.load_from_db()
        if success and use_cache:
            self.save_cache()
        return success

    @property
    def is_loaded(self) -> bool:
        return self._loaded and len(self.fragments) > 0

    def __len__(self) -> int:
        return len(self.fragments)


# ==================== SEMANTIC SEARCH ENGINE ====================

class SemanticSearchEngine:
    """Main semantic search engine implementing cosine similarity search."""

    def __init__(self, database: VectorDatabase):
        self.db = database
        self.model = EmbeddingModel()

    def preprocess_query(self, query: str) -> str:
        """Clean and preprocess the user query."""
        # Remove extra whitespace
        query = " ".join(query.split())
        # Strip leading/trailing whitespace
        query = query.strip()
        return query

    def compute_cosine_similarity(
        self, 
        query_vector: np.ndarray, 
        corpus_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all corpus vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Since embeddings are normalized, this simplifies to dot product.
        """
        # Normalize corpus vectors (in case they aren't already)
        norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
        corpus_normalized = corpus_vectors / np.clip(norms, 1e-10, None)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        query_normalized = query_vector / max(query_norm, 1e-10)
        
        # Compute dot product (= cosine similarity for normalized vectors)
        similarities = np.dot(corpus_normalized, query_normalized)
        
        return similarities

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search for the given query.

        Args:
            query: User question in natural language
            top_k: Number of results to return (default: from config)
            min_score: Minimum similarity score threshold
            document_id: Filter results by document ID

        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not self.db.is_loaded:
            logger.error("Database not loaded. Call db.load() first.")
            return []

        # Apply defaults from config
        top_k = top_k or search_config.top_k
        min_score = min_score or search_config.min_similarity_threshold

        # Preprocess query
        query = self.preprocess_query(query)
        if not query:
            logger.warning("Empty query provided")
            return []

        logger.info(f"Searching for: '{query[:50]}...'")

        # Generate query embedding
        query_vector = self.model.encode(query)

        # Compute similarities
        scores = self.compute_cosine_similarity(query_vector, self.db.vectors)

        # Get sorted indices (descending order)
        sorted_indices = np.argsort(scores)[::-1]

        # Build results
        results = []
        seen_fragments = set()

        for idx in sorted_indices:
            # Apply document filter
            if document_id is not None and self.db.document_ids[idx] != document_id:
                continue

            # Apply minimum score threshold
            if scores[idx] < min_score:
                continue

            fragment = self.db.fragments[idx]

            # Deduplicate if enabled
            if search_config.deduplicate_results:
                if fragment in seen_fragments:
                    continue
                seen_fragments.add(fragment)

            results.append(SearchResult(
                rank=len(results) + 1,
                fragment=fragment,
                score=float(scores[idx]),
                document_id=self.db.document_ids[idx]
            ))

            if len(results) >= top_k:
                break

        logger.info(f"Found {len(results)} results")
        return results


# ==================== CLI INTERFACE ====================

class CLI:
    """Beautiful command-line interface for the semantic search module."""

    def __init__(self, engine: SemanticSearchEngine):
        self.engine = engine

    def print_header(self):
        """Print the application header."""
        print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              SEMANTIC RAG - RECHERCHE INTELLIGENTE               ║
║                                                                  ║
║          Rose Blanche Group - Technical Documentation            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}""")

    def print_result(self, result: SearchResult):
        """Print a single search result with formatting."""
        # Score color based on value
        if result.score >= 0.8:
            score_color = Colors.GREEN
        elif result.score >= 0.6:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED

        print(f"""
{Colors.BOLD}┌─ Résultat {result.rank} ─────────────────────────────────────────────────┐{Colors.END}
│
│  {Colors.CYAN}Texte:{Colors.END}
│  {result.fragment}
│
│  {Colors.BOLD}Score:{Colors.END} {score_color}{result.score:.4f}{Colors.END}
│
{Colors.BOLD}└──────────────────────────────────────────────────────────────────┘{Colors.END}""")

    def print_results(self, results: List[SearchResult], query: str):
        """Print all search results."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}═══ TOP {len(results)} RÉSULTATS POUR: \"{query[:50]}...\" ═══{Colors.END}\n")
        
        if not results:
            print(f"{Colors.YELLOW}Aucun résultat trouvé.{Colors.END}")
            return

        for result in results:
            self.print_result(result)

        # Summary
        avg_score = sum(r.score for r in results) / len(results)
        print(f"\n{Colors.CYAN}Moyenne des scores: {avg_score:.4f}{Colors.END}")

    def print_stats(self, db: VectorDatabase):
        """Print database statistics."""
        print(f"""
{Colors.CYAN}━━━ Statistiques ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
  Fragments chargés: {Colors.GREEN}{len(db)}{Colors.END}
  Modèle: {Colors.GREEN}{model_config.model_name}{Colors.END}
  Dimension: {Colors.GREEN}{model_config.embedding_dimension}{Colors.END}
  Top K: {Colors.GREEN}{search_config.top_k}{Colors.END}
{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
""")

    def run_interactive(self):
        """Run the interactive search session."""
        self.print_header()
        self.print_stats(self.engine.db)

        print(f"{Colors.YELLOW}Tapez 'quit' ou 'exit' pour quitter.{Colors.END}")
        print(f"{Colors.YELLOW}Tapez 'help' pour voir les commandes disponibles.{Colors.END}\n")

        while True:
            try:
                print(f"{Colors.GREEN}{Colors.BOLD}?{Colors.END} ", end="")
                query = input(f"{Colors.BOLD}Votre question: {Colors.END}").strip()

                if not query:
                    continue

                if query.lower() in ("quit", "exit", "q"):
                    print(f"\n{Colors.CYAN}Au revoir!{Colors.END}\n")
                    break

                if query.lower() == "help":
                    self.print_help()
                    continue

                if query.lower() == "stats":
                    self.print_stats(self.engine.db)
                    continue

                # Perform search
                results = self.engine.search(query)
                self.print_results(results, query)
                print()

            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}Au revoir!{Colors.END}\n")
                break
            except Exception as e:
                print(f"{Colors.RED}Erreur: {e}{Colors.END}")

    def print_help(self):
        """Print help information."""
        print(f"""
{Colors.CYAN}━━━ Commandes Disponibles ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
  help   - Afficher cette aide
  stats  - Afficher les statistiques
  quit   - Quitter l'application
  
{Colors.CYAN}━━━ Exemples de Questions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
  • Quelles sont les quantités recommandées d'alpha-amylase?
  • Comment améliorer l'extensibilité de la pâte?
  • Quel est le dosage de l'acide ascorbique?
{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
""")

    def run_single_query(self, query: str):
        """Run a single query and exit."""
        results = self.engine.search(query)
        self.print_results(results, query)


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the semantic search module."""
    
    # Initialize database
    db = VectorDatabase()
    
    # Try to load data (cache first, then database)
    if not db.load(use_cache=True):
        logger.error("Failed to load embeddings. Please check database connection.")
        sys.exit(1)

    # Initialize search engine
    engine = SemanticSearchEngine(db)

    # Initialize CLI
    cli = CLI(engine)

    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        cli.run_single_query(query)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
