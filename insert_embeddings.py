"""
Embedding Insertion Script
===========================
Utility script to insert text fragments and their embeddings into the PostgreSQL database.
"""

import logging
from typing import List, Optional

import psycopg2
from sentence_transformers import SentenceTransformer

from config import db_config, model_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class EmbeddingInserter:
    """Handles insertion of text fragments and embeddings into the database."""

    def __init__(self):
        self.model = SentenceTransformer(model_config.model_name)
        logger.info(f"Loaded model: {model_config.model_name}")

    def connect(self) -> psycopg2.extensions.connection:
        """Create a database connection."""
        return psycopg2.connect(**db_config.to_dict())

    def create_table_if_not_exists(self):
        """Create the embeddings table if it doesn't exist."""
        conn = self.connect()
        cur = conn.cursor()

        try:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    id_document INTEGER NOT NULL,
                    texte_fragment TEXT NOT NULL,
                    vecteur VECTOR(384)
                )
            """)
            
            # Create index for faster similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings USING ivfflat (vecteur vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            conn.commit()
            logger.info("Table 'embeddings' created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def insert_fragments(
        self,
        texts: List[str],
        document_id: int = 1,
        skip_duplicates: bool = True
    ) -> int:
        """
        Insert text fragments with their embeddings into the database.

        Args:
            texts: List of text fragments to insert
            document_id: Document ID to associate with fragments
            skip_duplicates: Skip fragments that already exist

        Returns:
            Number of fragments inserted
        """
        if not texts:
            logger.warning("No texts provided for insertion")
            return 0

        conn = self.connect()
        cur = conn.cursor()
        inserted_count = 0

        try:
            logger.info(f"Generating embeddings for {len(texts)} fragments...")
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=model_config.normalize_embeddings,
                show_progress_bar=True
            )

            for text, embedding in zip(texts, embeddings):
                # Check for duplicates if enabled
                if skip_duplicates:
                    cur.execute("""
                        SELECT 1 FROM embeddings
                        WHERE texte_fragment = %s
                        LIMIT 1
                    """, (text,))
                    
                    if cur.fetchone():
                        logger.debug(f"Skipping duplicate: {text[:50]}...")
                        continue

                # Insert the fragment
                cur.execute("""
                    INSERT INTO embeddings (id_document, texte_fragment, vecteur)
                    VALUES (%s, %s, %s)
                """, (document_id, text, embedding.tolist()))
                
                inserted_count += 1

            conn.commit()
            logger.info(f"Successfully inserted {inserted_count} fragments")
            return inserted_count

        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def get_fragment_count(self) -> int:
        """Get the total number of fragments in the database."""
        conn = self.connect()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            return count
        finally:
            cur.close()
            conn.close()

    def clear_all(self, confirm: bool = False):
        """Clear all embeddings from the database. Requires confirmation."""
        if not confirm:
            logger.warning("clear_all() requires confirm=True")
            return

        conn = self.connect()
        cur = conn.cursor()
        
        try:
            cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY")
            conn.commit()
            logger.info("All embeddings cleared from database")
        finally:
            cur.close()
            conn.close()


# ==================== SAMPLE DATA ====================

SAMPLE_FRAGMENTS = [
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


# ==================== MAIN ====================

def main():
    """Main entry point."""
    inserter = EmbeddingInserter()
    
    # Check current count
    try:
        count = inserter.get_fragment_count()
        logger.info(f"Current fragments in database: {count}")
    except:
        logger.info("Creating table structure...")
        inserter.create_table_if_not_exists()

    # Insert sample data
    logger.info("Inserting sample fragments...")
    inserted = inserter.insert_fragments(SAMPLE_FRAGMENTS, document_id=1)
    
    logger.info(f"Done! Total inserted: {inserted}")


if __name__ == "__main__":
    main()
