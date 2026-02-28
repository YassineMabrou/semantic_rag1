"""
Configuration settings for Semantic RAG module.
Uses environment variables for sensitive data with fallback defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    dbname: str = os.getenv("DB_NAME", "database")
    user: str = os.getenv("DB_USER", "user")
    password: str = os.getenv("DB_PASSWORD", "password")

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password
        }


@dataclass
class ModelConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    normalize_embeddings: bool = True


@dataclass
class SearchConfig:
    """Search parameters configuration."""
    top_k: int = 3
    min_similarity_threshold: float = 0.0  # Minimum score to include in results
    deduplicate_results: bool = True


# Global configuration instances
db_config = DatabaseConfig()
model_config = ModelConfig()
search_config = SearchConfig()
