"""Shared Embedding Service to prevent multiple SentenceTransformer model loading.

This singleton service ensures only one instance of each embedding model is loaded,
reducing memory usage and startup time.
"""

import logging
import threading
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SharedEmbeddingService:
    """Singleton service for managing SentenceTransformer models."""
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, SentenceTransformer] = {}
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    logger.info("SharedEmbeddingService singleton created")
        return cls._instance
    
    def get_model(self, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        """Get or create a SentenceTransformer model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            SentenceTransformer model instance
        """
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    logger.info(f"Loading SentenceTransformer model: {model_name}")
                    self._models[model_name] = SentenceTransformer(model_name)
                    logger.info(f"Successfully loaded model: {model_name}")
                else:
                    logger.debug(f"Using cached model: {model_name}")
        else:
            logger.debug(f"Using cached model: {model_name}")
            
        return self._models[model_name]
    
    def encode(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2", **kwargs) -> List[List[float]]:
        """Encode texts using the specified model.
        
        Args:
            texts: List of texts to encode
            model_name: Model to use for encoding
            **kwargs: Additional arguments for the encode method
            
        Returns:
            List of embeddings
        """
        model = self.get_model(model_name)
        return model.encode(texts, **kwargs).tolist()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self._models.keys())
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        with self._lock:
            if model_name:
                if model_name in self._models:
                    del self._models[model_name]
                    logger.info(f"Cleared cached model: {model_name}")
            else:
                self._models.clear()
                logger.info("Cleared all cached models")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage information for loaded models."""
        return {
            "loaded_models_count": len(self._models),
            "model_names": list(self._models.keys())
        }


# Global instance
embedding_service = SharedEmbeddingService()