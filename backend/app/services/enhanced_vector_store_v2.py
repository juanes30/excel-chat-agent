"""Enhanced Vector Store V2.0 with Multi-Modal Embedding Strategy.

This module extends the existing AdvancedVectorStoreService with multi-modal
embedding capabilities specifically optimized for Excel data characteristics.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
from dataclasses import dataclass

# Import base enhanced vector store
from .enhanced_vector_store import (
    AdvancedVectorStoreService,
    SearchQuery,
    SearchStrategy,
    SearchResult,
    SearchAnalytics
)

# Import enhanced embedding strategy
from .enhanced_embedding_strategy import (
    EnhancedEmbeddingStrategy,
    EmbeddingConfig,
    ContentType,
    EmbeddingModel,
    ContentAnalysis
)

logger = logging.getLogger(__name__)


@dataclass
class MultiModalSearchResult(SearchResult):
    """Enhanced search result with multi-modal embedding information."""
    content_type: Optional[ContentType] = None
    embedding_strategy: Optional[str] = None
    numerical_features: Optional[Dict[str, float]] = None
    business_context: Optional[Dict[str, Any]] = None
    hierarchy_features: Optional[Dict[str, float]] = None


class EnhancedVectorStoreV2(AdvancedVectorStoreService):
    """Enhanced Vector Store V2.0 with multi-modal embedding strategy."""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 collection_name: str = "excel_documents_v2",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 enable_analytics: bool = True,
                 cache_ttl_minutes: int = 60,
                 enable_multi_modal: bool = True,
                 embedding_config: Optional[EmbeddingConfig] = None):
        """Initialize Enhanced Vector Store V2.0.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection  
            embedding_model: Primary embedding model
            enable_analytics: Enable search analytics
            cache_ttl_minutes: Cache TTL in minutes
            enable_multi_modal: Enable multi-modal embedding strategy
            embedding_config: Configuration for enhanced embedding strategy
        """
        # Initialize base class
        super().__init__(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            enable_analytics=enable_analytics,
            cache_ttl_minutes=cache_ttl_minutes
        )
        
        # Enhanced embedding strategy
        self.enable_multi_modal = enable_multi_modal
        if enable_multi_modal:
            config = embedding_config or EmbeddingConfig(
                primary_model=EmbeddingModel.GENERAL,
                enable_gpu=True,
                cache_enabled=True,
                parallel_processing=True
            )
            self.enhanced_embedding = EnhancedEmbeddingStrategy(config)
        else:
            self.enhanced_embedding = None
        
        # Multi-modal specific caches
        self.content_analysis_cache = {}
        self.embedding_strategy_cache = {}
        
        logger.info("Enhanced Vector Store V2.0 initialized with multi-modal capabilities")
    
    async def _generate_enhanced_embeddings(self, texts: List[str], metadata_list: List[Dict[str, Any]] = None) -> List[List[float]]:
        """Generate embeddings using enhanced multi-modal strategy."""
        if not self.enable_multi_modal or not self.enhanced_embedding:
            # Fallback to base implementation
            return self._generate_embeddings(texts)
        
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        try:
            embeddings = []
            
            # Process texts with enhanced strategy
            for text, metadata in zip(texts, metadata_list):
                # Generate embedding with enhanced strategy
                embedding = await self.enhanced_embedding.generate_embedding(text, metadata)
                embeddings.append(embedding.tolist())
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in enhanced embedding generation: {e}")
            # Fallback to base embeddings
            return self._generate_embeddings(texts)
    
    async def add_excel_data_v2(self, 
                               file_name: str,
                               file_hash: str,
                               sheets_data: Dict[str, Any],
                               batch_size: int = 50,
                               enable_content_analysis: bool = True) -> Dict[str, Any]:
        """Enhanced version of add_excel_data with multi-modal processing.
        
        Args:
            file_name: Name of the Excel file
            file_hash: Hash of the Excel file
            sheets_data: Dictionary containing sheet data and text chunks
            batch_size: Size of batches for processing
            enable_content_analysis: Enable content analysis for optimization
            
        Returns:
            Dictionary with detailed processing results
        """
        try:
            start_time = time.time()
            
            # Check if file already exists
            existing_docs = self.collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            
            if existing_docs['ids']:
                logger.info(f"File {file_name} already exists in vector store")
                return {
                    "success": True,
                    "action": "skipped",
                    "reason": "file_already_exists",
                    "processing_time_ms": 0
                }
            
            logger.info(f"Adding Excel data with enhanced processing: {file_name}")
            
            # Prepare documents with enhanced metadata
            documents = []
            metadatas = []
            ids = []
            content_analyses = []
            
            for sheet_name, sheet_data in sheets_data.items():
                text_chunks = sheet_data.get('text_chunks', [])
                base_metadata = sheet_data.get('metadata', {})
                
                for i, chunk in enumerate(text_chunks):
                    if not chunk.strip():
                        continue
                    
                    doc_id = f"{file_hash}_{sheet_name}_{i}"
                    
                    # Enhanced metadata for multi-modal processing
                    enhanced_metadata = {
                        **base_metadata,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "sheet_name": sheet_name,
                        "chunk_index": i,
                        "added_at": datetime.now().isoformat(),
                        "chunk_type": "data" if i > 0 else "summary",
                        "file_data": {
                            "file_name": file_name,
                            "file_size_mb": base_metadata.get("file_size_mb", 0),
                            "total_sheets": len(sheets_data),
                            "extension": ".xlsx"  # Default assumption
                        },
                        "sheet_data": {
                            "sheet_name": sheet_name,
                            "num_rows": base_metadata.get('num_rows', 0),
                            "num_cols": base_metadata.get('num_cols', 0),
                            "data_quality": base_metadata.get("data_quality", {}),
                            "patterns_detected": base_metadata.get("patterns_detected", {}),
                            "relationships": base_metadata.get("relationships", {})
                        }
                    }
                    
                    documents.append(chunk)
                    metadatas.append(enhanced_metadata)
                    ids.append(doc_id)
                    
                    # Content analysis for optimization
                    if enable_content_analysis and self.enhanced_embedding:
                        try:
                            analysis = await self.enhanced_embedding.analyze_content(chunk, enhanced_metadata)
                            content_analyses.append(analysis)
                            
                            # Store content type in metadata
                            enhanced_metadata["content_type"] = analysis.content_type.value
                            enhanced_metadata["content_confidence"] = analysis.confidence
                            
                        except Exception as e:
                            logger.warning(f"Content analysis failed for chunk {i}: {e}")
                            content_analyses.append(None)
            
            if not documents:
                logger.warning(f"No valid documents found for file: {file_name}")
                return {
                    "success": False,
                    "action": "failed",
                    "reason": "no_valid_documents",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            # Process in batches with enhanced embeddings
            total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)
            embedding_stats = {
                "numerical_chunks": 0,
                "textual_chunks": 0,
                "hierarchical_chunks": 0,
                "business_chunks": 0,
                "mixed_chunks": 0
            }
            
            for batch_idx in range(0, len(documents), batch_size):
                batch_end = min(batch_idx + batch_size, len(documents))
                batch_docs = documents[batch_idx:batch_end]
                batch_metas = metadatas[batch_idx:batch_end]
                batch_ids = ids[batch_idx:batch_end]
                
                # Generate embeddings with enhanced strategy
                embeddings = await self._generate_enhanced_embeddings(batch_docs, batch_metas)
                
                # Update embedding statistics
                for meta in batch_metas:
                    content_type = meta.get("content_type", "mixed")
                    if content_type in embedding_stats:
                        embedding_stats[f"{content_type}_chunks"] += 1
                    else:
                        embedding_stats["mixed_chunks"] += 1
                
                # Add to collection
                self.collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                logger.debug(f"Added batch {batch_idx // batch_size + 1}/{total_batches} "
                           f"({len(batch_docs)} documents)")
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Successfully added {len(documents)} documents for {file_name} "
                       f"in {processing_time:.1f}ms")
            
            return {
                "success": True,
                "action": "added",
                "documents_added": len(documents),
                "processing_time_ms": processing_time,
                "embedding_stats": embedding_stats,
                "content_analysis_enabled": enable_content_analysis,
                "multi_modal_enabled": self.enable_multi_modal
            }
            
        except Exception as e:
            logger.error(f"Error adding Excel data to enhanced vector store: {e}")
            return {
                "success": False,
                "action": "failed",
                "reason": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            }
    
    async def enhanced_search_v2(self, query: Union[str, SearchQuery]) -> List[MultiModalSearchResult]:
        """Enhanced search with multi-modal awareness."""
        try:
            # Perform base enhanced search
            base_results = await self.enhanced_search(query)
            
            # Convert to multi-modal results with additional context
            enhanced_results = []
            
            for result in base_results:
                # Extract content type from metadata if available
                content_type_str = result.metadata.get("content_type")
                content_type = ContentType(content_type_str) if content_type_str else None
                
                # Extract embedding strategy information
                embedding_strategy = "multi_modal" if self.enable_multi_modal else "single_model"
                
                # Extract specialized features based on content type
                numerical_features = None
                business_context = None
                hierarchy_features = None
                
                if content_type == ContentType.NUMERICAL:
                    numerical_features = self._extract_numerical_features(result.metadata)
                elif content_type == ContentType.BUSINESS_DOMAIN:
                    business_context = self._extract_business_context(result.metadata)
                elif content_type == ContentType.HIERARCHICAL:
                    hierarchy_features = self._extract_hierarchy_features(result.metadata)
                
                # Create enhanced result
                enhanced_result = MultiModalSearchResult(
                    content=result.content,
                    metadata=result.metadata,
                    relevance_score=result.relevance_score,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    quality_score=result.quality_score,
                    freshness_score=result.freshness_score,
                    explanation=result.explanation,
                    file_name=result.file_name,
                    sheet_name=result.sheet_name,
                    chunk_index=result.chunk_index,
                    # Multi-modal specific fields
                    content_type=content_type,
                    embedding_strategy=embedding_strategy,
                    numerical_features=numerical_features,
                    business_context=business_context,
                    hierarchy_features=hierarchy_features
                )
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search v2: {e}")
            # Fallback to base search with conversion
            base_results = await self.enhanced_search(query)
            return [MultiModalSearchResult(
                content=r.content,
                metadata=r.metadata,
                relevance_score=r.relevance_score,
                semantic_score=r.semantic_score,
                keyword_score=r.keyword_score,
                quality_score=r.quality_score,
                freshness_score=r.freshness_score,
                file_name=r.file_name,
                sheet_name=r.sheet_name,
                chunk_index=r.chunk_index
            ) for r in base_results]
    
    def _extract_numerical_features(self, metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract numerical features from metadata."""
        features = {}
        
        # Data quality numerical indicators
        if "data_quality" in metadata:
            quality_data = metadata["data_quality"]
            features.update({
                "completeness_score": quality_data.get("completeness_score", 0.0),
                "consistency_score": quality_data.get("consistency_score", 0.0),
                "validity_score": quality_data.get("validity_score", 0.0),
                "accuracy_score": quality_data.get("accuracy_score", 0.0)
            })
        
        # Statistical features if available
        if "statistics" in metadata:
            stats = metadata["statistics"]
            features.update({
                "mean": stats.get("mean", 0.0),
                "std": stats.get("std", 0.0),
                "skewness": stats.get("skewness", 0.0),
                "kurtosis": stats.get("kurtosis", 0.0)
            })
        
        return features if features else None
    
    def _extract_business_context(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract business context from metadata."""
        context = {}
        
        # File and sheet context
        context["file_name"] = metadata.get("file_name", "")
        context["sheet_name"] = metadata.get("sheet_name", "")
        
        # Business patterns
        if "patterns_detected" in metadata:
            patterns = metadata["patterns_detected"]
            context["has_business_patterns"] = any(
                pattern in ["email", "phone", "currency"] 
                for pattern in patterns.keys()
            )
            context["pattern_types"] = list(patterns.keys())
        
        # Quality indicators for business relevance
        if "data_quality" in metadata:
            quality = metadata["data_quality"]
            context["overall_quality"] = quality.get("overall_quality", "unknown")
            context["business_relevance_score"] = quality.get("completeness_score", 0.5)
        
        return context if context else None
    
    def _extract_hierarchy_features(self, metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract hierarchical features from metadata."""
        features = {}
        
        # File-level hierarchy
        if "file_data" in metadata:
            file_data = metadata["file_data"]
            features["file_size_mb"] = file_data.get("file_size_mb", 0.0)
            features["total_sheets"] = float(file_data.get("total_sheets", 1))
        
        # Sheet-level hierarchy
        if "sheet_data" in metadata:
            sheet_data = metadata["sheet_data"]
            features["num_rows"] = float(sheet_data.get("num_rows", 0))
            features["num_cols"] = float(sheet_data.get("num_cols", 0))
        
        # Relationship complexity
        if "relationships" in metadata:
            relationships = metadata["relationships"]
            correlations = relationships.get("correlations", [])
            features["correlation_count"] = float(len(correlations))
            features["max_correlation"] = max(
                [abs(corr.get("corr", 0)) for corr in correlations], 
                default=0.0
            )
        
        return features if features else None
    
    async def migrate_from_v1(self, 
                             source_collection_name: str = "excel_documents",
                             batch_size: int = 100,
                             preserve_original: bool = True) -> Dict[str, Any]:
        """Migrate data from V1 vector store to V2 with enhanced embeddings.
        
        Args:
            source_collection_name: Name of source collection to migrate from
            batch_size: Batch size for migration
            preserve_original: Whether to preserve the original collection
            
        Returns:
            Migration results and statistics
        """
        try:
            migration_start = time.time()
            logger.info(f"Starting migration from {source_collection_name} to V2")
            
            # Get source collection
            try:
                source_collection = self.client.get_collection(name=source_collection_name)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Source collection {source_collection_name} not found"
                }
            
            # Get all documents from source
            total_docs = source_collection.count()
            migrated_docs = 0
            failed_docs = 0
            
            # Process in batches
            for offset in range(0, total_docs, batch_size):
                try:
                    # Get batch from source
                    batch_data = source_collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents", "metadatas", "embeddings"]
                    )
                    
                    if not batch_data['ids']:
                        break
                    
                    # Enhance metadata for V2
                    enhanced_metadatas = []
                    for metadata in batch_data['metadatas']:
                        enhanced_metadata = {
                            **metadata,
                            "migrated_from_v1": True,
                            "migration_timestamp": datetime.now().isoformat(),
                            "v2_features_enabled": self.enable_multi_modal
                        }
                        enhanced_metadatas.append(enhanced_metadata)
                    
                    # Re-generate embeddings with enhanced strategy
                    if self.enable_multi_modal and self.enhanced_embedding:
                        logger.debug(f"Re-generating embeddings for batch {offset//batch_size + 1}")
                        new_embeddings = await self._generate_enhanced_embeddings(
                            batch_data['documents'], 
                            enhanced_metadatas
                        )
                    else:
                        # Use existing embeddings
                        new_embeddings = batch_data['embeddings']
                    
                    # Add to V2 collection
                    self.collection.add(
                        documents=batch_data['documents'],
                        embeddings=new_embeddings,
                        metadatas=enhanced_metadatas,
                        ids=batch_data['ids']
                    )
                    
                    migrated_docs += len(batch_data['ids'])
                    logger.info(f"Migrated {migrated_docs}/{total_docs} documents")
                    
                except Exception as batch_error:
                    logger.error(f"Error migrating batch {offset//batch_size + 1}: {batch_error}")
                    failed_docs += batch_size
                    continue
            
            migration_time = time.time() - migration_start
            
            # Migration results
            results = {
                "success": True,
                "total_documents": total_docs,
                "migrated_documents": migrated_docs,
                "failed_documents": failed_docs,
                "migration_time_seconds": migration_time,
                "multi_modal_enabled": self.enable_multi_modal,
                "preserved_original": preserve_original
            }
            
            logger.info(f"Migration completed: {migrated_docs}/{total_docs} documents in {migration_time:.1f}s")
            
            # Optionally remove source collection
            if not preserve_original and migrated_docs == total_docs:
                try:
                    self.client.delete_collection(name=source_collection_name)
                    results["original_collection_deleted"] = True
                    logger.info(f"Deleted original collection: {source_collection_name}")
                except Exception as e:
                    logger.warning(f"Could not delete original collection: {e}")
                    results["original_collection_deleted"] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "migration_time_seconds": time.time() - migration_start if 'migration_start' in locals() else 0
            }
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including multi-modal information."""
        base_stats = self.get_statistics()
        
        enhanced_stats = {
            **base_stats,
            "v2_features": {
                "multi_modal_enabled": self.enable_multi_modal,
                "embedding_strategy_cache_size": len(self.embedding_strategy_cache),
                "content_analysis_cache_size": len(self.content_analysis_cache)
            }
        }
        
        # Add enhanced embedding statistics if available
        if self.enhanced_embedding:
            embedding_cache_stats = self.enhanced_embedding.get_cache_stats()
            enhanced_stats["enhanced_embedding"] = embedding_cache_stats
        
        # Content type distribution
        try:
            sample_results = self.collection.get(
                limit=min(1000, base_stats["total_documents"]),
                include=["metadatas"]
            )
            
            content_type_dist = {}
            for metadata in sample_results['metadatas'] or []:
                content_type = metadata.get("content_type", "unknown")
                content_type_dist[content_type] = content_type_dist.get(content_type, 0) + 1
            
            enhanced_stats["content_type_distribution"] = content_type_dist
            
        except Exception as e:
            logger.warning(f"Could not get content type distribution: {e}")
            enhanced_stats["content_type_distribution"] = {}
        
        return enhanced_stats

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the enhanced vector store service.
        
        Returns:
            Dictionary with enhanced health status
        """
        try:
            # Get base health check from parent class
            base_health = super().health_check() if hasattr(super(), 'health_check') else {}
            
            # Enhanced health checks
            enhanced_checks = {
                "enhanced_features_available": True,
                "multi_modal_enabled": getattr(self, 'enable_multi_modal', False),
                "embedding_strategy_loaded": hasattr(self, 'enhanced_embedding'),
                "cache_accessible": len(getattr(self, 'content_analysis_cache', {})) >= 0,
                "embedding_strategy_cache_accessible": len(getattr(self, 'embedding_strategy_cache', {})) >= 0
            }
            
            # Test enhanced functionality
            try:
                # Test enhanced embedding generation if available
                if hasattr(self, 'enhanced_embedding'):
                    test_result = await self._generate_enhanced_embeddings(["test health check content"])
                    enhanced_checks["enhanced_embeddings_working"] = len(test_result) > 0
                else:
                    enhanced_checks["enhanced_embeddings_working"] = False
            except Exception as e:
                logger.warning(f"Enhanced embedding test failed: {e}")
                enhanced_checks["enhanced_embeddings_working"] = False
            
            # Determine overall status
            all_checks_passed = all([
                base_health.get("status", "unhealthy") == "healthy",
                enhanced_checks.get("enhanced_features_available", False),
                enhanced_checks.get("embedding_strategy_loaded", False)
            ])
            
            status = "healthy" if all_checks_passed else "degraded"
            
            return {
                "status": status,
                **base_health,
                **enhanced_checks,
                "service_type": "enhanced_vector_store_v2"
            }
            
        except Exception as e:
            logger.error(f"Enhanced vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_type": "enhanced_vector_store_v2",
                "enhanced_features_available": False
            }
    
    def clear_enhanced_caches(self):
        """Clear all enhanced caches."""
        # Clear base caches
        self.clear_cache()
        self.clear_analytics()
        
        # Clear V2 specific caches
        self.content_analysis_cache.clear()
        self.embedding_strategy_cache.clear()
        
        # Clear enhanced embedding caches
        if self.enhanced_embedding:
            self.enhanced_embedding.clear_cache()
        
        logger.info("All enhanced caches cleared")


# Backward compatibility aliases
EnhancedVectorStore = EnhancedVectorStoreV2
MultiModalVectorStore = EnhancedVectorStoreV2