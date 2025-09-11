"""Advanced Vector Store Service with Enhanced Semantic Search Capabilities.

This module provides sophisticated semantic search capabilities for Excel data,
including hybrid search, advanced relevance scoring, query analytics, and 
intelligent metadata integration.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import the base vector store service
from .vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy options."""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    KEYWORD_ONLY = "keyword_only"
    ADAPTIVE = "adaptive"


class RelevanceScoring(Enum):
    """Relevance scoring methods."""
    COSINE_SIMILARITY = "cosine_similarity"
    HYBRID_WEIGHTED = "hybrid_weighted"
    CONTEXT_AWARE = "context_aware"
    QUALITY_WEIGHTED = "quality_weighted"


@dataclass
class SearchQuery:
    """Structured search query with advanced options."""
    text: str
    strategy: SearchStrategy = SearchStrategy.ADAPTIVE
    scoring: RelevanceScoring = RelevanceScoring.CONTEXT_AWARE
    n_results: int = 5
    min_relevance: float = 0.0
    filters: Dict[str, Any] = field(default_factory=dict)
    facets: List[str] = field(default_factory=list)
    boost_factors: Dict[str, float] = field(default_factory=dict)
    include_explanation: bool = False


@dataclass
class SearchResult:
    """Enhanced search result with detailed metadata and scoring."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_score: float
    keyword_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    explanation: Optional[Dict[str, Any]] = None
    file_name: str = ""
    sheet_name: str = ""
    chunk_index: int = 0


@dataclass
class SearchAnalytics:
    """Search analytics and performance metrics."""
    query_id: str
    query_text: str
    strategy_used: SearchStrategy
    total_time_ms: float
    embedding_time_ms: float
    search_time_ms: float
    post_processing_time_ms: float
    results_count: int
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class QueryExpander:
    """Handles query expansion and enhancement."""
    
    def __init__(self):
        # Common synonyms and related terms for Excel/data analysis
        self.synonyms = {
            "sales": ["revenue", "income", "earnings", "turnover"],
            "cost": ["expense", "expenditure", "spending", "outlay"],
            "profit": ["earnings", "income", "gain", "return"],
            "customer": ["client", "buyer", "purchaser", "consumer"],
            "product": ["item", "article", "good", "merchandise"],
            "date": ["time", "period", "timestamp", "when"],
            "amount": ["quantity", "number", "total", "sum"],
            "average": ["mean", "typical", "standard", "normal"],
            "total": ["sum", "aggregate", "overall", "complete"]
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with synonyms and related terms."""
        expanded_queries = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:max_expansions]:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:max_expansions + 1]


class RelevanceScorer:
    """Advanced relevance scoring with multiple factors."""
    
    @staticmethod
    def calculate_semantic_score(distance: float) -> float:
        """Calculate semantic similarity score from distance."""
        return max(0.0, 1.0 - (distance / 2.0))
    
    @staticmethod
    def calculate_keyword_score(query: str, content: str) -> float:
        """Calculate keyword matching score using simple TF approach."""
        query_terms = set(query.lower().split())
        content_terms = content.lower().split()
        content_term_count = len(content_terms)
        
        if content_term_count == 0:
            return 0.0
        
        # Calculate term frequency for query terms in content
        matches = sum(1 for term in content_terms if term in query_terms)
        return matches / content_term_count if content_term_count > 0 else 0.0
    
    @staticmethod
    def calculate_quality_score(metadata: Dict[str, Any]) -> float:
        """Calculate content quality score from metadata."""
        # Default quality score
        quality_score = 0.5
        
        # Boost based on data quality metrics from enhanced processor
        if "data_quality" in metadata:
            quality_data = metadata["data_quality"]
            if isinstance(quality_data, dict):
                quality_score = quality_data.get("overall_score", 0.5)
        
        # Boost for completeness
        completeness = metadata.get("completeness_percentage", 50) / 100
        quality_score = (quality_score + completeness) / 2
        
        return max(0.0, min(1.0, quality_score))
    
    @staticmethod
    def calculate_freshness_score(metadata: Dict[str, Any]) -> float:
        """Calculate content freshness score."""
        added_at = metadata.get("added_at")
        if not added_at:
            return 0.5
        
        try:
            added_time = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
            age_days = (datetime.now() - added_time).days
            
            # Fresher content gets higher score (decay over 30 days)
            freshness = max(0.0, 1.0 - (age_days / 30.0))
            return freshness
        except (ValueError, AttributeError):
            return 0.5
    
    @classmethod
    def calculate_hybrid_score(
        cls,
        semantic_score: float,
        keyword_score: float,
        quality_score: float,
        freshness_score: float,
        weights: Dict[str, float] = None
    ) -> float:
        """Calculate weighted hybrid relevance score."""
        if weights is None:
            weights = {
                "semantic": 0.6,
                "keyword": 0.2,
                "quality": 0.15,
                "freshness": 0.05
            }
        
        hybrid_score = (
            semantic_score * weights.get("semantic", 0.6) +
            keyword_score * weights.get("keyword", 0.2) +
            quality_score * weights.get("quality", 0.15) +
            freshness_score * weights.get("freshness", 0.05)
        )
        
        return max(0.0, min(1.0, hybrid_score))


class AdvancedVectorStoreService(VectorStoreService):
    """Enhanced Vector Store Service with advanced semantic search capabilities."""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 collection_name: str = "excel_documents",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 enable_analytics: bool = True,
                 cache_ttl_minutes: int = 60):
        """Initialize the Advanced Vector Store Service.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model for embeddings
            enable_analytics: Enable search analytics tracking
            cache_ttl_minutes: Cache TTL in minutes
        """
        # Initialize base class
        super().__init__(persist_directory, collection_name, embedding_model)
        
        # Enhanced components
        self.query_expander = QueryExpander()
        self.relevance_scorer = RelevanceScorer()
        self.enable_analytics = enable_analytics
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Analytics storage
        self.search_analytics: List[SearchAnalytics] = []
        self.query_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0,
            "popular_queries": defaultdict(int)
        }
        
        logger.info("Initialized AdvancedVectorStoreService with enhanced capabilities")
    
    def _generate_query_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query."""
        return f"{hash(query.text)}_{query.strategy.value}_{query.n_results}_{hash(str(sorted(query.filters.items())))}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        return datetime.now() - timestamp < self.cache_ttl
    
    async def enhanced_search(self, query: Union[str, SearchQuery]) -> List[SearchResult]:
        """Perform enhanced semantic search with advanced features.
        
        Args:
            query: Search query (string or SearchQuery object)
            
        Returns:
            List of enhanced search results
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        # Convert string query to SearchQuery object
        if isinstance(query, str):
            search_query = SearchQuery(text=query)
        else:
            search_query = query
        
        try:
            # Check cache first
            cache_key = self._generate_query_cache_key(search_query)
            if cache_key in self.query_cache:
                cached_results, timestamp = self.query_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    if self.enable_analytics:
                        self.performance_metrics["cache_hits"] += 1
                        analytics = SearchAnalytics(
                            query_id=query_id,
                            query_text=search_query.text,
                            strategy_used=search_query.strategy,
                            total_time_ms=(time.time() - start_time) * 1000,
                            embedding_time_ms=0,
                            search_time_ms=0,
                            post_processing_time_ms=0,
                            results_count=len(cached_results),
                            cache_hit=True
                        )
                        self.search_analytics.append(analytics)
                    return cached_results
            
            # Perform search based on strategy
            if search_query.strategy == SearchStrategy.ADAPTIVE:
                # Choose strategy based on query characteristics
                strategy = self._choose_adaptive_strategy(search_query.text)
            else:
                strategy = search_query.strategy
            
            # Execute search
            embedding_start = time.time()
            results = await self._execute_search_strategy(search_query, strategy)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Post-process and score results
            post_process_start = time.time()
            enhanced_results = self._enhance_search_results(
                results, search_query, strategy
            )
            post_process_time = (time.time() - post_process_start) * 1000
            
            # Apply filters and sorting
            filtered_results = self._apply_filters_and_sort(
                enhanced_results, search_query
            )
            
            # Cache results
            self.query_cache[cache_key] = (filtered_results, datetime.now())
            
            # Record analytics
            total_time = (time.time() - start_time) * 1000
            if self.enable_analytics:
                self.performance_metrics["total_queries"] += 1
                self.performance_metrics["popular_queries"][search_query.text] += 1
                
                # Update average query time
                current_avg = self.performance_metrics["average_query_time"]
                total_queries = self.performance_metrics["total_queries"]
                self.performance_metrics["average_query_time"] = (
                    (current_avg * (total_queries - 1) + total_time) / total_queries
                )
                
                analytics = SearchAnalytics(
                    query_id=query_id,
                    query_text=search_query.text,
                    strategy_used=strategy,
                    total_time_ms=total_time,
                    embedding_time_ms=embedding_time,
                    search_time_ms=total_time - embedding_time - post_process_time,
                    post_processing_time_ms=post_process_time,
                    results_count=len(filtered_results),
                    cache_hit=False
                )
                self.search_analytics.append(analytics)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            # Fallback to basic search
            basic_results = await self.search(
                query=search_query.text,
                n_results=search_query.n_results
            )
            return [SearchResult(
                content=r["content"],
                metadata=r["metadata"],
                relevance_score=r["relevance_score"],
                semantic_score=r["relevance_score"],
                file_name=r["file_name"],
                sheet_name=r["sheet_name"],
                chunk_index=r["chunk_index"]
            ) for r in basic_results]
    
    def _choose_adaptive_strategy(self, query_text: str) -> SearchStrategy:
        """Choose search strategy based on query characteristics."""
        query_lower = query_text.lower()
        
        # Keywords that benefit from semantic search
        semantic_indicators = [
            "similar", "like", "related", "comparable", "equivalent",
            "pattern", "trend", "correlation", "relationship"
        ]
        
        # Keywords that benefit from keyword search
        keyword_indicators = [
            "exact", "specific", "precise", "contains", "includes",
            "name", "id", "code", "number"
        ]
        
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        keyword_score = sum(1 for indicator in keyword_indicators if indicator in query_lower)
        
        if semantic_score > keyword_score:
            return SearchStrategy.SEMANTIC_ONLY
        elif keyword_score > semantic_score:
            return SearchStrategy.KEYWORD_ONLY
        else:
            return SearchStrategy.HYBRID
    
    async def _execute_search_strategy(
        self, 
        search_query: SearchQuery, 
        strategy: SearchStrategy
    ) -> List[Dict[str, Any]]:
        """Execute search based on chosen strategy."""
        if strategy == SearchStrategy.SEMANTIC_ONLY:
            return await self._semantic_search(search_query)
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            return await self._keyword_search(search_query)
        elif strategy == SearchStrategy.HYBRID:
            return await self._hybrid_search(search_query)
        else:
            return await self._semantic_search(search_query)  # Default fallback
    
    async def _semantic_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform pure semantic search."""
        # Expand query for better semantic matching
        expanded_queries = self.query_expander.expand_query(search_query.text)
        
        all_results = []
        for query_text in expanded_queries:
            # Use base class search method
            results = await self.search(
                query=query_text,
                n_results=search_query.n_results * 2,  # Get more results for reranking
                file_filter=search_query.filters.get("file_name"),
                sheet_filter=search_query.filters.get("sheet_name")
            )
            all_results.extend(results)
        
        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(result["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results[:search_query.n_results]
    
    async def _keyword_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        # For now, use semantic search as base and boost keyword matches
        results = await self.search(
            query=search_query.text,
            n_results=search_query.n_results * 3,
            file_filter=search_query.filters.get("file_name"),
            sheet_filter=search_query.filters.get("sheet_name")
        )
        
        # Re-score based on keyword matching
        for result in results:
            keyword_score = self.relevance_scorer.calculate_keyword_score(
                search_query.text, result["content"]
            )
            # Boost relevance score for high keyword matches
            result["relevance_score"] = (result["relevance_score"] + keyword_score) / 2
        
        # Sort by keyword-boosted relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:search_query.n_results]
    
    async def _hybrid_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform hybrid semantic + keyword search."""
        # Get semantic results
        semantic_results = await self._semantic_search(search_query)
        
        # Get keyword-boosted results
        keyword_results = await self._keyword_search(search_query)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results with higher weight
        for result in semantic_results:
            content_hash = hash(result["content"])
            result["search_type"] = "semantic"
            combined_results[content_hash] = result
        
        # Add keyword results, merge if duplicate
        for result in keyword_results:
            content_hash = hash(result["content"])
            if content_hash in combined_results:
                # Average the scores
                existing = combined_results[content_hash]
                existing["relevance_score"] = (
                    existing["relevance_score"] + result["relevance_score"]
                ) / 2
                existing["search_type"] = "hybrid"
            else:
                result["search_type"] = "keyword"
                combined_results[content_hash] = result
        
        # Sort by combined relevance score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return final_results[:search_query.n_results]
    
    def _enhance_search_results(
        self, 
        results: List[Dict[str, Any]], 
        search_query: SearchQuery,
        strategy: SearchStrategy
    ) -> List[SearchResult]:
        """Enhance search results with detailed scoring and metadata."""
        enhanced_results = []
        
        for result in results:
            # Calculate various score components
            semantic_score = result["relevance_score"]
            keyword_score = self.relevance_scorer.calculate_keyword_score(
                search_query.text, result["content"]
            )
            quality_score = self.relevance_scorer.calculate_quality_score(
                result["metadata"]
            )
            freshness_score = self.relevance_scorer.calculate_freshness_score(
                result["metadata"]
            )
            
            # Calculate final hybrid score
            final_score = self.relevance_scorer.calculate_hybrid_score(
                semantic_score, keyword_score, quality_score, freshness_score
            )
            
            # Create explanation if requested
            explanation = None
            if search_query.include_explanation:
                explanation = {
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "quality_score": quality_score,
                    "freshness_score": freshness_score,
                    "strategy_used": strategy.value,
                    "search_type": result.get("search_type", "unknown")
                }
            
            enhanced_result = SearchResult(
                content=result["content"],
                metadata=result["metadata"],
                relevance_score=final_score,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                quality_score=quality_score,
                freshness_score=freshness_score,
                explanation=explanation,
                file_name=result["file_name"],
                sheet_name=result["sheet_name"],
                chunk_index=result["chunk_index"]
            )
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _apply_filters_and_sort(
        self, 
        results: List[SearchResult], 
        search_query: SearchQuery
    ) -> List[SearchResult]:
        """Apply filters and sorting to results."""
        filtered_results = results
        
        # Apply minimum relevance filter
        if search_query.min_relevance > 0:
            filtered_results = [
                r for r in filtered_results 
                if r.relevance_score >= search_query.min_relevance
            ]
        
        # Apply custom filters
        for filter_key, filter_value in search_query.filters.items():
            if filter_key not in ["file_name", "sheet_name"]:  # Already applied
                filtered_results = [
                    r for r in filtered_results
                    if r.metadata.get(filter_key) == filter_value
                ]
        
        # Sort by relevance score (already done, but ensure consistency)
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return filtered_results[:search_query.n_results]
    
    async def faceted_search(
        self, 
        query: str, 
        facets: List[str],
        n_results: int = 20
    ) -> Dict[str, Any]:
        """Perform faceted search with aggregations."""
        try:
            # Perform base search
            search_query = SearchQuery(
                text=query,
                n_results=n_results,
                facets=facets
            )
            results = await self.enhanced_search(search_query)
            
            # Calculate facet aggregations
            facet_counts = {}
            for facet in facets:
                facet_counts[facet] = defaultdict(int)
                for result in results:
                    facet_value = result.metadata.get(facet, "unknown")
                    facet_counts[facet][str(facet_value)] += 1
            
            return {
                "results": results,
                "facets": dict(facet_counts),
                "total_results": len(results),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error in faceted search: {e}")
            return {
                "results": [],
                "facets": {},
                "total_results": 0,
                "query": query,
                "error": str(e)
            }
    
    async def similar_content_search(
        self, 
        reference_content: str,
        n_results: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """Find content similar to reference content."""
        try:
            # Use the reference content as query
            search_query = SearchQuery(
                text=reference_content,
                strategy=SearchStrategy.SEMANTIC_ONLY,
                n_results=n_results + (1 if exclude_self else 0)
            )
            
            results = await self.enhanced_search(search_query)
            
            # Exclude the reference content itself if requested
            if exclude_self:
                results = [
                    r for r in results 
                    if r.content.strip() != reference_content.strip()
                ]
            
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in similar content search: {e}")
            return []
    
    def get_search_analytics(self, limit: int = 100) -> Dict[str, Any]:
        """Get search analytics and performance metrics."""
        recent_analytics = self.search_analytics[-limit:] if self.search_analytics else []
        
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "recent_queries": recent_analytics,
            "cache_stats": {
                "total_cached_queries": len(self.query_cache),
                "cache_hit_rate": (
                    self.performance_metrics["cache_hits"] / 
                    max(1, self.performance_metrics["total_queries"])
                ),
                "cache_size_mb": len(str(self.query_cache)) / (1024 * 1024)
            },
            "popular_queries": dict(self.performance_metrics["popular_queries"])
        }
    
    def clear_analytics(self):
        """Clear search analytics data."""
        self.search_analytics.clear()
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0,
            "popular_queries": defaultdict(int)
        }
        logger.info("Cleared search analytics data")
    
    def clear_cache(self):
        """Clear query result cache."""
        self.query_cache.clear()
        logger.info("Cleared query result cache")


# Backward compatibility alias
EnhancedVectorStore = AdvancedVectorStoreService