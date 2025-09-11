"""Integration helpers for Enhanced Vector Store with Excel Processing Pipeline.

This module provides seamless integration between the enhanced vector store
and the unified Excel processor, enabling rich metadata indexing and 
intelligent content organization.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from .enhanced_vector_store import (
    AdvancedVectorStoreService, 
    SearchQuery, 
    SearchStrategy,
    RelevanceScoring
)
from .excel_processor import OptimizedExcelProcessor

logger = logging.getLogger(__name__)


class VectorStoreIntegrator:
    """Integrates enhanced vector store with Excel processing pipeline."""
    
    def __init__(self, 
                 vector_store: AdvancedVectorStoreService,
                 excel_processor: OptimizedExcelProcessor):
        """Initialize the integrator.
        
        Args:
            vector_store: Enhanced vector store service
            excel_processor: Unified Excel processor
        """
        self.vector_store = vector_store
        self.excel_processor = excel_processor
        
    async def process_and_index_file(
        self, 
        file_path: str,
        analysis_mode: str = "comprehensive",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process Excel file and index with enhanced metadata.
        
        Args:
            file_path: Path to Excel file
            analysis_mode: Excel processing mode (basic, comprehensive, auto)
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with processing and indexing results
        """
        try:
            if progress_callback:
                await progress_callback(0, 100, "Starting Excel processing...")
            
            # Process Excel file with enhanced analysis
            excel_result = await asyncio.to_thread(
                self.excel_processor.process_excel_file,
                file_path,
                analysis_mode=analysis_mode
            )
            
            if progress_callback:
                await progress_callback(50, 100, "Excel processing complete, starting indexing...")
            
            # Enhanced metadata enrichment for vector store
            enriched_sheets = self._enrich_sheets_metadata(
                excel_result["sheets"], 
                excel_result
            )
            
            # Index in vector store
            success = await self.vector_store.add_excel_data(
                file_name=excel_result["file_name"],
                file_hash=excel_result["file_hash"],
                sheets_data=enriched_sheets
            )
            
            if progress_callback:
                await progress_callback(100, 100, "Indexing complete!")
            
            return {
                "success": success,
                "file_info": {
                    "name": excel_result["file_name"],
                    "hash": excel_result["file_hash"],
                    "sheets": len(enriched_sheets),
                    "total_chunks": sum(
                        len(sheet["text_chunks"]) 
                        for sheet in enriched_sheets.values()
                    )
                },
                "processing_result": excel_result
            }
            
        except Exception as e:
            logger.error(f"Error processing and indexing file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_info": None,
                "processing_result": None
            }
    
    def _enrich_sheets_metadata(
        self, 
        sheets_data: Dict[str, Any], 
        file_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich sheet metadata for enhanced vector search."""
        enriched_sheets = {}
        
        for sheet_name, sheet_data in sheets_data.items():
            metadata = sheet_data.get("metadata", {})
            
            # Add file-level metadata
            enriched_metadata = {
                **metadata,
                "file_size_mb": file_data.get("file_size_mb", 0),
                "file_extension": file_data.get("extension", ""),
                "processing_mode": file_data.get("processing_mode", "unknown"),
                "last_modified": file_data.get("last_modified", ""),
                "processed_at": datetime.now().isoformat()
            }
            
            # Add enhanced analysis results if available
            if "data_quality" in metadata:
                enriched_metadata["has_quality_analysis"] = True
                enriched_metadata["overall_quality"] = metadata["data_quality"].get("overall_quality", "unknown")
                enriched_metadata["completeness_score"] = metadata["data_quality"].get("completeness_score", 0)
            
            if "patterns_detected" in metadata:
                enriched_metadata["has_patterns"] = True
                enriched_metadata["pattern_types"] = list(metadata["patterns_detected"].keys())
            
            if "relationships" in metadata:
                enriched_metadata["has_relationships"] = True
                enriched_metadata["correlation_count"] = len(metadata["relationships"].get("correlations", []))
            
            # Create enhanced text chunks with context
            enhanced_chunks = self._create_enhanced_text_chunks(
                sheet_data.get("text_chunks", []),
                enriched_metadata,
                sheet_name
            )
            
            enriched_sheets[sheet_name] = {
                "metadata": enriched_metadata,
                "text_chunks": enhanced_chunks,
                "data": sheet_data.get("data", [])
            }
        
        return enriched_sheets
    
    def _create_enhanced_text_chunks(
        self, 
        original_chunks: List[str], 
        metadata: Dict[str, Any],
        sheet_name: str
    ) -> List[str]:
        """Create enhanced text chunks with additional context."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(original_chunks):
            # Add contextual information to chunks
            if i == 0:  # First chunk is usually summary
                context_prefix = (
                    f"High-quality data sheet '{sheet_name}' "
                    f"(Quality: {metadata.get('overall_quality', 'unknown')}, "
                    f"Completeness: {metadata.get('completeness_score', 0):.1%}) - "
                )
                enhanced_chunk = context_prefix + chunk
            else:
                enhanced_chunk = chunk
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    async def intelligent_search(
        self, 
        query: str,
        context_filters: Optional[Dict[str, Any]] = None,
        search_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform intelligent search with context-aware optimization.
        
        Args:
            query: Search query
            context_filters: Contextual filters (quality, patterns, etc.)
            search_preferences: User search preferences
            
        Returns:
            Enhanced search results with intelligent context
        """
        try:
            # Build advanced search query
            search_query = self._build_intelligent_query(
                query, context_filters, search_preferences
            )
            
            # Perform enhanced search
            results = await self.vector_store.enhanced_search(search_query)
            
            # Add intelligent context to results
            contextualized_results = self._add_intelligent_context(results, query)
            
            # Perform faceted search for additional insights
            faceted_results = await self.vector_store.faceted_search(
                query=query,
                facets=["overall_quality", "file_name", "sheet_name", "chunk_type"]
            )
            
            return {
                "primary_results": contextualized_results,
                "faceted_insights": faceted_results["facets"],
                "total_results": len(contextualized_results),
                "search_strategy": search_query.strategy.value,
                "query_analysis": self._analyze_query_intent(query)
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent search: {e}")
            return {
                "primary_results": [],
                "faceted_insights": {},
                "total_results": 0,
                "search_strategy": "error",
                "error": str(e)
            }
    
    def _build_intelligent_query(
        self, 
        query: str,
        context_filters: Optional[Dict[str, Any]] = None,
        search_preferences: Optional[Dict[str, Any]] = None
    ) -> SearchQuery:
        """Build intelligent search query with context awareness."""
        # Analyze query intent
        intent_analysis = self._analyze_query_intent(query)
        
        # Choose optimal strategy based on intent
        if intent_analysis["is_analytical"]:
            strategy = SearchStrategy.SEMANTIC_ONLY
            scoring = RelevanceScoring.QUALITY_WEIGHTED
        elif intent_analysis["is_specific"]:
            strategy = SearchStrategy.KEYWORD_ONLY
            scoring = RelevanceScoring.HYBRID_WEIGHTED
        else:
            strategy = SearchStrategy.ADAPTIVE
            scoring = RelevanceScoring.CONTEXT_AWARE
        
        # Build filters
        filters = {}
        if context_filters:
            # Quality filters
            if "min_quality" in context_filters:
                # This will be handled in post-processing
                pass
            
            # Pattern filters
            if "required_patterns" in context_filters:
                # This will be handled in post-processing
                pass
            
            # Standard filters
            for key in ["file_name", "sheet_name"]:
                if key in context_filters:
                    filters[key] = context_filters[key]
        
        # Search preferences
        n_results = 10
        min_relevance = 0.0
        include_explanation = False
        
        if search_preferences:
            n_results = search_preferences.get("max_results", 10)
            min_relevance = search_preferences.get("min_relevance", 0.0)
            include_explanation = search_preferences.get("explain_results", False)
        
        return SearchQuery(
            text=query,
            strategy=strategy,
            scoring=scoring,
            n_results=n_results,
            min_relevance=min_relevance,
            filters=filters,
            include_explanation=include_explanation
        )
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent for intelligent search optimization."""
        query_lower = query.lower()
        
        # Analytical intent indicators
        analytical_terms = [
            "analyze", "compare", "trend", "pattern", "correlation",
            "insight", "summary", "overview", "distribution", "relationship"
        ]
        
        # Specific search intent indicators
        specific_terms = [
            "find", "show", "get", "where", "which", "what",
            "exact", "specific", "contains", "equals"
        ]
        
        # Aggregation intent indicators
        aggregation_terms = [
            "total", "sum", "average", "mean", "count", "maximum",
            "minimum", "aggregate", "group by"
        ]
        
        is_analytical = any(term in query_lower for term in analytical_terms)
        is_specific = any(term in query_lower for term in specific_terms)
        is_aggregation = any(term in query_lower for term in aggregation_terms)
        
        return {
            "is_analytical": is_analytical,
            "is_specific": is_specific,
            "is_aggregation": is_aggregation,
            "query_complexity": len(query.split()),
            "has_numbers": any(char.isdigit() for char in query),
            "intent_confidence": 0.8 if (is_analytical or is_specific) else 0.5
        }
    
    def _add_intelligent_context(
        self, 
        results: List[Any], 
        original_query: str
    ) -> List[Dict[str, Any]]:
        """Add intelligent context to search results."""
        contextualized_results = []
        
        for result in results:
            # Convert SearchResult to dict if needed
            if hasattr(result, '__dict__'):
                result_dict = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "relevance_score": result.relevance_score,
                    "semantic_score": result.semantic_score,
                    "keyword_score": result.keyword_score,
                    "quality_score": result.quality_score,
                    "freshness_score": result.freshness_score,
                    "file_name": result.file_name,
                    "sheet_name": result.sheet_name,
                    "chunk_index": result.chunk_index,
                    "explanation": result.explanation
                }
            else:
                result_dict = result
            
            # Add intelligent context
            context = self._generate_result_context(result_dict, original_query)
            result_dict["intelligent_context"] = context
            
            contextualized_results.append(result_dict)
        
        return contextualized_results
    
    def _generate_result_context(
        self, 
        result: Dict[str, Any], 
        query: str
    ) -> Dict[str, Any]:
        """Generate intelligent context for a search result."""
        metadata = result.get("metadata", {})
        
        # Data quality context
        quality_context = "Unknown quality"
        if "overall_quality" in metadata:
            quality_level = metadata["overall_quality"]
            quality_context = f"Data quality: {quality_level}"
        
        # Content type context
        content_type = "General data"
        if "chunk_type" in metadata:
            chunk_type = metadata["chunk_type"]
            if chunk_type == "summary":
                content_type = "Summary information"
            elif chunk_type == "data":
                content_type = "Detailed data"
        
        # Relevance context
        relevance_level = "Medium"
        relevance_score = result.get("relevance_score", 0.5)
        if relevance_score > 0.8:
            relevance_level = "High"
        elif relevance_score < 0.3:
            relevance_level = "Low"
        
        # Pattern context
        pattern_context = ""
        if metadata.get("has_patterns"):
            pattern_types = metadata.get("pattern_types", [])
            if pattern_types:
                pattern_context = f"Contains patterns: {', '.join(pattern_types[:3])}"
        
        return {
            "quality_context": quality_context,
            "content_type": content_type,
            "relevance_level": relevance_level,
            "pattern_context": pattern_context,
            "source_info": f"From {result.get('file_name', 'unknown')} - {result.get('sheet_name', 'unknown')}",
            "why_relevant": self._explain_relevance(result, query)
        }
    
    def _explain_relevance(self, result: Dict[str, Any], query: str) -> str:
        """Explain why a result is relevant to the query."""
        explanations = []
        
        # Semantic relevance
        semantic_score = result.get("semantic_score", 0)
        if semantic_score > 0.7:
            explanations.append("Strong semantic match")
        elif semantic_score > 0.4:
            explanations.append("Moderate semantic match")
        
        # Keyword relevance
        keyword_score = result.get("keyword_score", 0)
        if keyword_score > 0.3:
            explanations.append("Contains matching keywords")
        
        # Quality relevance
        quality_score = result.get("quality_score", 0)
        if quality_score > 0.8:
            explanations.append("High-quality data source")
        
        # Freshness relevance
        freshness_score = result.get("freshness_score", 0)
        if freshness_score > 0.7:
            explanations.append("Recently updated content")
        
        if not explanations:
            explanations.append("General content match")
        
        return "; ".join(explanations)
    
    async def batch_process_directory(
        self, 
        directory_path: str,
        analysis_mode: str = "auto",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process and index all Excel files in a directory.
        
        Args:
            directory_path: Path to directory containing Excel files
            analysis_mode: Excel processing mode
            progress_callback: Optional progress callback
            
        Returns:
            Batch processing results
        """
        try:
            # Get all Excel files in directory
            excel_files = []
            for ext in [".xlsx", ".xls", ".xlsm"]:
                excel_files.extend(
                    self.excel_processor.data_directory.glob(f"**/*{ext}")
                )
            
            total_files = len(excel_files)
            successful_files = 0
            failed_files = []
            processing_results = []
            
            for i, file_path in enumerate(excel_files):
                try:
                    if progress_callback:
                        await progress_callback(
                            i, total_files, 
                            f"Processing {file_path.name}..."
                        )
                    
                    result = await self.process_and_index_file(
                        str(file_path), analysis_mode
                    )
                    
                    if result["success"]:
                        successful_files += 1
                        processing_results.append({
                            "file": file_path.name,
                            "status": "success",
                            "info": result["file_info"]
                        })
                    else:
                        failed_files.append({
                            "file": file_path.name,
                            "error": result.get("error", "Unknown error")
                        })
                        processing_results.append({
                            "file": file_path.name,
                            "status": "failed",
                            "error": result.get("error", "Unknown error")
                        })
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    failed_files.append({
                        "file": file_path.name,
                        "error": str(e)
                    })
            
            if progress_callback:
                await progress_callback(
                    total_files, total_files, 
                    f"Completed: {successful_files}/{total_files} files"
                )
            
            return {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": len(failed_files),
                "success_rate": successful_files / max(1, total_files),
                "processing_results": processing_results,
                "failed_details": failed_files
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {
                "total_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "success_rate": 0.0,
                "error": str(e)
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics and health metrics."""
        try:
            vector_stats = self.vector_store.get_statistics()
            excel_stats = self.excel_processor.get_file_statistics()
            search_analytics = self.vector_store.get_search_analytics()
            
            return {
                "vector_store_stats": vector_stats,
                "excel_processor_stats": excel_stats,
                "search_analytics": search_analytics,
                "integration_health": {
                    "vector_store_healthy": vector_stats.get("total_documents", 0) > 0,
                    "excel_processor_healthy": excel_stats.get("total_files", 0) > 0,
                    "integration_active": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting integration stats: {e}")
            return {
                "error": str(e),
                "integration_health": {
                    "vector_store_healthy": False,
                    "excel_processor_healthy": False,
                    "integration_active": False
                }
            }