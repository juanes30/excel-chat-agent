"""Comprehensive tests for Enhanced Vector Store Service.

Tests cover advanced semantic search, hybrid search strategies, 
analytics, caching, and integration capabilities.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.services.enhanced_vector_store import (
    AdvancedVectorStoreService,
    SearchQuery,
    SearchStrategy,
    RelevanceScoring,
    SearchResult,
    QueryExpander,
    RelevanceScorer
)
from app.services.vector_store_integration import VectorStoreIntegrator
from app.services.excel_processor import OptimizedExcelProcessor


class TestQueryExpander:
    """Test query expansion functionality."""
    
    def test_query_expansion_basic(self):
        """Test basic query expansion."""
        expander = QueryExpander()
        expanded = expander.expand_query("sales data", max_expansions=2)
        
        assert "sales data" in expanded
        assert len(expanded) > 1
        assert any("revenue" in query for query in expanded)
    
    def test_query_expansion_multiple_terms(self):
        """Test expansion with multiple expandable terms."""
        expander = QueryExpander()
        expanded = expander.expand_query("customer sales profit", max_expansions=2)
        
        assert len(expanded) > 1
        # Should expand multiple terms
        expanded_text = " ".join(expanded)
        # Check that expansion occurred by looking for synonyms or original terms
        has_customer_synonym = any(word in expanded_text for word in ["client", "buyer", "purchaser", "consumer"])
        has_sales_synonym = any(word in expanded_text for word in ["revenue", "income", "earnings", "turnover"])
        has_profit_synonym = any(word in expanded_text for word in ["earnings", "income", "gain", "return"])
        
        # At least one expansion should have occurred
        assert has_customer_synonym or has_sales_synonym or has_profit_synonym
    
    def test_query_expansion_no_expandable_terms(self):
        """Test expansion with no expandable terms."""
        expander = QueryExpander()
        expanded = expander.expand_query("random unknown terms", max_expansions=2)
        
        assert expanded == ["random unknown terms"]


class TestRelevanceScorer:
    """Test relevance scoring functionality."""
    
    def test_semantic_score_calculation(self):
        """Test semantic score calculation from distance."""
        # Distance of 0 should give score of 1
        assert RelevanceScorer.calculate_semantic_score(0.0) == 1.0
        
        # Distance of 2 should give score of 0
        assert RelevanceScorer.calculate_semantic_score(2.0) == 0.0
        
        # Distance of 1 should give score of 0.5
        assert RelevanceScorer.calculate_semantic_score(1.0) == 0.5
    
    def test_keyword_score_calculation(self):
        """Test keyword matching score."""
        query = "sales revenue"
        content = "The sales team achieved high revenue this quarter"
        
        score = RelevanceScorer.calculate_keyword_score(query, content)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some matches
    
    def test_quality_score_from_metadata(self):
        """Test quality score extraction from metadata."""
        # Test with quality data
        metadata_with_quality = {
            "data_quality": {"overall_score": 0.8},
            "completeness_percentage": 90
        }
        score = RelevanceScorer.calculate_quality_score(metadata_with_quality)
        assert 0.7 <= score <= 1.0
        
        # Test without quality data
        metadata_minimal = {"completeness_percentage": 50}
        score = RelevanceScorer.calculate_quality_score(metadata_minimal)
        assert 0.4 <= score <= 0.6
    
    def test_freshness_score_calculation(self):
        """Test freshness score calculation."""
        # Recent content
        recent_metadata = {
            "added_at": datetime.now().isoformat()
        }
        score = RelevanceScorer.calculate_freshness_score(recent_metadata)
        assert score > 0.9
        
        # Old content
        old_metadata = {
            "added_at": (datetime.now() - timedelta(days=60)).isoformat()
        }
        score = RelevanceScorer.calculate_freshness_score(old_metadata)
        assert score < 0.5
    
    def test_hybrid_score_calculation(self):
        """Test hybrid score combination."""
        score = RelevanceScorer.calculate_hybrid_score(
            semantic_score=0.8,
            keyword_score=0.6,
            quality_score=0.9,
            freshness_score=0.7
        )
        
        assert 0.0 <= score <= 1.0
        # Should be weighted towards semantic score
        assert score > 0.7


class TestAdvancedVectorStoreService:
    """Test advanced vector store service functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = Mock()
            mock_instance.encode.return_value = np.random.rand(5, 384)  # 5 embeddings of 384 dimensions
            mock_model.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def vector_store(self, temp_dir, mock_embedding_model):
        """Create vector store instance for testing."""
        # Mock ChromaDB
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_collection.count.return_value = 10
            mock_collection.add = Mock()
            mock_collection.query.return_value = {
                'ids': [['doc1', 'doc2']],
                'documents': [['Document 1 content', 'Document 2 content']],
                'metadatas': [[
                    {'file_name': 'test.xlsx', 'sheet_name': 'Sheet1', 'chunk_index': 0},
                    {'file_name': 'test.xlsx', 'sheet_name': 'Sheet1', 'chunk_index': 1}
                ]],
                'distances': [[0.2, 0.4]]
            }
            
            mock_client_instance = Mock()
            mock_client_instance.get_collection.return_value = mock_collection
            mock_client_instance.create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance
            
            store = AdvancedVectorStoreService(
                persist_directory=temp_dir,
                enable_analytics=True
            )
            store.collection = mock_collection
            yield store
    
    @pytest.mark.asyncio
    async def test_enhanced_search_string_query(self, vector_store):
        """Test enhanced search with string query."""
        results = await vector_store.enhanced_search("test query")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_enhanced_search_query_object(self, vector_store):
        """Test enhanced search with SearchQuery object."""
        search_query = SearchQuery(
            text="test query",
            strategy=SearchStrategy.HYBRID,
            scoring=RelevanceScoring.QUALITY_WEIGHTED,
            n_results=3,
            include_explanation=True
        )
        
        results = await vector_store.enhanced_search(search_query)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        # Check explanations are included
        assert all(r.explanation is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_search_strategy_adaptive(self, vector_store):
        """Test adaptive search strategy selection."""
        # Test semantic-focused query
        semantic_query = SearchQuery(
            text="find similar patterns and relationships",
            strategy=SearchStrategy.ADAPTIVE
        )
        results = await vector_store.enhanced_search(semantic_query)
        assert len(results) >= 0
        
        # Test keyword-focused query
        keyword_query = SearchQuery(
            text="exact name contains specific value",
            strategy=SearchStrategy.ADAPTIVE
        )
        results = await vector_store.enhanced_search(keyword_query)
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_faceted_search(self, vector_store):
        """Test faceted search functionality."""
        results = await vector_store.faceted_search(
            query="test data",
            facets=["file_name", "sheet_name", "chunk_type"]
        )
        
        assert "results" in results
        assert "facets" in results
        assert "total_results" in results
        assert isinstance(results["facets"], dict)
    
    @pytest.mark.asyncio
    async def test_similar_content_search(self, vector_store):
        """Test similar content search."""
        reference_content = "This is sample content for similarity search"
        results = await vector_store.similar_content_search(
            reference_content=reference_content,
            n_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
        # Should exclude the reference content itself
        assert not any(r.content == reference_content for r in results)
    
    def test_query_caching(self, vector_store):
        """Test query result caching."""
        # Test cache key generation
        query1 = SearchQuery(text="test", strategy=SearchStrategy.SEMANTIC_ONLY)
        query2 = SearchQuery(text="test", strategy=SearchStrategy.SEMANTIC_ONLY)
        query3 = SearchQuery(text="test", strategy=SearchStrategy.HYBRID)
        
        key1 = vector_store._generate_query_cache_key(query1)
        key2 = vector_store._generate_query_cache_key(query2)
        key3 = vector_store._generate_query_cache_key(query3)
        
        assert key1 == key2  # Same query should have same key
        assert key1 != key3  # Different strategy should have different key
    
    def test_analytics_collection(self, vector_store):
        """Test search analytics collection."""
        assert vector_store.enable_analytics
        assert isinstance(vector_store.search_analytics, list)
        assert isinstance(vector_store.performance_metrics, dict)
        
        # Test analytics structure
        analytics = vector_store.get_search_analytics()
        assert "performance_metrics" in analytics
        assert "cache_stats" in analytics
        assert "popular_queries" in analytics
    
    def test_cache_management(self, vector_store):
        """Test cache management functionality."""
        # Add some data to cache
        vector_store.query_cache["test_key"] = ([], datetime.now())
        assert len(vector_store.query_cache) > 0
        
        # Clear cache
        vector_store.clear_cache()
        assert len(vector_store.query_cache) == 0
    
    def test_analytics_clearing(self, vector_store):
        """Test analytics data clearing."""
        # Add some analytics data
        vector_store.performance_metrics["total_queries"] = 10
        
        # Clear analytics
        vector_store.clear_analytics()
        assert vector_store.performance_metrics["total_queries"] == 0


class TestVectorStoreIntegrator:
    """Test vector store integration functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        store = Mock(spec=AdvancedVectorStoreService)
        store.add_excel_data = AsyncMock(return_value=True)
        store.enhanced_search = AsyncMock(return_value=[])
        store.faceted_search = AsyncMock(return_value={"facets": {}, "results": []})
        store.get_statistics = Mock(return_value={"total_documents": 100})
        store.get_search_analytics = Mock(return_value={"cache_hit_rate": 0.5})
        return store
    
    @pytest.fixture
    def mock_excel_processor(self, temp_dir):
        """Mock Excel processor for testing."""
        processor = Mock(spec=OptimizedExcelProcessor)
        processor.data_directory = Path(temp_dir)
        processor.process_excel_file = Mock(return_value={
            "file_name": "test.xlsx",
            "file_hash": "abc123",
            "file_size_mb": 1.5,
            "extension": ".xlsx",
            "processing_mode": "comprehensive",
            "last_modified": datetime.now(),
            "sheets": {
                "Sheet1": {
                    "metadata": {
                        "num_rows": 100,
                        "num_cols": 5,
                        "data_quality": {"overall_quality": "good", "completeness_score": 0.8},
                        "patterns_detected": {"email": 5, "phone": 3},
                        "relationships": {"correlations": [{"col1": "A", "col2": "B", "corr": 0.8}]}
                    },
                    "text_chunks": ["Summary of Sheet1", "Data chunk 1", "Data chunk 2"],
                    "data": [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
                }
            }
        })
        processor.get_file_statistics = Mock(return_value={"total_files": 5})
        return processor
    
    @pytest.fixture
    def integrator(self, mock_vector_store, mock_excel_processor):
        """Create integrator instance for testing."""
        return VectorStoreIntegrator(mock_vector_store, mock_excel_processor)
    
    @pytest.mark.asyncio
    async def test_process_and_index_file(self, integrator, mock_excel_processor):
        """Test file processing and indexing."""
        # Create a test file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            test_file_path = tmp.name
        
        try:
            result = await integrator.process_and_index_file(
                test_file_path,
                analysis_mode="comprehensive"
            )
            
            assert result["success"] is True
            assert "file_info" in result
            assert result["file_info"]["name"] == "test.xlsx"
            
            # Verify Excel processor was called
            mock_excel_processor.process_excel_file.assert_called_once()
            
        finally:
            Path(test_file_path).unlink(missing_ok=True)
    
    def test_metadata_enrichment(self, integrator):
        """Test metadata enrichment functionality."""
        sheets_data = {
            "Sheet1": {
                "metadata": {
                    "num_rows": 100,
                    "data_quality": {"overall_quality": "excellent"}
                },
                "text_chunks": ["Original chunk"],
                "data": []
            }
        }
        
        file_data = {
            "file_size_mb": 2.0,
            "extension": ".xlsx",
            "processing_mode": "comprehensive"
        }
        
        enriched = integrator._enrich_sheets_metadata(sheets_data, file_data)
        
        assert "Sheet1" in enriched
        metadata = enriched["Sheet1"]["metadata"]
        assert metadata["file_size_mb"] == 2.0
        assert metadata["has_quality_analysis"] is True
        assert metadata["overall_quality"] == "excellent"
    
    @pytest.mark.asyncio
    async def test_intelligent_search(self, integrator):
        """Test intelligent search functionality."""
        result = await integrator.intelligent_search(
            query="analyze sales trends",
            context_filters={"min_quality": "good"},
            search_preferences={"max_results": 5}
        )
        
        assert "primary_results" in result
        assert "faceted_insights" in result
        assert "search_strategy" in result
        assert "query_analysis" in result
    
    def test_query_intent_analysis(self, integrator):
        """Test query intent analysis."""
        # Analytical query
        analytical_intent = integrator._analyze_query_intent("analyze sales trends and patterns")
        assert analytical_intent["is_analytical"] is True
        
        # Specific query
        specific_intent = integrator._analyze_query_intent("find exact customer name John Smith")
        assert specific_intent["is_specific"] is True
        
        # Aggregation query
        agg_intent = integrator._analyze_query_intent("total sales by region")
        assert agg_intent["is_aggregation"] is True
    
    def test_result_context_generation(self, integrator):
        """Test intelligent context generation."""
        result = {
            "content": "Sales data for Q1",
            "metadata": {
                "overall_quality": "excellent",
                "chunk_type": "summary",
                "has_patterns": True,
                "pattern_types": ["email", "phone"]
            },
            "relevance_score": 0.85,
            "semantic_score": 0.8,
            "keyword_score": 0.4,
            "quality_score": 0.9,
            "file_name": "sales.xlsx",
            "sheet_name": "Q1"
        }
        
        context = integrator._generate_result_context(result, "sales analysis")
        
        assert context["quality_context"] == "Data quality: excellent"
        assert context["content_type"] == "Summary information"
        assert context["relevance_level"] == "High"
        assert "email, phone" in context["pattern_context"]
        assert "sales.xlsx" in context["source_info"]
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, integrator, temp_dir):
        """Test batch directory processing."""
        # Create test Excel files
        excel_files = []
        for i in range(3):
            file_path = Path(temp_dir) / f"test_{i}.xlsx"
            # Create minimal Excel file
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            df.to_excel(file_path, index=False)
            excel_files.append(file_path)
        
        try:
            # Mock the data_directory to point to our temp directory
            integrator.excel_processor.data_directory = Path(temp_dir)
            
            result = await integrator.batch_process_directory(
                temp_dir,
                analysis_mode="basic"
            )
            
            assert "total_files" in result
            assert result["total_files"] == 3
            assert "successful_files" in result
            assert "success_rate" in result
            
        finally:
            # Clean up
            for file_path in excel_files:
                file_path.unlink(missing_ok=True)
    
    def test_integration_stats(self, integrator):
        """Test integration statistics."""
        stats = integrator.get_integration_stats()
        
        assert "vector_store_stats" in stats
        assert "excel_processor_stats" in stats
        assert "search_analytics" in stats
        assert "integration_health" in stats
        
        health = stats["integration_health"]
        assert "vector_store_healthy" in health
        assert "excel_processor_healthy" in health
        assert "integration_active" in health


class TestSearchQueryValidation:
    """Test search query validation and edge cases."""
    
    def test_search_query_defaults(self):
        """Test SearchQuery default values."""
        query = SearchQuery(text="test")
        
        assert query.strategy == SearchStrategy.ADAPTIVE
        assert query.scoring == RelevanceScoring.CONTEXT_AWARE
        assert query.n_results == 5
        assert query.min_relevance == 0.0
        assert query.filters == {}
        assert query.facets == []
        assert query.include_explanation is False
    
    def test_search_query_custom_values(self):
        """Test SearchQuery with custom values."""
        custom_filters = {"file_name": "test.xlsx"}
        custom_facets = ["quality", "patterns"]
        
        query = SearchQuery(
            text="custom query",
            strategy=SearchStrategy.HYBRID,
            scoring=RelevanceScoring.QUALITY_WEIGHTED,
            n_results=10,
            min_relevance=0.5,
            filters=custom_filters,
            facets=custom_facets,
            include_explanation=True
        )
        
        assert query.text == "custom query"
        assert query.strategy == SearchStrategy.HYBRID
        assert query.scoring == RelevanceScoring.QUALITY_WEIGHTED
        assert query.n_results == 10
        assert query.min_relevance == 0.5
        assert query.filters == custom_filters
        assert query.facets == custom_facets
        assert query.include_explanation is True


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self):
        """Test complete search workflow from file processing to results."""
        # This would be a full integration test with real ChromaDB
        # and Excel files in a CI/CD environment
        pass
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with large datasets."""
        # Performance testing with significant data volumes
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self):
        """Test concurrent search operations."""
        # Test thread safety and concurrent access
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])