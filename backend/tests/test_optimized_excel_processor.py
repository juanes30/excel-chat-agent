"""Comprehensive tests for the OptimizedExcelProcessor.

Tests cover memory efficiency, parallel processing, chunking strategies,
and performance optimization features.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.services.excel_processor import (
    OptimizedExcelProcessor,
    ProcessingProgress,
    MemoryEfficientDataFrame,
    ExcelProcessor  # Backward compatibility alias
)


class TestProcessingProgress:
    """Test the ProcessingProgress tracking class."""
    
    def test_initialization(self):
        """Test progress initialization."""
        progress = ProcessingProgress(total_steps=10)
        assert progress.total_steps == 10
        assert progress.current_step == 0
        assert progress.status == "initialized"
        assert len(progress.callbacks) == 0
    
    def test_add_callback(self):
        """Test adding progress callbacks."""
        progress = ProcessingProgress(10)
        
        def callback(current, total, status):
            pass
        
        progress.add_callback(callback)
        assert len(progress.callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_update_async(self):
        """Test asynchronous progress updates."""
        progress = ProcessingProgress(10)
        
        callback_calls = []
        
        async def async_callback(current, total, status):
            callback_calls.append((current, total, status))
        
        progress.add_callback(async_callback)
        
        await progress.update(5, "Processing")
        assert progress.current_step == 5
        assert progress.status == "Processing"
        assert len(callback_calls) == 1
        assert callback_calls[0] == (5, 10, "Processing")
    
    def test_increment(self):
        """Test progress increment."""
        progress = ProcessingProgress(10)
        
        result = progress.increment("Step 1")
        assert result is progress  # Should return self for chaining
        assert progress.current_step == 1
        assert progress.status == "Step 1"


class TestMemoryEfficientDataFrame:
    """Test the MemoryEfficientDataFrame wrapper."""
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create a sample Excel file for testing."""
        # Create test data
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'] * 1000,  # 5000 rows
            'Age': np.random.randint(20, 80, 5000),
            'Score': np.random.uniform(0, 100, 5000),
            'Date': pd.date_range('2020-01-01', periods=5000, freq='H')
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df.to_excel(tmp.name, sheet_name='TestSheet', index=False)
            yield Path(tmp.name)
        
        # Cleanup
        os.unlink(tmp.name)
    
    def test_initialization(self, sample_excel_file):
        """Test MemoryEfficientDataFrame initialization."""
        mem_df = MemoryEfficientDataFrame(sample_excel_file, 'TestSheet', chunk_size=1000)
        
        assert mem_df.file_path == sample_excel_file
        assert mem_df.sheet_name == 'TestSheet'
        assert mem_df.chunk_size == 1000
        assert mem_df._cached_info is None
    
    def test_info_property(self, sample_excel_file):
        """Test info property extraction."""
        mem_df = MemoryEfficientDataFrame(sample_excel_file, 'TestSheet')
        
        info = mem_df.info
        assert isinstance(info, dict)
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'memory_estimate_mb' in info
        
        # Should cache the result
        assert mem_df._cached_info is not None
        
        # Second call should return cached result
        info2 = mem_df.info
        assert info == info2
    
    @pytest.mark.asyncio
    async def test_to_parquet_async(self, sample_excel_file):
        """Test asynchronous Parquet conversion."""
        mem_df = MemoryEfficientDataFrame(sample_excel_file, 'TestSheet')
        
        parquet_path = await mem_df.to_parquet_async()
        
        assert parquet_path.exists()
        assert parquet_path.suffix == '.parquet'
        assert mem_df._temp_parquet_path == parquet_path
        
        # Second call should return cached path
        parquet_path2 = await mem_df.to_parquet_async()
        assert parquet_path == parquet_path2
    
    @pytest.mark.asyncio
    async def test_iter_chunks(self, sample_excel_file):
        """Test chunk iteration."""
        mem_df = MemoryEfficientDataFrame(sample_excel_file, 'TestSheet', chunk_size=1000)
        
        chunks = []
        async for chunk in mem_df.iter_chunks():
            chunks.append(chunk)
            # Limit to avoid long test
            if len(chunks) >= 3:
                break
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 1000  # Should respect chunk size
    
    def test_optimize_dtypes(self, sample_excel_file):
        """Test data type optimization."""
        mem_df = MemoryEfficientDataFrame(sample_excel_file, 'TestSheet')
        
        # Create test DataFrame with suboptimal types
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'category_col': ['A', 'B', 'A', 'B', 'A'],  # Low cardinality
            'text_col': ['unique1', 'unique2', 'unique3', 'unique4', 'unique5']  # High cardinality
        })
        
        optimized_df = mem_df._optimize_dtypes(df)
        
        # Check that optimization occurred
        assert optimized_df.dtypes['category_col'] == 'category'  # Should be converted to category
        # Integer and float columns might be downcasted
        assert optimized_df.memory_usage(deep=True).sum() <= df.memory_usage(deep=True).sum()


class TestOptimizedExcelProcessor:
    """Test the OptimizedExcelProcessor class."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def processor(self, temp_directory):
        """Create an OptimizedExcelProcessor instance."""
        return OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            max_file_size_mb=50,
            chunk_size=1000,
            max_cache_size_mb=100,
            enable_parallel_processing=True
        )
    
    @pytest.fixture
    def small_excel_file(self, temp_directory):
        """Create a small Excel file for testing."""
        data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Score': [85.5, 92.0, 78.5]
        }
        df = pd.DataFrame(data)
        
        file_path = temp_directory / "small_test.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def large_excel_file(self, temp_directory):
        """Create a large Excel file for testing chunked processing."""
        # Create a file that's considered "large" for testing
        data = {
            'ID': range(10000),
            'Name': [f'Person_{i}' for i in range(10000)],
            'Value': np.random.uniform(0, 100, 10000),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'Date': pd.date_range('2020-01-01', periods=10000, freq='H')
        }
        df = pd.DataFrame(data)
        
        file_path = temp_directory / "large_test.xlsx"
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            # Add a second sheet
            df.head(1000).to_excel(writer, sheet_name='Sheet2', index=False)
        
        return file_path
    
    def test_initialization(self, temp_directory):
        """Test processor initialization."""
        processor = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            max_file_size_mb=100,
            chunk_size=2000,
            enable_parallel_processing=False
        )
        
        assert processor.data_directory == temp_directory
        assert processor.max_file_size_mb == 100
        assert processor.chunk_size == 2000
        assert processor.enable_parallel_processing is False
        assert processor.temp_dir.exists()
    
    @pytest.mark.asyncio
    async def test_get_file_hash_async(self, processor, small_excel_file):
        """Test asynchronous file hash generation."""
        hash1 = await processor.get_file_hash_async(small_excel_file)
        hash2 = await processor.get_file_hash_async(small_excel_file)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        assert isinstance(hash1, str)
    
    def test_validate_excel_file(self, processor, small_excel_file):
        """Test Excel file validation."""
        result = processor.validate_excel_file(small_excel_file)
        
        assert result['file_name'] == small_excel_file.name
        assert result['file_size_mb'] > 0
        assert result['extension'] == '.xlsx'
        assert 'last_modified' in result
        assert 'is_large_file' in result
        assert 'requires_streaming' in result
    
    def test_validate_excel_file_large(self, processor, large_excel_file):
        """Test validation of large Excel file."""
        result = processor.validate_excel_file(large_excel_file)
        
        assert 'processing_strategy' in result
        assert 'sheet_count' in result
    
    def test_validate_excel_file_invalid(self, processor, temp_directory):
        """Test validation of invalid files."""
        # Test non-existent file
        with pytest.raises(ValueError, match="File does not exist"):
            processor.validate_excel_file(temp_directory / "nonexistent.xlsx")
        
        # Test unsupported format
        txt_file = temp_directory / "test.txt"
        txt_file.write_text("not an excel file")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.validate_excel_file(txt_file)
    
    @pytest.mark.asyncio
    async def test_cache_memory_management(self, processor):
        """Test cache memory management."""
        # Fill cache with test data
        for i in range(100):
            key = f"test_key_{i}"
            data = f"test_data_{i}" * 1000  # Make data larger
            processor._update_cache(key, data)
        
        initial_cache_size = len(processor._file_cache)
        initial_memory = processor._cache_memory_usage
        
        # Trigger cache cleanup
        await processor._manage_cache_memory()
        
        # Should have cleaned up some entries if memory was high
        # (Exact behavior depends on memory usage threshold)
        assert len(processor._file_cache) <= initial_cache_size
    
    @pytest.mark.asyncio
    async def test_extract_sheet_metadata_async(self, processor):
        """Test asynchronous metadata extraction."""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Score': [85.5, 92.0, 78.5],
            'Date': pd.date_range('2020-01-01', periods=3)
        })
        
        metadata = await processor.extract_sheet_metadata_async(df, 'TestSheet')
        
        assert metadata['sheet_name'] == 'TestSheet'
        assert metadata['num_rows'] == 3
        assert metadata['num_cols'] == 4
        assert len(metadata['columns']) == 4
        assert 'memory_usage_mb' in metadata
        assert 'was_sampled' in metadata
        
        # Check column information
        name_col = next(col for col in metadata['columns'] if col['name'] == 'Name')
        assert name_col['data_type'] == 'text'
        
        age_col = next(col for col in metadata['columns'] if col['name'] == 'Age')
        assert age_col['data_type'] == 'numeric'
    
    @pytest.mark.asyncio
    async def test_create_searchable_text_async(self, processor):
        """Test asynchronous searchable text creation."""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob'],
            'Age': [25, 30],
            'Score': [85.5, 92.0]
        })
        
        metadata = await processor.extract_sheet_metadata_async(df, 'TestSheet')
        text_chunks = await processor.create_searchable_text_async(df, 'TestSheet', metadata)
        
        assert len(text_chunks) > 0
        
        # Should contain sheet summary
        summary_chunk = text_chunks[0]
        assert 'TestSheet' in summary_chunk
        assert '2 rows' in summary_chunk
        
        # Should contain column descriptions
        column_chunks = [chunk for chunk in text_chunks if 'Column' in chunk]
        assert len(column_chunks) >= 3  # One for each column
    
    @pytest.mark.asyncio
    async def test_process_excel_file_async_small(self, processor, small_excel_file):
        """Test processing of small Excel file."""
        result = await processor.process_excel_file_async(small_excel_file)
        
        assert result['file_name'] == small_excel_file.name
        assert result['total_sheets'] == 1
        assert result['total_rows'] == 3
        assert 'sheets' in result
        assert 'all_text_chunks' in result
        assert 'processing_info' in result
        
        # Check processing strategy
        processing_info = result['processing_info']
        assert processing_info['strategy'] == 'standard'
        assert processing_info['parallel_processing'] == processor.enable_parallel_processing
    
    @pytest.mark.asyncio
    async def test_process_excel_file_async_large(self, processor, large_excel_file):
        """Test processing of large Excel file with chunking."""
        # Mock file size to trigger chunked processing
        with patch.object(processor, 'validate_excel_file') as mock_validate:
            mock_validate.return_value = {
                'file_name': large_excel_file.name,
                'file_size_mb': 150,  # Large enough to trigger chunked processing
                'extension': '.xlsx',
                'last_modified': datetime.now(),
                'is_large_file': True,
                'requires_streaming': False,
                'processing_strategy': 'chunked'
            }
            
            result = await processor.process_excel_file_async(large_excel_file)
            
            assert result['file_name'] == large_excel_file.name
            assert result['total_sheets'] >= 1
            assert 'processing_info' in result
            
            processing_info = result['processing_info']
            assert processing_info['strategy'] == 'chunked'
    
    @pytest.mark.asyncio
    async def test_process_all_files_async(self, processor, small_excel_file, temp_directory):
        """Test processing all files asynchronously."""
        # Create another test file
        data = {'X': [1, 2], 'Y': [3, 4]}
        df = pd.DataFrame(data)
        file2 = temp_directory / "test2.xlsx"
        df.to_excel(file2, index=False)
        
        progress_calls = []
        
        def progress_callback(current, total, status):
            progress_calls.append((current, total, status))
        
        results = await processor.process_all_files_async(progress_callback)
        
        assert len(results) == 2
        assert all('file_name' in result for result in results)
        assert len(progress_calls) > 0  # Should have made progress callbacks
    
    @pytest.mark.asyncio
    async def test_get_file_statistics_async(self, processor, small_excel_file):
        """Test getting file statistics asynchronously."""
        # First process the file
        await processor.process_excel_file_async(small_excel_file)
        
        stats = await processor.get_file_statistics_async()
        
        assert stats['total_files'] >= 1
        assert stats['total_sheets'] >= 1
        assert stats['total_rows'] >= 3
        assert 'processing_summary' in stats
        assert 'strategy_distribution' in stats['processing_summary']
        assert 'cache_efficiency' in stats['processing_summary']
    
    def test_backward_compatibility(self):
        """Test that ExcelProcessor is an alias for OptimizedExcelProcessor."""
        assert ExcelProcessor is OptimizedExcelProcessor
    
    def test_cleanup(self, processor):
        """Test processor cleanup."""
        # Add some cache data
        processor._update_cache("test_key", "test_data")
        assert len(processor._file_cache) > 0
        
        processor.cleanup()
        
        assert len(processor._file_cache) == 0
        assert processor._cache_memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_processing(self, temp_directory):
        """Test performance difference between parallel and sequential processing."""
        # Create test data
        data = {f'Col_{i}': np.random.random(100) for i in range(20)}  # 20 columns
        df = pd.DataFrame(data)
        file_path = temp_directory / "parallel_test.xlsx"
        df.to_excel(file_path, index=False)
        
        # Test with parallel processing
        processor_parallel = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            enable_parallel_processing=True
        )
        
        start_time = asyncio.get_event_loop().time()
        result_parallel = await processor_parallel.process_excel_file_async(file_path)
        parallel_time = asyncio.get_event_loop().time() - start_time
        
        # Test with sequential processing
        processor_sequential = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            enable_parallel_processing=False
        )
        
        start_time = asyncio.get_event_loop().time()
        result_sequential = await processor_sequential.process_excel_file_async(file_path)
        sequential_time = asyncio.get_event_loop().time() - start_time
        
        # Both should produce similar results
        assert result_parallel['total_rows'] == result_sequential['total_rows']
        assert result_parallel['total_columns'] == result_sequential['total_columns']
        
        # Parallel might be faster for this case (though not guaranteed in tests)
        print(f"Parallel time: {parallel_time:.3f}s, Sequential time: {sequential_time:.3f}s")


class TestIntegration:
    """Integration tests for the complete optimization pipeline."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def complex_excel_file(self, temp_directory):
        """Create a complex Excel file with multiple sheets and data types."""
        file_path = temp_directory / "complex_test.xlsx"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Sheet 1: Numeric data
            df1 = pd.DataFrame({
                'ID': range(1000),
                'Value': np.random.normal(100, 15, 1000),
                'Category': np.random.choice(['A', 'B', 'C'], 1000)
            })
            df1.to_excel(writer, sheet_name='Numeric', index=False)
            
            # Sheet 2: Text data
            df2 = pd.DataFrame({
                'Name': [f'Name_{i}' for i in range(500)],
                'Email': [f'user{i}@example.com' for i in range(500)],
                'Phone': [f'+1-555-{i:04d}' for i in range(500)],
                'Notes': [f'Long note text for user {i} ' * 10 for i in range(500)]
            })
            df2.to_excel(writer, sheet_name='Text', index=False)
            
            # Sheet 3: Date data
            df3 = pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=365, freq='D'),
                'Sales': np.random.uniform(1000, 5000, 365),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 365)
            })
            df3.to_excel(writer, sheet_name='Dates', index=False)
        
        return file_path
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self, temp_directory, complex_excel_file):
        """Test complete end-to-end processing with optimization features."""
        processor = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            max_file_size_mb=100,
            chunk_size=500,
            enable_parallel_processing=True,
            adaptive_batching=True
        )
        
        progress_updates = []
        
        async def progress_callback(current, total, status):
            progress_updates.append(f"{current}/{total}: {status}")
        
        # Process the complex file
        result = await processor.process_excel_file_async(
            complex_excel_file,
            progress_callback=progress_callback
        )
        
        # Verify results
        assert result['total_sheets'] == 3
        assert result['total_rows'] > 1000  # Should have processed all sheets
        assert len(result['all_text_chunks']) > 0
        
        # Verify all sheets were processed
        sheet_names = list(result['sheets'].keys())
        assert 'Numeric' in sheet_names
        assert 'Text' in sheet_names
        assert 'Dates' in sheet_names
        
        # Check that progress callbacks were called
        assert len(progress_updates) > 0
        
        # Verify different data types were detected
        numeric_sheet = result['sheets']['Numeric']
        text_sheet = result['sheets']['Text']
        date_sheet = result['sheets']['Dates']
        
        # Check column types
        numeric_cols = [col for col in numeric_sheet['metadata']['columns'] 
                       if col['data_type'] == 'numeric']
        text_cols = [col for col in text_sheet['metadata']['columns'] 
                    if col['data_type'] == 'text']
        date_cols = [col for col in date_sheet['metadata']['columns'] 
                    if col['data_type'] == 'datetime']
        
        assert len(numeric_cols) > 0
        assert len(text_cols) > 0
        assert len(date_cols) > 0
        
        # Verify processing info
        processing_info = result['processing_info']
        assert 'strategy' in processing_info
        assert 'parallel_processing' in processing_info
        assert processing_info['parallel_processing'] is True
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_comparison(self, temp_directory):
        """Test memory efficiency improvements."""
        # Create a moderately large file
        data = {f'Col_{i}': np.random.random(5000) for i in range(10)}
        df = pd.DataFrame(data)
        file_path = temp_directory / "memory_test.xlsx"
        df.to_excel(file_path, index=False)
        
        # Track memory usage during processing
        import psutil
        process = psutil.Process()
        
        processor = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            chunk_size=1000,
            max_cache_size_mb=50
        )
        
        initial_memory = process.memory_info().rss
        
        result = await processor.process_excel_file_async(file_path)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Verify processing completed successfully
        assert result['total_rows'] == 5000
        assert result['total_columns'] == 10
        
        # Memory increase should be reasonable (this is somewhat environment-dependent)
        print(f"Memory increase: {memory_increase:.1f} MB")
        assert memory_increase < 200  # Should not use more than 200MB for this test
        
        # Verify cache management worked
        cache_stats = processor._cache_memory_usage / 1024 / 1024
        print(f"Cache usage: {cache_stats:.1f} MB")
        assert cache_stats < processor.max_cache_size_mb
    
    def test_performance_metrics(self, temp_directory, complex_excel_file):
        """Test that performance metrics are collected."""
        processor = OptimizedExcelProcessor(
            data_directory=str(temp_directory),
            enable_parallel_processing=True
        )
        
        # Process synchronously for simpler testing
        result = processor.process_excel_file(complex_excel_file)
        
        # Check that timing information is available
        assert 'processed_at' in result
        assert 'processing_info' in result
        
        processing_info = result['processing_info']
        assert 'strategy' in processing_info
        assert 'chunk_size' in processing_info
        assert 'parallel_processing' in processing_info
        
        # Get statistics
        stats = processor.get_file_statistics()
        assert 'processing_summary' in stats
        assert 'cache_efficiency' in stats['processing_summary']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])