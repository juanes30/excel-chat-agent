"""Comprehensive example demonstrating optimized Excel processing.

This example shows how to use the OptimizedExcelProcessor and 
OptimizedVectorStoreService together for maximum performance.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our optimized services
from app.services.excel_processor import OptimizedExcelProcessor
from app.services.vector_store import OptimizedVectorStoreService
from app.services.performance_monitor import (
    performance_monitor, 
    BenchmarkSuite, 
    OptimizationRecommendations,
    monitor_performance
)


class OptimizedExcelChatSystem:
    """Example implementation of optimized Excel chat system."""
    
    def __init__(self, data_directory: str = "data/excel_files"):
        """Initialize the optimized system."""
        # Initialize optimized processor with performance settings
        self.excel_processor = OptimizedExcelProcessor(
            data_directory=data_directory,
            max_file_size_mb=500,  # Handle larger files
            chunk_size=10000,      # Optimized chunk size
            max_cache_size_mb=200, # Reasonable cache limit
            enable_parallel_processing=True
        )
        
        # Initialize optimized vector store
        self.vector_store = OptimizedVectorStoreService(
            persist_directory="chroma_db_optimized",
            collection_name="excel_documents_optimized",
            embedding_model="all-MiniLM-L6-v2",
            max_cache_size_mb=300,
            enable_parallel_processing=True,
            adaptive_batching=True
        )
        
        # Initialize benchmark suite
        self.benchmark_suite = BenchmarkSuite(
            output_dir=Path("benchmark_results")
        )
        
        logger.info("Optimized Excel Chat System initialized")
    
    @monitor_performance("process_and_index_file")
    async def process_and_index_file(self, file_path: Path, progress_callback=None) -> Dict[str, Any]:
        """Process Excel file and add to vector store with full optimization."""
        
        async def combined_progress(current, total, status):
            """Combine progress from processing and indexing."""
            if progress_callback:
                # Processing takes ~70% of time, indexing ~30%
                if "Processing" in status:
                    progress = int(current * 0.7)
                elif "batch" in status.lower():
                    progress = 70 + int((current / total) * 30)
                else:
                    progress = current
                
                await progress_callback(progress, 100, status)
        
        logger.info(f"Starting optimized processing of {file_path.name}")
        
        # Process Excel file with progress tracking
        result = await self.excel_processor.process_excel_file_async(
            file_path,
            progress_callback=combined_progress
        )
        
        # Add to vector store with progress tracking
        success = await self.vector_store.add_excel_data(
            file_name=result['file_name'],
            file_hash=result['file_hash'],
            sheets_data=result['sheets'],
            progress_callback=combined_progress
        )
        
        if not success:
            raise Exception("Failed to add data to vector store")
        
        # Return comprehensive result
        return {
            **result,
            "vector_store_success": success,
            "optimization_metrics": {
                "processor_strategy": result.get("processing_info", {}).get("strategy"),
                "memory_efficiency": result.get("processing_info", {}).get("memory_usage"),
                "parallel_processing_used": self.excel_processor.enable_parallel_processing,
                "adaptive_batching_used": self.vector_store.adaptive_batching
            }
        }
    
    @monitor_performance("intelligent_search")
    async def intelligent_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform intelligent search with performance optimization."""
        
        # Use optimized vector search
        search_results = await self.vector_store.search(
            query=query,
            n_results=max_results
        )
        
        # Enhance results with processing metadata
        enhanced_results = []
        for result in search_results:
            enhanced_result = {
                **result,
                "processing_strategy": result["metadata"].get("processing_strategy", "unknown"),
                "data_quality": result["metadata"].get("data_quality", "unknown"),
                "was_sampled": result["metadata"].get("was_sampled", False)
            }
            enhanced_results.append(enhanced_result)
        
        return {
            "query": query,
            "results": enhanced_results,
            "total_results": len(enhanced_results),
            "search_performance": {
                "vector_store_type": "optimized",
                "cache_enabled": True,
                "parallel_embedding": self.vector_store.enable_parallel_processing
            }
        }
    
    async def run_comprehensive_benchmark(self, test_files: List[Path]) -> Dict[str, Any]:
        """Run comprehensive benchmark of the optimized system."""
        logger.info("Starting comprehensive benchmark")
        
        # Benchmark Excel processing
        processing_benchmark = await self.benchmark_suite.run_excel_processing_benchmark(
            self.excel_processor, 
            test_files
        )
        
        # Test vector store performance
        vector_store_benchmark = await self._benchmark_vector_store(test_files[:3])  # Subset for vector store
        
        # Generate optimization recommendations
        optimization_report = OptimizationRecommendations.generate_optimization_report(
            processing_benchmark
        )
        
        return {
            "processing_benchmark": processing_benchmark,
            "vector_store_benchmark": vector_store_benchmark,
            "optimization_report": optimization_report,
            "system_health": await self._get_system_health()
        }
    
    async def _benchmark_vector_store(self, test_files: List[Path]) -> Dict[str, Any]:
        """Benchmark vector store operations."""
        benchmark_results = {
            "indexing_performance": [],
            "search_performance": [],
            "summary": {}
        }
        
        # Benchmark indexing
        for file_path in test_files:
            async with performance_monitor.monitor_async(f"vector_index_{file_path.name}") as metrics:
                try:
                    result = await self.excel_processor.process_excel_file_async(file_path)
                    
                    success = await self.vector_store.add_excel_data(
                        file_name=result['file_name'],
                        file_hash=result['file_hash'],
                        sheets_data=result['sheets']
                    )
                    
                    benchmark_results["indexing_performance"].append({
                        "file_name": file_path.name,
                        "success": success,
                        "duration_ms": metrics.duration_ms,
                        "memory_peak_mb": metrics.memory_peak_mb,
                        "documents_indexed": len(result.get('all_text_chunks', []))
                    })
                    
                except Exception as e:
                    logger.error(f"Indexing benchmark failed for {file_path.name}: {e}")
        
        # Benchmark search operations
        test_queries = [
            "financial data",
            "sales numbers",
            "customer information",
            "quarterly results",
            "budget analysis"
        ]
        
        for query in test_queries:
            async with performance_monitor.monitor_async(f"vector_search_{query}") as metrics:
                try:
                    results = await self.vector_store.search(query, n_results=10)
                    
                    benchmark_results["search_performance"].append({
                        "query": query,
                        "duration_ms": metrics.duration_ms,
                        "results_count": len(results),
                        "memory_delta_mb": metrics.memory_end_mb - metrics.memory_start_mb
                    })
                    
                except Exception as e:
                    logger.error(f"Search benchmark failed for query '{query}': {e}")
        
        # Calculate summary
        if benchmark_results["indexing_performance"]:
            indexing_durations = [r["duration_ms"] for r in benchmark_results["indexing_performance"] if r["success"]]
            search_durations = [r["duration_ms"] for r in benchmark_results["search_performance"]]
            
            benchmark_results["summary"] = {
                "indexing_stats": self._calculate_simple_stats(indexing_durations, "ms"),
                "search_stats": self._calculate_simple_stats(search_durations, "ms"),
                "indexing_success_rate": sum(1 for r in benchmark_results["indexing_performance"] if r["success"]) / len(benchmark_results["indexing_performance"]),
                "average_documents_per_file": np.mean([r["documents_indexed"] for r in benchmark_results["indexing_performance"] if r["success"]])
            }
        
        return benchmark_results
    
    def _calculate_simple_stats(self, values: List[float], unit: str) -> Dict[str, Any]:
        """Calculate simple statistics."""
        if not values:
            return {"error": f"No {unit} values"}
        
        return {
            "mean": round(np.mean(values), 2),
            "median": round(np.median(values), 2),
            "min": round(np.min(values), 2),
            "max": round(np.max(values), 2),
            "unit": unit
        }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            "excel_processor": {
                "cache_usage_mb": round(self.excel_processor._cache_memory_usage / 1024 / 1024, 2),
                "cache_size": len(self.excel_processor._file_cache),
                "active_operations": len(self.excel_processor._active_operations),
                "parallel_processing": self.excel_processor.enable_parallel_processing
            },
            "vector_store": self.vector_store.health_check(),
            "performance_monitor": {
                "active_operations": len(performance_monitor.get_active_operations()),
                "metrics_history_size": len(performance_monitor.metrics_history)
            }
        }
    
    async def cleanup(self):
        """Cleanup system resources."""
        logger.info("Cleaning up optimized system")
        
        self.excel_processor.cleanup()
        self.vector_store.cleanup()
        performance_monitor.clear_history()
        
        logger.info("Cleanup completed")


async def create_test_data(output_dir: Path) -> List[Path]:
    """Create test Excel files for demonstration."""
    output_dir.mkdir(exist_ok=True)
    test_files = []
    
    # Small file - standard processing
    small_data = {
        'Product': ['Widget A', 'Widget B', 'Widget C'] * 10,
        'Sales': np.random.uniform(1000, 5000, 30),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 30),
        'Date': pd.date_range('2023-01-01', periods=30, freq='D')
    }
    small_file = output_dir / "small_sales_data.xlsx"
    pd.DataFrame(small_data).to_excel(small_file, index=False)
    test_files.append(small_file)
    
    # Medium file - optimized processing
    medium_data = {
        'CustomerID': range(5000),
        'Name': [f'Customer_{i}' for i in range(5000)],
        'Email': [f'customer{i}@company.com' for i in range(5000)],
        'Revenue': np.random.uniform(100, 10000, 5000),
        'Segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], 5000),
        'SignupDate': pd.date_range('2020-01-01', periods=5000, freq='3H')
    }
    medium_file = output_dir / "customer_database.xlsx"
    pd.DataFrame(medium_data).to_excel(medium_file, index=False)
    test_files.append(medium_file)
    
    # Large file - chunked processing
    large_file = output_dir / "large_analytics_data.xlsx"
    with pd.ExcelWriter(large_file, engine='openpyxl') as writer:
        # Multiple sheets with different data types
        
        # Numeric analysis sheet
        numeric_data = {
            'TransactionID': range(10000),
            'Amount': np.random.uniform(10, 1000, 10000),
            'Tax': np.random.uniform(1, 100, 10000),
            'Discount': np.random.uniform(0, 50, 10000),
            'Profit': np.random.uniform(-50, 200, 10000)
        }
        pd.DataFrame(numeric_data).to_excel(writer, sheet_name='Financial', index=False)
        
        # Text analysis sheet
        text_data = {
            'ReviewID': range(3000),
            'ProductName': [f'Product {np.random.randint(1, 100)}' for _ in range(3000)],
            'Review': [f'This is a review text for product {i}. ' * 5 for i in range(3000)],
            'Rating': np.random.randint(1, 6, 3000),
            'Verified': np.random.choice([True, False], 3000)
        }
        pd.DataFrame(text_data).to_excel(writer, sheet_name='Reviews', index=False)
        
        # Time series sheet
        time_data = {
            'Timestamp': pd.date_range('2022-01-01', periods=8760, freq='H'),  # One year hourly
            'Value': np.random.uniform(0, 100, 8760) + 10 * np.sin(np.arange(8760) * 2 * np.pi / 24),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 8760)
        }
        pd.DataFrame(time_data).to_excel(writer, sheet_name='TimeSeries', index=False)
    
    test_files.append(large_file)
    
    logger.info(f"Created {len(test_files)} test files in {output_dir}")
    return test_files


async def main():
    """Main demonstration function."""
    logger.info("Starting Optimized Excel Processing Demonstration")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        test_files = await create_test_data(temp_path)
        
        # Initialize optimized system
        system = OptimizedExcelChatSystem(data_directory=str(temp_path))
        
        try:
            # Demonstrate individual file processing
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATION 1: Individual File Processing")
            logger.info("="*50)
            
            progress_updates = []
            
            async def progress_callback(current, total, status):
                progress_updates.append(f"{current}/{total}: {status}")
                if current % 20 == 0 or current == total:  # Log every 20% or at completion
                    logger.info(f"Progress: {current}/{total} - {status}")
            
            # Process each test file
            for i, file_path in enumerate(test_files):
                logger.info(f"\nProcessing file {i+1}/{len(test_files)}: {file_path.name}")
                
                result = await system.process_and_index_file(file_path, progress_callback)
                
                logger.info(f"Processing completed:")
                logger.info(f"  - Sheets: {result['total_sheets']}")
                logger.info(f"  - Rows: {result['total_rows']}")
                logger.info(f"  - Text chunks: {len(result['all_text_chunks'])}")
                logger.info(f"  - Strategy: {result['optimization_metrics']['processor_strategy']}")
                logger.info(f"  - Vector store success: {result['vector_store_success']}")
            
            # Demonstrate intelligent search
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATION 2: Intelligent Search")
            logger.info("="*50)
            
            test_queries = [
                "financial data and revenue",
                "customer reviews and ratings",
                "sales performance by region",
                "time series analytics"
            ]
            
            for query in test_queries:
                logger.info(f"\nSearching for: '{query}'")
                
                search_results = await system.intelligent_search(query, max_results=5)
                
                logger.info(f"Found {search_results['total_results']} results:")
                for j, result in enumerate(search_results['results'][:3]):  # Show top 3
                    logger.info(f"  {j+1}. {result['file_name']} - {result['sheet_name']} "
                               f"(relevance: {result['relevance_score']:.3f})")
            
            # Demonstrate comprehensive benchmarking
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATION 3: Comprehensive Benchmarking")
            logger.info("="*50)
            
            benchmark_results = await system.run_comprehensive_benchmark(test_files)
            
            # Display benchmark summary
            processing_summary = benchmark_results["processing_benchmark"]["summary"]
            logger.info("\nProcessing Benchmark Results:")
            logger.info(f"  - Success rate: {processing_summary['success_rate']:.1%}")
            logger.info(f"  - Average duration: {processing_summary['duration_stats']['mean']:.0f}ms")
            logger.info(f"  - Average throughput: {processing_summary.get('throughput_rows_stats', {}).get('mean', 0):.0f} rows/sec")
            
            vector_summary = benchmark_results["vector_store_benchmark"]["summary"]
            logger.info("\nVector Store Benchmark Results:")
            logger.info(f"  - Indexing success rate: {vector_summary.get('indexing_success_rate', 0):.1%}")
            logger.info(f"  - Average search time: {vector_summary.get('search_stats', {}).get('mean', 0):.0f}ms")
            
            # Display optimization recommendations
            recommendations = benchmark_results["optimization_report"]["recommendations"]
            if recommendations:
                logger.info("\nOptimization Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    logger.info(f"  - {rec['category'].upper()}: {rec['recommendation']}")
            else:
                logger.info("\nNo optimization recommendations - system performing well!")
            
            # Display system health
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATION 4: System Health Monitoring")
            logger.info("="*50)
            
            health = await system._get_system_health()
            
            logger.info("Excel Processor Health:")
            processor_health = health["excel_processor"]
            logger.info(f"  - Cache usage: {processor_health['cache_usage_mb']:.1f}MB")
            logger.info(f"  - Active operations: {processor_health['active_operations']}")
            logger.info(f"  - Parallel processing: {processor_health['parallel_processing']}")
            
            logger.info("Vector Store Health:")
            vector_health = health["vector_store"]
            logger.info(f"  - Status: {vector_health['status']}")
            logger.info(f"  - Document count: {vector_health['document_count']}")
            logger.info(f"  - Cache utilization: {vector_health.get('detailed_status', {}).get('cache_utilization', 0):.1%}")
            
            # Performance monitoring statistics
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATION 5: Performance Monitoring")
            logger.info("="*50)
            
            # Get performance statistics
            perf_stats = performance_monitor.get_statistics()
            
            logger.info("Overall Performance Statistics:")
            logger.info(f"  - Total operations: {perf_stats['total_operations']}")
            logger.info(f"  - Success rate: {perf_stats['success_rate']:.1%}")
            
            if 'duration_stats' in perf_stats:
                duration_stats = perf_stats['duration_stats']
                logger.info(f"  - Average duration: {duration_stats['mean']:.0f}ms")
                logger.info(f"  - 95th percentile: {duration_stats['p95']:.0f}ms")
            
            if 'memory_peak_stats' in perf_stats:
                memory_stats = perf_stats['memory_peak_stats']
                logger.info(f"  - Average peak memory: {memory_stats['mean']:.1f}MB")
                logger.info(f"  - Max peak memory: {memory_stats['max']:.1f}MB")
            
        finally:
            # Cleanup
            await system.cleanup()
            logger.info("\nDemonstration completed successfully!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())