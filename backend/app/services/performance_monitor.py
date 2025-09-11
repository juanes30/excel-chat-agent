"""Performance monitoring and optimization for Excel processing.

This module provides comprehensive performance monitoring, benchmarking,
and optimization recommendations for the Excel Chat Agent system.
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, AsyncContextManager
import threading
from concurrent.futures import ThreadPoolExecutor

import psutil
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_start_mb: float = 0
    memory_end_mb: float = 0
    memory_peak_mb: float = 0
    cpu_usage_percent: float = 0
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def finalize(self, error: Optional[Exception] = None):
        """Finalize metrics collection."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if error:
            self.success = False
            self.error_message = str(error)
        
        # Update memory usage
        try:
            process = psutil.Process()
            self.memory_end_mb = process.memory_info().rss / 1024 / 1024
            self.cpu_usage_percent = process.cpu_percent()
        except Exception:
            pass  # Ignore psutil errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'memory_start_mb': self.memory_start_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_delta_mb': self.memory_end_mb - self.memory_start_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'success': self.success,
            'error_message': self.error_message,
            'custom_metrics': self.custom_metrics
        }


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.peak_memory_mb = 0
        self.memory_samples = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.memory_samples = []
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage."""
        self.is_monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return self.peak_memory_mb
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        try:
            process = psutil.Process()
            
            while self.is_monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_samples.append(memory_mb)
                    self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                    
                    time.sleep(self.sampling_interval)
                except Exception:
                    break  # Exit on any error
                    
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def monitor_async(self, operation_name: str, **custom_metrics) -> AsyncContextManager[PerformanceMetrics]:
        """Async context manager for monitoring operations."""
        metrics = self._start_monitoring(operation_name, custom_metrics)
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
        
        try:
            yield metrics
        except Exception as e:
            metrics.finalize(error=e)
            raise
        else:
            metrics.finalize()
        finally:
            metrics.memory_peak_mb = memory_monitor.stop_monitoring()
            self._finish_monitoring(operation_name, metrics)
    
    @contextmanager
    def monitor(self, operation_name: str, **custom_metrics):
        """Sync context manager for monitoring operations."""
        metrics = self._start_monitoring(operation_name, custom_metrics)
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
        
        try:
            yield metrics
        except Exception as e:
            metrics.finalize(error=e)
            raise
        else:
            metrics.finalize()
        finally:
            metrics.memory_peak_mb = memory_monitor.stop_monitoring()
            self._finish_monitoring(operation_name, metrics)
    
    def _start_monitoring(self, operation_name: str, custom_metrics: Dict[str, Any]) -> PerformanceMetrics:
        """Start monitoring an operation."""
        try:
            process = psutil.Process()
            memory_start = process.memory_info().rss / 1024 / 1024
        except Exception:
            memory_start = 0
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            memory_start_mb=memory_start,
            custom_metrics=custom_metrics
        )
        
        with self._lock:
            self.active_operations[operation_name] = metrics
        
        return metrics
    
    def _finish_monitoring(self, operation_name: str, metrics: PerformanceMetrics):
        """Finish monitoring an operation."""
        with self._lock:
            # Remove from active operations
            self.active_operations.pop(operation_name, None)
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Maintain history limit
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
    
    def get_statistics(self, operation_name: Optional[str] = None, 
                      time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            # Filter metrics
            filtered_metrics = self.metrics_history
            
            if operation_name:
                filtered_metrics = [m for m in filtered_metrics if m.operation_name == operation_name]
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_metrics = [m for m in filtered_metrics if m.start_time >= cutoff_time]
            
            if not filtered_metrics:
                return {"error": "No metrics found for the specified criteria"}
            
            # Calculate statistics
            durations = [m.duration_ms for m in filtered_metrics if m.duration_ms is not None]
            memory_deltas = [m.memory_end_mb - m.memory_start_mb for m in filtered_metrics]
            memory_peaks = [m.memory_peak_mb for m in filtered_metrics if m.memory_peak_mb > 0]
            cpu_usages = [m.cpu_usage_percent for m in filtered_metrics if m.cpu_usage_percent > 0]
            success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
            
            stats = {
                "operation_name": operation_name or "all_operations",
                "total_operations": len(filtered_metrics),
                "success_rate": round(success_rate, 3),
                "time_window_hours": time_window.total_seconds() / 3600 if time_window else "all_time",
                "duration_stats": self._calculate_stats(durations, "ms"),
                "memory_delta_stats": self._calculate_stats(memory_deltas, "MB"),
                "memory_peak_stats": self._calculate_stats(memory_peaks, "MB"),
                "cpu_usage_stats": self._calculate_stats(cpu_usages, "%"),
                "errors": [
                    {"operation": m.operation_name, "error": m.error_message, "time": m.start_time.isoformat()}
                    for m in filtered_metrics if not m.success
                ]
            }
            
            return stats
    
    def _calculate_stats(self, values: List[float], unit: str) -> Dict[str, Any]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {"error": f"No {unit} values available"}
        
        values_array = np.array(values)
        
        return {
            "count": len(values),
            "mean": round(float(np.mean(values_array)), 2),
            "median": round(float(np.median(values_array)), 2),
            "std": round(float(np.std(values_array)), 2),
            "min": round(float(np.min(values_array)), 2),
            "max": round(float(np.max(values_array)), 2),
            "p95": round(float(np.percentile(values_array, 95)), 2),
            "p99": round(float(np.percentile(values_array, 99)), 2),
            "unit": unit
        }
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        with self._lock:
            current_time = datetime.now()
            return [
                {
                    "operation_name": name,
                    "start_time": metrics.start_time.isoformat(),
                    "duration_ms": (current_time - metrics.start_time).total_seconds() * 1000,
                    "custom_metrics": metrics.custom_metrics
                }
                for name, metrics in self.active_operations.items()
            ]
    
    def clear_history(self):
        """Clear performance history."""
        with self._lock:
            self.metrics_history.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str = None, **custom_metrics):
    """Decorator for monitoring function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            async with performance_monitor.monitor_async(op_name, **custom_metrics) as metrics:
                # Add function-specific metrics
                metrics.custom_metrics.update({
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            with performance_monitor.monitor(op_name, **custom_metrics) as metrics:
                # Add function-specific metrics
                metrics.custom_metrics.update({
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class BenchmarkSuite:
    """Comprehensive benchmarking suite for Excel processing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    async def run_excel_processing_benchmark(self, excel_processor, test_files: List[Path]) -> Dict[str, Any]:
        """Run comprehensive Excel processing benchmark."""
        logger.info("Starting Excel processing benchmark")
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "test_files": len(test_files),
            "processor_config": {
                "max_file_size_mb": excel_processor.max_file_size_mb,
                "chunk_size": excel_processor.chunk_size,
                "parallel_processing": excel_processor.enable_parallel_processing
            },
            "file_results": [],
            "summary": {}
        }
        
        total_files = len(test_files)
        for i, file_path in enumerate(test_files):
            logger.info(f"Benchmarking file {i+1}/{total_files}: {file_path.name}")
            
            file_result = await self._benchmark_single_file(excel_processor, file_path)
            benchmark_results["file_results"].append(file_result)
        
        # Calculate summary statistics
        benchmark_results["summary"] = self._calculate_benchmark_summary(benchmark_results["file_results"])
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    async def _benchmark_single_file(self, excel_processor, file_path: Path) -> Dict[str, Any]:
        """Benchmark processing of a single file."""
        file_stats = file_path.stat()
        
        file_result = {
            "file_name": file_path.name,
            "file_size_mb": round(file_stats.st_size / 1024 / 1024, 2),
            "processing_metrics": {},
            "quality_metrics": {},
            "error": None
        }
        
        try:
            # Benchmark processing with detailed monitoring
            async with performance_monitor.monitor_async(
                f"benchmark_{file_path.name}",
                file_size_mb=file_result["file_size_mb"]
            ) as metrics:
                
                result = await excel_processor.process_excel_file_async(file_path)
                
                # Extract processing metrics
                file_result["processing_metrics"] = {
                    "duration_ms": metrics.duration_ms,
                    "memory_peak_mb": metrics.memory_peak_mb,
                    "memory_delta_mb": metrics.memory_end_mb - metrics.memory_start_mb,
                    "cpu_usage_percent": metrics.cpu_usage_percent,
                    "processing_strategy": result.get("processing_info", {}).get("strategy", "unknown"),
                    "total_sheets": result.get("total_sheets", 0),
                    "total_rows": result.get("total_rows", 0),
                    "total_columns": result.get("total_columns", 0),
                    "text_chunks_generated": len(result.get("all_text_chunks", [])),
                    "throughput_rows_per_second": (
                        result.get("total_rows", 0) / (metrics.duration_ms / 1000) 
                        if metrics.duration_ms and metrics.duration_ms > 0 else 0
                    ),
                    "throughput_mb_per_second": (
                        file_result["file_size_mb"] / (metrics.duration_ms / 1000)
                        if metrics.duration_ms and metrics.duration_ms > 0 else 0
                    )
                }
                
                # Extract quality metrics
                file_result["quality_metrics"] = self._extract_quality_metrics(result)
                
        except Exception as e:
            file_result["error"] = str(e)
            file_result["processing_metrics"]["success"] = False
            logger.error(f"Benchmark failed for {file_path.name}: {e}")
        
        return file_result
    
    def _extract_quality_metrics(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality metrics from processing result."""
        quality_metrics = {
            "data_quality_scores": [],
            "completeness_scores": [],
            "processing_issues": []
        }
        
        for sheet_name, sheet_data in processing_result.get("sheets", {}).items():
            metadata = sheet_data.get("metadata", {})
            
            # Extract data quality information
            if "data_quality" in metadata:
                dq = metadata["data_quality"]
                quality_metrics["data_quality_scores"].append({
                    "sheet": sheet_name,
                    "overall_quality": dq.get("overall_quality", "unknown"),
                    "completeness_score": dq.get("completeness_score", 0),
                    "consistency_score": dq.get("consistency_score", 0),
                    "validity_score": dq.get("validity_score", 0),
                    "issues_count": len(dq.get("issues", []))
                })
            
            # Check for processing issues
            processing_info = sheet_data.get("processing_info", {})
            if processing_info.get("was_chunked") or processing_info.get("was_streamed"):
                quality_metrics["processing_issues"].append({
                    "sheet": sheet_name,
                    "issue_type": "large_dataset_processing",
                    "details": processing_info
                })
        
        return quality_metrics
    
    def _calculate_benchmark_summary(self, file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark results."""
        successful_results = [r for r in file_results if r.get("error") is None]
        
        if not successful_results:
            return {"error": "No successful benchmark results"}
        
        # Aggregate metrics
        durations = [r["processing_metrics"]["duration_ms"] for r in successful_results]
        memory_peaks = [r["processing_metrics"]["memory_peak_mb"] for r in successful_results]
        throughput_rows = [r["processing_metrics"]["throughput_rows_per_second"] for r in successful_results]
        throughput_mb = [r["processing_metrics"]["throughput_mb_per_second"] for r in successful_results]
        
        return {
            "total_files_tested": len(file_results),
            "successful_files": len(successful_results),
            "success_rate": len(successful_results) / len(file_results),
            "duration_stats": performance_monitor._calculate_stats(durations, "ms"),
            "memory_peak_stats": performance_monitor._calculate_stats(memory_peaks, "MB"),
            "throughput_rows_stats": performance_monitor._calculate_stats(throughput_rows, "rows/sec"),
            "throughput_mb_stats": performance_monitor._calculate_stats(throughput_mb, "MB/sec"),
            "strategy_distribution": self._get_strategy_distribution(successful_results),
            "quality_summary": self._summarize_quality_metrics(successful_results)
        }
    
    def _get_strategy_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of processing strategies used."""
        strategies = {}
        for result in results:
            strategy = result["processing_metrics"].get("processing_strategy", "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1
        return strategies
    
    def _summarize_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize quality metrics across all results."""
        all_quality_scores = []
        total_issues = 0
        
        for result in results:
            quality_metrics = result.get("quality_metrics", {})
            all_quality_scores.extend(quality_metrics.get("data_quality_scores", []))
            total_issues += len(quality_metrics.get("processing_issues", []))
        
        if not all_quality_scores:
            return {"error": "No quality metrics available"}
        
        completeness_scores = [score["completeness_score"] for score in all_quality_scores]
        consistency_scores = [score["consistency_score"] for score in all_quality_scores]
        
        return {
            "total_sheets_analyzed": len(all_quality_scores),
            "completeness_stats": performance_monitor._calculate_stats(completeness_scores, "score"),
            "consistency_stats": performance_monitor._calculate_stats(consistency_scores, "score"),
            "total_processing_issues": total_issues,
            "quality_distribution": self._get_quality_distribution(all_quality_scores)
        }
    
    def _get_quality_distribution(self, quality_scores: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of overall quality ratings."""
        distribution = {}
        for score in quality_scores:
            quality = score.get("overall_quality", "unknown")
            distribution[quality] = distribution.get(quality, 0) + 1
        return distribution
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = self.output_dir / filename
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")


class OptimizationRecommendations:
    """Generate optimization recommendations based on performance data."""
    
    @staticmethod
    def analyze_performance_data(performance_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance data and generate recommendations."""
        recommendations = []
        
        # Analyze duration performance
        duration_stats = performance_stats.get("duration_stats", {})
        if duration_stats.get("mean", 0) > 30000:  # > 30 seconds average
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "issue": "High average processing time",
                "recommendation": "Consider enabling parallel processing or increasing chunk size",
                "details": f"Average duration: {duration_stats.get('mean', 0):.0f}ms"
            })
        
        # Analyze memory usage
        memory_stats = performance_stats.get("memory_peak_stats", {})
        if memory_stats.get("max", 0) > 1000:  # > 1GB peak memory
            recommendations.append({
                "category": "memory",
                "priority": "medium",
                "issue": "High memory usage detected",
                "recommendation": "Reduce chunk size or enable streaming for large files",
                "details": f"Peak memory usage: {memory_stats.get('max', 0):.0f}MB"
            })
        
        # Analyze success rate
        success_rate = performance_stats.get("success_rate", 1.0)
        if success_rate < 0.9:  # < 90% success rate
            recommendations.append({
                "category": "reliability",
                "priority": "high",
                "issue": "Low success rate",
                "recommendation": "Review error logs and implement better error handling",
                "details": f"Success rate: {success_rate:.1%}"
            })
        
        # Analyze CPU usage
        cpu_stats = performance_stats.get("cpu_usage_stats", {})
        if cpu_stats.get("mean", 0) > 80:  # > 80% average CPU
            recommendations.append({
                "category": "cpu",
                "priority": "medium",
                "issue": "High CPU usage",
                "recommendation": "Reduce parallel processing concurrency or optimize algorithms",
                "details": f"Average CPU usage: {cpu_stats.get('mean', 0):.1f}%"
            })
        
        return recommendations
    
    @staticmethod
    def generate_optimization_report(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        summary = benchmark_results.get("summary", {})
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_summary": summary,
            "recommendations": OptimizationRecommendations.analyze_performance_data(summary),
            "configuration_suggestions": OptimizationRecommendations._suggest_configuration(summary),
            "scaling_analysis": OptimizationRecommendations._analyze_scaling(benchmark_results)
        }
        
        return report
    
    @staticmethod
    def _suggest_configuration(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal configuration based on performance data."""
        suggestions = {}
        
        # Chunk size suggestions
        throughput_stats = summary.get("throughput_rows_stats", {})
        avg_throughput = throughput_stats.get("mean", 0)
        
        if avg_throughput < 1000:  # Low throughput
            suggestions["chunk_size"] = "Increase chunk size to 15000-20000 for better throughput"
        elif avg_throughput > 10000:  # High throughput
            suggestions["chunk_size"] = "Current chunk size appears optimal"
        
        # Parallel processing suggestions
        strategy_distribution = summary.get("strategy_distribution", {})
        if strategy_distribution.get("chunked", 0) > strategy_distribution.get("standard", 0):
            suggestions["parallel_processing"] = "Enable parallel processing for better performance on large files"
        
        # Memory suggestions
        memory_stats = summary.get("memory_peak_stats", {})
        if memory_stats.get("mean", 0) > 500:
            suggestions["memory_management"] = "Consider reducing cache size or implementing more aggressive cleanup"
        
        return suggestions
    
    @staticmethod
    def _analyze_scaling(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scaling characteristics."""
        file_results = benchmark_results.get("file_results", [])
        
        if len(file_results) < 3:
            return {"error": "Insufficient data for scaling analysis"}
        
        # Sort by file size
        sorted_results = sorted(file_results, key=lambda x: x.get("file_size_mb", 0))
        
        # Analyze relationship between file size and processing time
        file_sizes = [r["file_size_mb"] for r in sorted_results]
        durations = [r["processing_metrics"]["duration_ms"] for r in sorted_results 
                    if r.get("error") is None]
        
        if len(durations) >= 3:
            # Calculate correlation
            correlation = np.corrcoef(file_sizes[:len(durations)], durations)[0, 1]
            
            scaling_analysis = {
                "size_duration_correlation": round(correlation, 3),
                "scaling_characteristic": (
                    "Linear scaling" if 0.7 <= correlation <= 1.0 else
                    "Sub-linear scaling" if 0.3 <= correlation < 0.7 else
                    "Poor scaling" if correlation < 0.3 else
                    "Unknown"
                ),
                "recommendations": []
            }
            
            if correlation > 0.9:
                scaling_analysis["recommendations"].append(
                    "Processing time scales linearly with file size - consider chunking optimizations"
                )
            elif correlation < 0.3:
                scaling_analysis["recommendations"].append(
                    "Processing time doesn't scale predictably - investigate bottlenecks"
                )
            
            return scaling_analysis
        
        return {"error": "Insufficient successful results for scaling analysis"}


# Export main components
__all__ = [
    'PerformanceMetrics',
    'PerformanceMonitor',
    'BenchmarkSuite',
    'OptimizationRecommendations',
    'monitor_performance',
    'performance_monitor'
]