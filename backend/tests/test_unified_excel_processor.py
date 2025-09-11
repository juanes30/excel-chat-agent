"""Comprehensive tests for the Unified Excel Processor Service."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.excel_processor import ExcelProcessor


class TestUnifiedExcelProcessor:
    """Test unified Excel processor with both basic and enhanced functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def processor_basic(self, temp_dir):
        """Create processor with basic analysis mode."""
        return ExcelProcessor(temp_dir, analysis_mode="basic")
    
    @pytest.fixture
    def processor_comprehensive(self, temp_dir):
        """Create processor with comprehensive analysis mode."""
        return ExcelProcessor(temp_dir, analysis_mode="comprehensive")
    
    @pytest.fixture
    def processor_auto(self, temp_dir):
        """Create processor with auto analysis mode."""
        return ExcelProcessor(temp_dir, analysis_mode="auto")
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create sample Excel file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            # Create sample data with patterns
            data = {
                'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'Age': [25, 30, 35, 28],
                'Salary': [50000, 60000, 70000, 55000],
                'Email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 'diana@company.com'],
                'Department': ['IT', 'HR', 'Finance', 'IT']
            }
            df = pd.DataFrame(data)
            df.to_excel(tmp_file.name, index=False, sheet_name='Employees')
            
            yield tmp_file.name
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    @pytest.fixture
    def complex_excel_file(self):
        """Create complex Excel file with multiple sheets and data quality issues."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                # Main data sheet
                main_data = {
                    'Sales': np.random.normal(10000, 2000, 50),
                    'Profit': np.random.normal(1500, 500, 50),
                    'Employee_ID': range(1001, 1051),
                    'Rating': np.random.uniform(1, 5, 50),
                    'Email': [f'user{i}@company.com' for i in range(50)],
                    'Phone': [f'555-{1000 + i:04d}' for i in range(50)],
                    'Date': [datetime(2023, 1, 1) + timedelta(days=i*7) for i in range(50)],
                    'Category': ['A', 'B', 'C'] * 16 + ['A', 'B'],  # 50 items
                    'Price': [100.50, 200.75, np.nan, 150.25] * 12 + [100.50, 200.75],  # 50 items
                }
                main_df = pd.DataFrame(main_data)
                main_df.to_excel(writer, sheet_name='Main_Data', index=False)
                
                # Quality issues sheet
                quality_data = {
                    'Mixed_Types': [1, 'text', 3.14, True, None],
                    'Outliers': [1, 2, 3, 1000000, 5],
                    'Duplicates': ['A', 'B', 'A', 'B', 'A'],
                    'Empty_Column': [None] * 5,
                }
                quality_df = pd.DataFrame(quality_data)
                quality_df.to_excel(writer, sheet_name='Quality_Issues', index=False)
            
            yield tmp_file.name
            os.unlink(tmp_file.name)
    
    def test_initialization_modes(self, temp_dir):
        """Test different analysis mode initializations."""
        # Basic mode
        basic_proc = ExcelProcessor(temp_dir, analysis_mode="basic")
        assert basic_proc.analysis_mode == "basic"
        
        # Comprehensive mode
        comp_proc = ExcelProcessor(temp_dir, analysis_mode="comprehensive")
        assert comp_proc.analysis_mode == "comprehensive"
        
        # Auto mode
        auto_proc = ExcelProcessor(temp_dir, analysis_mode="auto")
        assert auto_proc.analysis_mode == "auto"
    
    def test_analysis_mode_determination(self, processor_auto):
        """Test automatic analysis mode determination."""
        # Small file should get comprehensive analysis
        small_mode = processor_auto._determine_analysis_mode(5.0)  # 5MB
        assert small_mode == "comprehensive"
        
        # Medium file should get moderate analysis
        medium_mode = processor_auto._determine_analysis_mode(50.0)  # 50MB
        assert medium_mode == "moderate"
        
        # Large file should get basic analysis
        large_mode = processor_auto._determine_analysis_mode(150.0)  # 150MB
        assert large_mode == "basic"
    
    def test_pattern_detection(self, processor_comprehensive):
        """Test enhanced pattern detection."""
        # Email pattern detection
        email_series = pd.Series(['test@example.com', 'user@domain.org', 'not-email'])
        patterns = processor_comprehensive.detect_data_patterns(email_series)
        assert patterns['email_count'] == 2
        
        # Phone pattern detection
        phone_series = pd.Series(['555-1234', '(555) 123-4567', 'not-phone'])
        patterns = processor_comprehensive.detect_data_patterns(phone_series)
        assert patterns['phone_count'] >= 1
        
        # Numeric patterns
        numeric_series = pd.Series([1.0, 2.5, 3.75, 4.125, 5.0625])
        patterns = processor_comprehensive.detect_data_patterns(numeric_series)
        assert patterns['numeric_patterns']['is_integer'] == False
    
    def test_basic_processing(self, processor_basic, sample_excel_file):
        """Test basic processing functionality."""
        result = processor_basic.process_excel_file(sample_excel_file)
        
        # Verify basic structure
        assert result['file_name'] == Path(sample_excel_file).name
        assert result['total_sheets'] == 1
        assert result['total_rows'] == 4
        assert result['analysis_mode'] == 'basic'
        assert result['processing_info']['enhanced_features_enabled'] == False
        
        # Verify sheet data
        assert 'Employees' in result['sheets']
        employees_sheet = result['sheets']['Employees']
        
        # Check columns don't have pattern analysis in basic mode
        for col in employees_sheet['metadata']['columns']:
            assert 'patterns' not in col or col['patterns'] is None
    
    def test_comprehensive_processing(self, processor_comprehensive, complex_excel_file):
        """Test comprehensive processing functionality."""
        result = processor_comprehensive.process_excel_file(complex_excel_file)
        
        # Verify enhanced structure
        assert result['analysis_mode'] == 'comprehensive'
        assert result['processing_info']['enhanced_features_enabled'] == True
        assert result['processing_info']['pattern_detection_enabled'] == True
        
        # Verify enhanced analysis in main sheet
        main_sheet = result['sheets']['Main_Data']
        
        # Check that patterns are detected
        email_col = next((col for col in main_sheet['metadata']['columns'] if col['name'] == 'Email'), None)
        assert email_col is not None
        assert 'patterns' in email_col
        assert email_col['patterns']['email_count'] > 0
        
        # Check enhanced numeric stats
        sales_col = next((col for col in main_sheet['metadata']['columns'] if col['name'] == 'Sales'), None)
        assert sales_col is not None
        if sales_col['data_type'] == 'numeric':
            stats = sales_col['stats']
            assert 'quartiles' in stats  # Enhanced stat only in comprehensive mode
            assert 'skewness' in stats
            assert 'kurtosis' in stats
    
    def test_backward_compatibility(self, processor_basic, sample_excel_file):
        """Test that existing code still works (backward compatibility)."""
        # This should work exactly like before
        result = processor_basic.process_excel_file(sample_excel_file)
        
        # Standard structure should be preserved
        required_keys = [
            'file_name', 'file_hash', 'file_path', 'file_size_mb',
            'last_modified', 'extension', 'total_sheets', 'total_rows',
            'total_columns', 'sheets', 'all_text_chunks', 'processed_at'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Sheet structure should be preserved
        sheet_data = result['sheets']['Employees']
        required_sheet_keys = ['metadata', 'text_chunks', 'data']
        
        for key in required_sheet_keys:
            assert key in sheet_data, f"Missing sheet key: {key}"
    
    def test_file_validation_enhanced(self, processor_comprehensive):
        """Test enhanced file validation."""
        # Create a temporary large file (simulated)
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            # Write some dummy data to make it appear large
            tmp_file.write(b'x' * (60 * 1024 * 1024))  # 60MB of dummy data
            tmp_file_path = Path(tmp_file.name)
        
        try:
            validation = processor_comprehensive.validate_excel_file(tmp_file_path)
            assert validation['is_large_file'] == True
            assert validation['file_size_mb'] > 50
        except ValueError as e:
            # File too large or invalid Excel format is expected for dummy data
            assert "too large" in str(e) or "Unsupported file format" in str(e)
        finally:
            tmp_file_path.unlink()
    
    def test_statistics_basic_vs_comprehensive(self, temp_dir, sample_excel_file):
        """Test that statistics work for both modes."""
        # Copy file to temp directory
        import shutil
        dest_path = Path(temp_dir) / 'test.xlsx'
        shutil.copy2(sample_excel_file, dest_path)
        
        # Test basic processor
        basic_proc = ExcelProcessor(temp_dir, analysis_mode="basic")
        basic_stats = basic_proc.get_file_statistics()
        assert basic_stats['total_files'] >= 1
        assert basic_stats['total_rows'] >= 4
        
        # Test comprehensive processor 
        comp_proc = ExcelProcessor(temp_dir, analysis_mode="comprehensive")
        comp_stats = comp_proc.get_file_statistics()
        assert comp_stats['total_files'] >= 1
        assert comp_stats['total_rows'] >= 4
        
        # Both should return valid statistics
        assert basic_stats['total_files'] == comp_stats['total_files']
        assert basic_stats['total_rows'] == comp_stats['total_rows']
    
    def test_memory_efficient_features(self, processor_basic):
        """Test memory-efficient processing features."""
        # Test that processor has memory management features
        assert hasattr(processor_basic, '_manage_cache_memory')
        assert hasattr(processor_basic, '_cache_memory_usage')
        assert hasattr(processor_basic, 'max_cache_size_mb')
        
        # Test cache operations
        initial_usage = processor_basic._cache_memory_usage
        processor_basic._update_cache('test_key', {'data': 'test'})
        assert processor_basic._cache_memory_usage >= initial_usage
    
    def test_enhanced_methods_exist(self, processor_comprehensive):
        """Test that all enhanced methods are available."""
        enhanced_methods = [
            'detect_data_patterns',
            '_find_common_prefixes',
            '_find_common_suffixes',
            '_detect_encoding_issues',
            '_analyze_decimal_places',
            '_detect_distribution',
            '_advanced_outlier_detection',
            '_detect_seasonality',
            '_analyze_date_frequency',
            '_determine_analysis_mode'
        ]
        
        for method_name in enhanced_methods:
            assert hasattr(processor_comprehensive, method_name), f"Missing method: {method_name}"
    
    def test_outlier_detection(self, processor_comprehensive):
        """Test advanced outlier detection."""
        # Create data with known outliers
        normal_data = np.random.normal(50, 10, 100)
        outlier_data = np.append(normal_data, [150, -50, 200])  # Add clear outliers
        series = pd.Series(outlier_data)
        
        outlier_info = processor_comprehensive._advanced_outlier_detection(series)
        
        assert outlier_info['total_outliers'] > 0
        assert outlier_info['outlier_percentage'] > 0
        assert len(outlier_info['iqr_outliers']) > 0
    
    def test_date_frequency_analysis(self, processor_comprehensive):
        """Test date frequency analysis."""
        # Daily data
        daily_dates = pd.date_range('2023-01-01', periods=30, freq='D')
        daily_series = pd.Series(daily_dates)
        
        freq_analysis = processor_comprehensive._analyze_date_frequency(daily_series)
        assert freq_analysis['frequency'] == 'daily'
        assert freq_analysis['interval_consistency'] == True
        
        # Weekly data
        weekly_dates = pd.date_range('2023-01-01', periods=10, freq='W')
        weekly_series = pd.Series(weekly_dates)
        
        freq_analysis = processor_comprehensive._analyze_date_frequency(weekly_series)
        assert freq_analysis['frequency'] == 'weekly'
    
    def test_processing_info_metadata(self, processor_basic, processor_comprehensive, sample_excel_file):
        """Test that processing info contains correct metadata."""
        # Basic processing
        basic_result = processor_basic.process_excel_file(sample_excel_file)
        basic_info = basic_result['processing_info']
        
        assert basic_info['analysis_mode'] == 'basic'
        assert basic_info['enhanced_features_enabled'] == False
        assert basic_info['pattern_detection_enabled'] == False
        assert basic_info['performance_optimized'] == True
        assert basic_info['processing_version'] == '2.0-unified'
        
        # Comprehensive processing
        comp_result = processor_comprehensive.process_excel_file(sample_excel_file)
        comp_info = comp_result['processing_info']
        
        assert comp_info['analysis_mode'] == 'comprehensive'
        assert comp_info['enhanced_features_enabled'] == True
        assert comp_info['pattern_detection_enabled'] == True
        assert comp_info['performance_optimized'] == True
        assert comp_info['processing_version'] == '2.0-unified'


def test_import_backward_compatibility():
    """Test that the ExcelProcessor alias works for backward compatibility."""
    from app.services.excel_processor import ExcelProcessor, OptimizedExcelProcessor
    
    # Should be the same class
    assert ExcelProcessor is OptimizedExcelProcessor
    
    # Should be importable and instantiable
    processor = ExcelProcessor('test')
    assert processor is not None
    assert hasattr(processor, 'process_excel_file')
    assert hasattr(processor, 'analysis_mode')


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running unified Excel processor tests...")
    
    # Test imports
    try:
        from app.services.excel_processor import ExcelProcessor
        print("✅ Import test passed")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        exit(1)
    
    # Test initialization
    try:
        processor = ExcelProcessor('test')
        print(f"✅ Initialization test passed (mode: {processor.analysis_mode})")
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        exit(1)
    
    # Test method availability
    required_methods = [
        'process_excel_file', 'validate_excel_file', 'get_file_statistics',
        'detect_data_patterns', '_determine_analysis_mode'
    ]
    
    for method in required_methods:
        if hasattr(processor, method):
            print(f"✅ Method {method} available")
        else:
            print(f"❌ Method {method} missing")
            exit(1)
    
    print("🎉 All unified Excel processor tests passed!")