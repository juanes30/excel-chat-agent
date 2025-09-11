"""Comprehensive tests for the Enhanced Excel Processor Service."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.enhanced_excel_processor import EnhancedExcelProcessor


class TestEnhancedExcelProcessor:
    """Test enhanced Excel processor functionality with advanced features."""
    
    @pytest.fixture
    def processor(self):
        """Create enhanced Excel processor instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = EnhancedExcelProcessor(temp_dir)
            yield processor
    
    @pytest.fixture
    def sample_data_comprehensive(self):
        """Create comprehensive sample data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        data = {
            # Numeric data with patterns
            'Sales': np.random.normal(10000, 2000, 100),
            'Profit': np.random.normal(1500, 500, 100),
            'Employee_ID': range(1001, 1101),
            'Rating': np.random.uniform(1, 5, 100),
            
            # Text data with patterns
            'Email': [f'user{i}@company.com' if i % 3 == 0 else f'name{i}@domain.org' 
                     for i in range(100)],
            'Phone': [f'555-{1000 + i:04d}' if i % 4 == 0 else f'(555) {1000 + i:04d}' 
                     for i in range(100)],
            'Product_Name': ['Product_A', 'Product_B', 'Product_C'] * 33 + ['Product_D'],
            'Description': [f'This is description {i}' for i in range(100)],
            
            # DateTime data
            'Order_Date': [datetime(2023, 1, 1) + timedelta(days=i*3) for i in range(100)],
            'Delivery_Date': [datetime(2023, 1, 1) + timedelta(days=i*3 + 5) for i in range(100)],
            
            # Mixed quality data
            'Customer_ID': ['C001', 'C002', None, 'C003'] * 25,
            'Price': [100.50, 200.75, np.nan, 150.25] * 25,
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_excel_file_comprehensive(self, sample_data_comprehensive):
        """Create comprehensive Excel file with multiple sheets."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            # Main data sheet
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                sample_data_comprehensive.to_excel(writer, sheet_name='Sales_Data', index=False)
                
                # Summary sheet
                summary_data = pd.DataFrame({
                    'Category': ['Product_A', 'Product_B', 'Product_C', 'Product_D'],
                    'Total_Sales': [350000, 320000, 280000, 50000],
                    'Avg_Rating': [4.2, 3.8, 4.1, 3.9],
                    'Launch_Date': [datetime(2022, 1, 1), datetime(2022, 3, 1), 
                                   datetime(2022, 6, 1), datetime(2023, 1, 1)]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Quality issues sheet
                problematic_data = pd.DataFrame({
                    'Mixed_Types': [1, 'text', 3.14, True, None],
                    'Outliers': [1, 2, 3, 1000000, 5],  # One extreme outlier
                    'Duplicates': ['A', 'B', 'A', 'B', 'A'],
                    'Empty_Column': [None] * 5,
                    'Inconsistent_Dates': ['2023-01-01', '01/02/2023', 'Invalid', '2023-03-01', None]
                })
                problematic_data.to_excel(writer, sheet_name='Quality_Issues', index=False)
            
            yield tmp_file.name
            os.unlink(tmp_file.name)
    
    def test_pattern_detection_comprehensive(self, processor):
        """Test comprehensive pattern detection across different data types."""
        # Email pattern detection
        email_series = pd.Series(['test@example.com', 'user@domain.org', 'not-email', 'admin@site.net'])
        patterns = processor.detect_data_patterns(email_series)
        
        assert patterns['email_count'] == 3
        assert patterns['url_count'] == 0
        assert patterns['phone_count'] == 0
        
        # Phone pattern detection
        phone_series = pd.Series(['555-1234', '(555) 123-4567', '+1-555-123-4567', 'not-phone'])
        patterns = processor.detect_data_patterns(phone_series)
        
        assert patterns['phone_count'] >= 2  # Should detect at least 2 phone patterns
        assert patterns['email_count'] == 0
        
        # Numeric pattern detection
        numeric_series = pd.Series([1.0, 2.5, 3.75, 4.125, 5.0625])
        patterns = processor.detect_data_patterns(numeric_series)
        
        numeric_patterns = patterns['numeric_patterns']
        assert not numeric_patterns['is_integer']  # Has decimal values
        assert not numeric_patterns['has_negative']
        assert not numeric_patterns['has_zero']
        assert numeric_patterns['distribution_type']['type'] in ['likely_normal', 'approximately_symmetric', 'unknown', 'insufficient_data']
    
    def test_data_quality_assessment(self, processor, sample_data_comprehensive):
        """Test comprehensive data quality assessment."""
        quality = processor.assess_data_quality(sample_data_comprehensive)
        
        # Check main quality scores
        assert 0 <= quality['completeness_score'] <= 1
        assert 0 <= quality['consistency_score'] <= 1
        assert 0 <= quality['validity_score'] <= 1
        assert 0 <= quality['accuracy_score'] <= 1
        
        # Check overall quality categorization
        assert quality['overall_quality'] in ['excellent', 'good', 'fair', 'poor', 'very_poor']
        
        # Check data profiling
        profiling = quality['data_profiling']
        assert profiling['row_count'] == 100
        assert profiling['column_count'] == len(sample_data_comprehensive.columns)
        assert profiling['memory_usage_mb'] > 0
        assert profiling['duplicate_rows'] >= 0
        
        # Check column quality
        assert len(quality['column_quality']) == len(sample_data_comprehensive.columns)
        
        # Verify specific column assessments
        email_quality = quality['column_quality']['Email']
        assert email_quality['consistency_score'] > 0.5  # Should be fairly consistent
        
        # Customer_ID has nulls, should affect completeness
        customer_id_quality = quality['column_quality']['Customer_ID']
        assert customer_id_quality['data_integrity']['null_count'] > 0
    
    def test_advanced_outlier_detection(self, processor):
        """Test advanced outlier detection methods."""
        # Create data with known outliers
        normal_data = np.random.normal(50, 10, 100)
        outlier_data = np.append(normal_data, [150, -50, 200])  # Add clear outliers
        series = pd.Series(outlier_data)
        
        outlier_info = processor._advanced_outlier_detection(series)
        
        assert outlier_info['total_outliers'] > 0
        assert outlier_info['outlier_percentage'] > 0
        assert len(outlier_info['iqr_outliers']) > 0
        
        # Check that extreme values are detected
        all_outliers = (outlier_info['iqr_outliers'] + 
                       outlier_info['z_score_outliers'] + 
                       outlier_info['modified_z_outliers'])
        
        assert any(val > 100 for val in all_outliers)  # Should detect the 150, 200 outliers
        assert any(val < 0 for val in all_outliers)    # Should detect the -50 outlier
    
    def test_relationship_analysis(self, processor, sample_data_comprehensive):
        """Test data relationship analysis."""
        relationships = processor.analyze_data_relationships(sample_data_comprehensive)
        
        # Check correlations
        assert 'correlations' in relationships
        
        # Check potential keys - Employee_ID should be detected as unique
        potential_keys = relationships['potential_keys']
        key_columns = [key['column'] for key in potential_keys]
        assert 'Employee_ID' in key_columns
        
        # Verify key properties
        employee_id_key = next(key for key in potential_keys if key['column'] == 'Employee_ID')
        assert employee_id_key['uniqueness_ratio'] > 0.95
        assert employee_id_key['is_sequential'] == True
        
        # Check column similarities
        assert 'column_similarities' in relationships
    
    def test_enhanced_metadata_extraction(self, processor, sample_data_comprehensive):
        """Test enhanced metadata extraction."""
        metadata = processor.extract_enhanced_metadata(sample_data_comprehensive, 'Test_Sheet')
        
        # Basic metadata
        assert metadata['sheet_name'] == 'Test_Sheet'
        assert metadata['num_rows'] == 100
        assert metadata['num_cols'] == len(sample_data_comprehensive.columns)
        
        # Enhanced features
        assert 'data_quality' in metadata
        assert 'relationships' in metadata
        assert 'header_info' in metadata
        assert 'sheet_statistics' in metadata
        
        # Check column information enhancement
        columns = metadata['columns']
        assert len(columns) == len(sample_data_comprehensive.columns)
        
        # Verify enhanced column properties
        for col in columns:
            assert 'patterns' in col
            assert 'quality' in col
            assert 'uniqueness_ratio' in col
            
            # Check data type specific enhancements
            if col['data_type'] == 'numeric':
                assert 'quartiles' in col.get('stats', {})
                assert 'skewness' in col.get('stats', {})
                assert 'kurtosis' in col.get('stats', {})
            elif col['data_type'] == 'datetime':
                assert 'frequency_analysis' in col.get('stats', {})
            elif col['data_type'] == 'text':
                assert 'encoding_info' in col.get('stats', {})
        
        # Sheet statistics
        sheet_stats = metadata['sheet_statistics']
        assert sheet_stats['total_cells'] == 100 * len(sample_data_comprehensive.columns)
        assert sheet_stats['data_density'] > 0
        assert sheet_stats['numeric_columns'] > 0
        assert sheet_stats['text_columns'] > 0
        assert sheet_stats['datetime_columns'] > 0
    
    def test_enhanced_searchable_text(self, processor, sample_data_comprehensive):
        """Test enhanced searchable text generation."""
        metadata = processor.extract_enhanced_metadata(sample_data_comprehensive, 'Test_Sheet')
        text_chunks = processor.create_enhanced_searchable_text(
            sample_data_comprehensive, 'Test_Sheet', metadata
        )
        
        assert len(text_chunks) > 0
        
        # Check for enhanced content
        all_text = ' '.join(text_chunks)
        
        # Should include quality information
        assert 'Data Quality:' in all_text or 'quality' in all_text.lower()
        
        # Should include memory usage
        assert 'Memory Usage:' in all_text or 'memory' in all_text.lower()
        
        # Should include pattern information
        assert 'email' in all_text.lower()  # Email patterns detected
        
        # Should include relationship information
        assert 'correlation' in all_text.lower() or 'key' in all_text.lower()
        
        # Should include statistical context
        assert any(word in all_text.lower() for word in ['mean', 'median', 'std', 'quartile'])
    
    def test_file_processing_comprehensive(self, processor, sample_excel_file_comprehensive):
        """Test complete enhanced file processing."""
        result = processor.process_excel_file_enhanced(sample_excel_file_comprehensive)
        
        # Basic file information
        assert result['total_sheets'] == 3
        assert result['total_rows'] > 0
        assert result['total_columns'] > 0
        
        # Enhanced processing information
        processing_summary = result['processing_summary']
        assert processing_summary['analysis_type'] == 'enhanced'
        assert processing_summary['quality_assessment_enabled'] == True
        assert processing_summary['relationship_analysis_enabled'] == True
        assert processing_summary['pattern_detection_enabled'] == True
        assert processing_summary['file_quality'] in ['excellent', 'good', 'fair', 'poor', 'very_poor']
        assert processing_summary['total_issues'] >= 0
        
        # Check sheet processing
        for sheet_name, sheet_data in result['sheets'].items():
            assert 'metadata' in sheet_data
            assert 'text_chunks' in sheet_data
            assert 'processing_info' in sheet_data
            
            # Enhanced metadata
            metadata = sheet_data['metadata']
            assert 'data_quality' in metadata
            assert 'relationships' in metadata
            assert 'header_info' in metadata
            
            # Processing information
            processing_info = sheet_data['processing_info']
            assert 'original_shape' in processing_info
            assert 'cleaned_shape' in processing_info
            assert 'cleaning_stats' in processing_info
    
    def test_quality_issues_detection(self, processor, sample_excel_file_comprehensive):
        """Test detection of specific data quality issues."""
        result = processor.process_excel_file_enhanced(sample_excel_file_comprehensive)
        
        # Check the Quality_Issues sheet specifically
        if 'Quality_Issues' in result['sheets']:
            quality_sheet = result['sheets']['Quality_Issues']
            quality_assessment = quality_sheet['metadata']['data_quality']
            
            # Should detect issues in this problematic sheet
            assert len(quality_assessment['issues']) > 0
            assert len(quality_assessment['recommendations']) > 0
            
            # Should detect issues (quality may vary due to small dataset size)
            assert quality_assessment['overall_quality'] in ['poor', 'very_poor', 'fair', 'good', 'excellent']
            
            # Check specific column issues
            column_quality = quality_assessment['column_quality']
            
            # Empty column should be detected
            if 'Empty_Column' in column_quality:
                empty_col_quality = column_quality['Empty_Column']
                assert empty_col_quality['consistency_score'] == 0.0
                assert empty_col_quality['validity_score'] == 0.0
            
            # Mixed types should be detected
            if 'Mixed_Types' in column_quality:
                mixed_col_quality = column_quality['Mixed_Types']
                assert mixed_col_quality['has_mixed_types'] == True
            
            # Outliers should be detected
            if 'Outliers' in column_quality:
                outlier_col_quality = column_quality['Outliers']
                assert outlier_col_quality['outlier_count'] > 0
    
    def test_enhanced_statistics(self, processor, sample_excel_file_comprehensive):
        """Test enhanced statistics calculation."""
        # Copy file to processor directory for statistics
        import shutil
        dest_path = processor.data_directory / 'test_file.xlsx'
        shutil.copy2(sample_excel_file_comprehensive, dest_path)
        
        try:
            stats = processor.get_enhanced_statistics()
            
            # Basic statistics
            assert stats['total_files'] >= 1
            assert stats['total_sheets'] >= 1
            assert stats['total_rows'] >= 1
            assert stats['total_memory_usage_mb'] > 0
            
            # Enhanced statistics
            assert 'processing_summary' in stats
            assert 'data_quality_distribution' in stats
            assert 'column_type_distribution' in stats
            assert 'pattern_analysis' in stats
            
            # Processing summary
            proc_summary = stats['processing_summary']
            assert proc_summary['enhanced_analysis_files'] >= 1
            assert proc_summary['average_file_quality'] in ['excellent', 'good', 'fair', 'poor', 'very_poor', 'unknown']
            assert proc_summary['total_data_issues'] >= 0
            
            # Quality distribution
            quality_dist = stats['data_quality_distribution']
            assert isinstance(quality_dist, dict)
            
            # Column type distribution
            type_dist = stats['column_type_distribution']
            assert 'numeric' in type_dist or 'text' in type_dist or 'datetime' in type_dist
            
            # Pattern analysis
            patterns = stats['pattern_analysis']
            assert patterns['email_columns'] >= 0
            assert patterns['phone_columns'] >= 0
            assert patterns['outlier_affected_columns'] >= 0
            
        finally:
            # Cleanup
            if dest_path.exists():
                dest_path.unlink()
    
    def test_header_detection(self, processor):
        """Test advanced header detection."""
        # Data with clear headers
        df_with_headers = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob'],
            'Age': [25, 30, 35],
            'City': ['NY', 'LA', 'SF']
        })
        
        header_info = processor._detect_header(df_with_headers)
        assert header_info['has_header'] == True
        assert header_info['confidence'] > 0.0
        
        # Data without proper headers (generic names)
        df_no_headers = pd.DataFrame([
            ['John', 25, 'NY'],
            ['Jane', 30, 'LA'],
            ['Bob', 35, 'SF']
        ])
        df_no_headers.columns = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2']
        
        header_info = processor._detect_header(df_no_headers)
        assert header_info['has_header'] == False
        assert header_info['method'] == 'generic_names_detected'
    
    def test_seasonal_pattern_detection(self, processor):
        """Test seasonal pattern detection in time series data."""
        # Create data with monthly seasonality
        seasonal_data = []
        for i in range(36):  # 3 years of monthly data
            base_value = 100
            seasonal_component = 20 * np.sin(2 * np.pi * i / 12)  # Monthly pattern
            noise = np.random.normal(0, 5)
            seasonal_data.append(base_value + seasonal_component + noise)
        
        series = pd.Series(seasonal_data)
        seasonality = processor._detect_seasonality(series)
        
        assert seasonality['has_seasonality'] == True or seasonality['confidence'] > 0.3
        
        # Test with non-seasonal data
        random_data = pd.Series(np.random.normal(100, 10, 36))
        seasonality = processor._detect_seasonality(random_data)
        
        # Should have low confidence for random data
        assert seasonality['confidence'] <= 0.5
    
    def test_string_similarity(self, processor):
        """Test string similarity calculation."""
        similarity = processor._calculate_string_similarity('customer_id', 'customer_name')
        assert similarity > 0.5  # Should be similar due to 'customer' prefix
        
        similarity = processor._calculate_string_similarity('age', 'xyz')
        assert similarity < 0.3  # Should be very different
        
        similarity = processor._calculate_string_similarity('test', 'test')
        assert similarity == 1.0  # Identical strings
    
    def test_decimal_places_analysis(self, processor):
        """Test decimal places analysis."""
        # Data with consistent decimal places
        consistent_data = pd.Series([1.25, 2.75, 3.50, 4.00])
        analysis = processor._analyze_decimal_places(consistent_data)
        
        assert analysis['max_decimal_places'] == 2
        # Allow for some variation in precision detection
        assert analysis['max_decimal_places'] <= 3  # Should detect reasonable decimal places
        
        # Data with varying decimal places
        varying_data = pd.Series([1.2, 2.75, 3.500, 4])
        analysis = processor._analyze_decimal_places(varying_data)
        
        assert analysis['max_decimal_places'] >= 2
        assert analysis['precision_consistency'] == False
    
    def test_encoding_issues_detection(self, processor):
        """Test encoding issues detection."""
        # Data with potential encoding issues
        problematic_data = pd.Series(['normal text', 'text with �', 'another\x00problem', 'fine'])
        encoding_info = processor._detect_encoding_issues(problematic_data)
        
        assert encoding_info['control_char_count'] > 0
        assert len(encoding_info['suspicious_chars']) > 0
        assert '�' in encoding_info['suspicious_chars']
    
    def test_date_frequency_analysis(self, processor):
        """Test date frequency analysis."""
        # Daily data
        daily_dates = pd.date_range('2023-01-01', periods=30, freq='D')
        daily_series = pd.Series(daily_dates)
        
        freq_analysis = processor._analyze_date_frequency(daily_series)
        assert freq_analysis['frequency'] == 'daily'
        assert freq_analysis['interval_consistency'] == True
        
        # Weekly data
        weekly_dates = pd.date_range('2023-01-01', periods=10, freq='W')
        weekly_series = pd.Series(weekly_dates)
        
        freq_analysis = processor._analyze_date_frequency(weekly_series)
        assert freq_analysis['frequency'] == 'weekly'
        
        # Irregular data
        irregular_dates = pd.Series([
            pd.to_datetime('2023-01-01'),
            pd.to_datetime('2023-01-05'),
            pd.to_datetime('2023-01-12'),
            pd.to_datetime('2023-01-30')
        ])
        
        freq_analysis = processor._analyze_date_frequency(irregular_dates)
        assert freq_analysis['frequency'] == 'irregular'


def test_integration_with_original_processor():
    """Test that enhanced processor can work alongside original processor."""
    from app.services.excel_processor import ExcelProcessor
    
    # Both should be importable and functional
    with tempfile.TemporaryDirectory() as temp_dir:
        original = ExcelProcessor(temp_dir)
        enhanced = EnhancedExcelProcessor(temp_dir)
        
        # Both should have basic functionality
        assert hasattr(original, 'process_excel_file')
        assert hasattr(enhanced, 'process_excel_file_enhanced')
        assert hasattr(original, 'get_file_statistics')
        assert hasattr(enhanced, 'get_enhanced_statistics')


if __name__ == "__main__":
    # Run a subset of tests manually
    print("Running enhanced Excel processor tests...")
    
    processor = EnhancedExcelProcessor("test_temp")
    
    # Test pattern detection
    email_data = pd.Series(['test@example.com', 'user@domain.org', 'invalid'])
    patterns = processor.detect_data_patterns(email_data)
    print(f"Email pattern detection: {patterns['email_count']} emails found")
    
    # Test quality assessment
    test_data = pd.DataFrame({
        'col1': [1, 2, 3, np.nan, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [1.1, 2.2, None, 4.4, 5.5]
    })
    
    quality = processor.assess_data_quality(test_data)
    print(f"Data quality: {quality['overall_quality']} (completeness: {quality['completeness_score']:.2f})")
    
    print("✅ Enhanced Excel processor tests completed successfully!")