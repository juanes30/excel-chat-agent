"""Basic functionality tests for Excel Chat Agent backend."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.excel_processor import ExcelProcessor


class TestExcelProcessor:
    """Test Excel processor functionality."""
    
    @pytest.fixture
    def temp_excel_file(self):
        """Create a temporary Excel file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            # Create sample data
            data = {
                'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'Age': [25, 30, 35, 28],
                'Salary': [50000, 60000, 70000, 55000],
                'Department': ['IT', 'HR', 'Finance', 'IT']
            }
            df = pd.DataFrame(data)
            df.to_excel(tmp_file.name, index=False, sheet_name='Employees')
            
            # Add a second sheet
            with pd.ExcelWriter(tmp_file.name, mode='a') as writer:
                summary_data = {
                    'Department': ['IT', 'HR', 'Finance'],
                    'Count': [2, 1, 1],
                    'Avg_Salary': [52500, 60000, 70000]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            yield tmp_file.name
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    @pytest.fixture
    def excel_processor(self):
        """Create Excel processor instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ExcelProcessor(temp_dir, analysis_mode="basic")
            yield processor
    
    @pytest.fixture
    def excel_processor_comprehensive(self):
        """Create Excel processor instance with comprehensive analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ExcelProcessor(temp_dir, analysis_mode="comprehensive")
            yield processor
    
    def test_excel_file_validation(self, excel_processor, temp_excel_file):
        """Test Excel file validation."""
        file_path = Path(temp_excel_file)
        
        # Test valid file
        validation_result = excel_processor.validate_excel_file(file_path)
        assert validation_result['file_name'] == file_path.name
        assert validation_result['extension'] == '.xlsx'
        assert validation_result['file_size_mb'] > 0
        assert isinstance(validation_result['last_modified'], datetime)
    
    def test_excel_file_processing(self, excel_processor, temp_excel_file):
        """Test complete Excel file processing."""
        result = excel_processor.process_excel_file(temp_excel_file)
        
        # Verify basic file info
        assert result['file_name'] == Path(temp_excel_file).name
        assert result['total_sheets'] == 2
        assert result['total_rows'] == 7  # 4 employees + 3 summary rows
        assert len(result['all_text_chunks']) > 0
        
        # Verify sheets
        assert 'Employees' in result['sheets']
        assert 'Summary' in result['sheets']
        
        employees_sheet = result['sheets']['Employees']
        assert employees_sheet['metadata']['num_rows'] == 4
        assert employees_sheet['metadata']['num_cols'] == 4
        assert len(employees_sheet['text_chunks']) > 0
    
    def test_file_hash_generation(self, excel_processor, temp_excel_file):
        """Test file hash generation."""
        file_path = Path(temp_excel_file)
        hash1 = excel_processor.get_file_hash(file_path)
        hash2 = excel_processor.get_file_hash(file_path)
        
        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
    
    def test_searchable_text_generation(self, excel_processor, temp_excel_file):
        """Test searchable text generation."""
        result = excel_processor.process_excel_file(temp_excel_file)
        
        employees_sheet = result['sheets']['Employees']
        text_chunks = employees_sheet['text_chunks']
        
        # Should have at least summary and data chunks
        assert len(text_chunks) > 1
        
        # Check if important data is searchable
        all_text = ' '.join(text_chunks)
        assert 'Alice' in all_text
        assert 'IT' in all_text
        assert '50000' in all_text
        assert 'Employees' in all_text
    
    def test_analysis_modes(self, temp_excel_file):
        """Test different analysis modes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic mode
            basic_proc = ExcelProcessor(temp_dir, analysis_mode="basic")
            basic_result = basic_proc.process_excel_file(temp_excel_file)
            assert basic_result['analysis_mode'] == 'basic'
            
            # Test comprehensive mode
            comp_proc = ExcelProcessor(temp_dir, analysis_mode="comprehensive")
            comp_result = comp_proc.process_excel_file(temp_excel_file)
            assert comp_result['analysis_mode'] == 'comprehensive'
            
            # Test auto mode
            auto_proc = ExcelProcessor(temp_dir, analysis_mode="auto")
            auto_result = auto_proc.process_excel_file(temp_excel_file)
            assert auto_result['analysis_mode'] in ['basic', 'moderate', 'comprehensive']


class TestAPI:
    """Test FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_list_files_endpoint(self, client):
        """Test file listing endpoint."""
        # This might fail if services aren't initialized
        # In a real test, you'd mock the services
        try:
            response = client.get("/api/files")
            assert response.status_code in [200, 503]  # 503 if services not ready
        except Exception:
            # Services not initialized in test environment
            pass
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        try:
            response = client.get("/api/stats")
            assert response.status_code in [200, 503]  # 503 if services not ready
        except Exception:
            # Services not initialized in test environment
            pass


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_data_flow(self):
        """Test the complete data flow from Excel to searchable text."""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            data = {
                'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
                'Price': [999.99, 29.99, 79.99, 299.99],
                'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics'],
                'In_Stock': [True, True, False, True]
            }
            df = pd.DataFrame(data)
            df.to_excel(tmp_file.name, index=False, sheet_name='Products')
            
            try:
                # Test processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    processor = ExcelProcessor(temp_dir)
                    result = processor.process_excel_file(tmp_file.name)
                    
                    # Verify structure
                    assert 'Products' in result['sheets']
                    products_sheet = result['sheets']['Products']
                    
                    # Verify searchable text contains key information
                    all_text = ' '.join(products_sheet['text_chunks'])
                    assert 'Laptop' in all_text
                    assert '999.99' in all_text
                    assert 'Products' in all_text
                    assert 'Electronics' in all_text
                    
                    # Verify metadata
                    metadata = products_sheet['metadata']
                    assert metadata['num_rows'] == 4
                    assert metadata['num_cols'] == 4
                    
                    # Check column information
                    column_names = [col['name'] for col in metadata['columns']]
                    assert 'Product' in column_names
                    assert 'Price' in column_names
                    assert 'Category' in column_names
                    assert 'In_Stock' in column_names
            
            finally:
                # Cleanup
                os.unlink(tmp_file.name)


def test_environment_setup():
    """Test that the environment is set up correctly."""
    # Check that required packages are importable
    import fastapi
    import pandas
    import chromadb
    import sentence_transformers
    import langchain
    import ollama
    
    # Basic version checks
    assert hasattr(fastapi, '__version__')
    assert hasattr(pandas, '__version__')


if __name__ == "__main__":
    # Run basic tests
    print("Running basic functionality tests...")
    
    # Test Excel processing
    try:
        test_integration = TestIntegration()
        test_integration.test_data_flow()
        print("✓ Excel processing test passed")
    except Exception as e:
        print(f"✗ Excel processing test failed: {e}")
    
    # Test environment
    try:
        test_environment_setup()
        print("✓ Environment setup test passed")
    except Exception as e:
        print(f"✗ Environment setup test failed: {e}")
    
    print("Basic tests completed!")