"""Excel Processor Service for analyzing and extracting data from Excel files."""

import hashlib
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from openpyxl import load_workbook

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Service for processing Excel files and extracting structured data."""

    def __init__(self, data_directory: str = "data/excel_files"):
        """Initialize the Excel processor.
        
        Args:
            data_directory: Directory containing Excel files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Supported Excel file extensions
        self.supported_extensions = {'.xlsx', '.xls', '.xlsm'}
        
        # Maximum file size in MB
        self.max_file_size_mb = 100
        
        # Cache for file hashes and metadata
        self._file_cache = {}

    def get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash for a file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def validate_excel_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate Excel file before processing.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If file is invalid
        """
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB")
        
        return {
            "file_name": file_path.name,
            "file_size_mb": round(file_size_mb, 2),
            "extension": file_path.suffix.lower(),
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime)
        }

    def extract_sheet_metadata(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Extract metadata from a DataFrame.
        
        Args:
            df: Pandas DataFrame
            sheet_name: Name of the Excel sheet
            
        Returns:
            Dictionary with sheet metadata
        """
        # Basic dimensions
        num_rows, num_cols = df.shape
        
        # Column information
        columns_info = []
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                dtype = str(df[col].dtype)
                non_null_count = col_data.count()
                
                # Detect data types more specifically
                if pd.api.types.is_numeric_dtype(df[col]):
                    data_type = "numeric"
                    stats = {
                        "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                        "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                        "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                        "std": float(col_data.std()) if not pd.isna(col_data.std()) else None
                    }
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    data_type = "datetime"
                    stats = {
                        "min": str(col_data.min()),
                        "max": str(col_data.max())
                    }
                else:
                    data_type = "text"
                    unique_count = col_data.nunique()
                    stats = {
                        "unique_values": min(unique_count, 10),
                        "sample_values": col_data.unique()[:5].tolist()
                    }
                
                columns_info.append({
                    "name": str(col),
                    "data_type": data_type,
                    "dtype": dtype,
                    "non_null_count": int(non_null_count),
                    "null_count": int(num_rows - non_null_count),
                    "stats": stats
                })
        
        return {
            "sheet_name": sheet_name,
            "num_rows": int(num_rows),
            "num_cols": int(num_cols),
            "columns": columns_info,
            "has_header": True,  # Assume first row is header
            "data_preview": df.head(3).to_dict('records') if num_rows > 0 else []
        }

    def create_searchable_text(self, df: pd.DataFrame, sheet_name: str, 
                             metadata: Dict[str, Any]) -> List[str]:
        """Create searchable text representations of the data.
        
        Args:
            df: Pandas DataFrame
            sheet_name: Name of the Excel sheet
            metadata: Sheet metadata
            
        Returns:
            List of text chunks for vector search
        """
        text_chunks = []
        
        # Sheet summary
        summary = (
            f"Sheet: {sheet_name}\n"
            f"Dimensions: {metadata['num_rows']} rows × {metadata['num_cols']} columns\n"
            f"Columns: {', '.join([col['name'] for col in metadata['columns']])}\n"
        )
        text_chunks.append(summary)
        
        # Column descriptions
        for col_info in metadata['columns']:
            col_desc = (
                f"Column '{col_info['name']}' in sheet '{sheet_name}':\n"
                f"Type: {col_info['data_type']}\n"
                f"Non-null values: {col_info['non_null_count']}\n"
            )
            
            if col_info['data_type'] == 'numeric':
                stats = col_info['stats']
                col_desc += (
                    f"Range: {stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}\n"
                    f"Mean: {stats.get('mean', 'N/A')}\n"
                )
            elif col_info['data_type'] == 'text':
                stats = col_info['stats']
                col_desc += f"Sample values: {', '.join(map(str, stats.get('sample_values', [])))}\n"
            
            text_chunks.append(col_desc)
        
        # Data samples (chunked by rows)
        chunk_size = 50  # Process data in chunks
        for i in range(0, min(len(df), 500), chunk_size):  # Limit to first 500 rows
            chunk_df = df.iloc[i:i+chunk_size]
            
            # Convert chunk to text
            chunk_text = f"Data from sheet '{sheet_name}' (rows {i+1}-{i+len(chunk_df)}):\n"
            
            # Add key-value pairs for better searchability
            for _, row in chunk_df.iterrows():
                row_items = []
                for col, value in row.items():
                    if pd.notna(value):
                        row_items.append(f"{col}: {value}")
                
                if row_items:
                    chunk_text += " | ".join(row_items) + "\n"
            
            if chunk_text.strip():
                text_chunks.append(chunk_text)
        
        return text_chunks

    @lru_cache(maxsize=32)
    def process_excel_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single Excel file and extract all relevant information.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with file information and processed data
        """
        file_path = Path(file_path)
        
        try:
            # Validate file
            file_info = self.validate_excel_file(file_path)
            file_hash = self.get_file_hash(file_path)
            
            logger.info(f"Processing Excel file: {file_path.name}")
            
            # Read all sheets
            try:
                # Try to read with openpyxl first (better for .xlsx files)
                if file_path.suffix.lower() in ['.xlsx', '.xlsm']:
                    excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
                else:
                    excel_data = pd.read_excel(file_path, sheet_name=None)
            except Exception as e:
                logger.warning(f"Failed to read with pandas, trying openpyxl: {e}")
                # Fallback: read sheet names only
                workbook = load_workbook(file_path, read_only=True)
                sheet_names = workbook.sheetnames
                excel_data = {}
                for sheet_name in sheet_names:
                    try:
                        excel_data[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
                    except Exception as sheet_error:
                        logger.error(f"Failed to read sheet {sheet_name}: {sheet_error}")
                        continue
            
            # Process each sheet
            sheets_data = {}
            total_rows = 0
            total_columns = set()
            all_text_chunks = []
            
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                
                # Clean the DataFrame
                df = df.dropna(how='all')  # Remove completely empty rows
                df = df.dropna(axis=1, how='all')  # Remove completely empty columns
                
                if df.empty:
                    continue
                
                # Extract metadata
                metadata = self.extract_sheet_metadata(df, sheet_name)
                
                # Create searchable text
                text_chunks = self.create_searchable_text(df, sheet_name, metadata)
                
                # Store sheet data
                sheets_data[sheet_name] = {
                    "metadata": metadata,
                    "text_chunks": text_chunks,
                    "data": df.to_dict('records')[:100]  # Store first 100 rows as sample
                }
                
                total_rows += metadata['num_rows']
                total_columns.update([col['name'] for col in metadata['columns']])
                all_text_chunks.extend(text_chunks)
            
            result = {
                "file_name": file_info["file_name"],
                "file_hash": file_hash,
                "file_path": str(file_path),
                "file_size_mb": file_info["file_size_mb"],
                "last_modified": file_info["last_modified"],
                "extension": file_info["extension"],
                "total_sheets": len(sheets_data),
                "total_rows": total_rows,
                "total_columns": len(total_columns),
                "sheets": sheets_data,
                "all_text_chunks": all_text_chunks,
                "processed_at": datetime.now()
            }
            
            logger.info(f"Successfully processed {file_path.name}: "
                       f"{len(sheets_data)} sheets, {total_rows} rows, "
                       f"{len(all_text_chunks)} text chunks")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all Excel files in the data directory.
        
        Returns:
            List of processed file data
        """
        results = []
        
        # Find all Excel files
        excel_files = []
        for ext in self.supported_extensions:
            excel_files.extend(self.data_directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(excel_files)} Excel files to process")
        
        for file_path in excel_files:
            try:
                result = self.process_excel_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return results

    def query_data(self, file_name: Optional[str] = None, 
                   sheet_name: Optional[str] = None,
                   column_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query processed data with optional filters.
        
        Args:
            file_name: Filter by file name
            sheet_name: Filter by sheet name
            column_name: Filter by column name
            
        Returns:
            List of matching data chunks
        """
        all_files = self.process_all_files()
        results = []
        
        for file_data in all_files:
            if file_name and file_name.lower() not in file_data['file_name'].lower():
                continue
            
            for sheet_name_key, sheet_data in file_data['sheets'].items():
                if sheet_name and sheet_name.lower() not in sheet_name_key.lower():
                    continue
                
                if column_name:
                    # Filter by column name
                    matching_columns = [
                        col for col in sheet_data['metadata']['columns']
                        if column_name.lower() in col['name'].lower()
                    ]
                    if not matching_columns:
                        continue
                
                # Add file and sheet context to results
                for chunk in sheet_data['text_chunks']:
                    results.append({
                        "file_name": file_data['file_name'],
                        "sheet_name": sheet_name_key,
                        "content": chunk,
                        "file_hash": file_data['file_hash']
                    })
        
        return results

    def get_file_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed files.
        
        Returns:
            Dictionary with file statistics
        """
        all_files = self.process_all_files()
        
        if not all_files:
            return {
                "total_files": 0,
                "total_sheets": 0,
                "total_rows": 0,
                "total_columns": 0,
                "total_text_chunks": 0
            }
        
        total_files = len(all_files)
        total_sheets = sum(file_data['total_sheets'] for file_data in all_files)
        total_rows = sum(file_data['total_rows'] for file_data in all_files)
        total_columns = sum(file_data['total_columns'] for file_data in all_files)
        total_text_chunks = sum(len(file_data['all_text_chunks']) for file_data in all_files)
        
        return {
            "total_files": total_files,
            "total_sheets": total_sheets,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_text_chunks": total_text_chunks,
            "average_sheets_per_file": round(total_sheets / total_files, 2),
            "average_rows_per_file": round(total_rows / total_files, 2)
        }