/**
 * FileSelector component for managing Excel file uploads and selection
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  Upload,
  File,
  FileSpreadsheet,
  X,
  Download,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Loader2,
  Calendar,
  HardDrive
} from 'lucide-react';
import { ExcelFileInfo } from '../types/chat.types';

interface FileSelectorProps {
  files: ExcelFileInfo[];
  selectedFile: string | null;
  selectedSheet: string | null;
  onFileSelect: (fileName: string | null) => void;
  onSheetSelect: (sheetName: string | null) => void;
  onFileUpload: (file: File) => Promise<void>;
  onRefresh: () => Promise<void>;
  isUploading?: boolean;
  uploadProgress?: number;
  className?: string;
  allowMultipleFiles?: boolean;
  maxFileSize?: number; // in MB
}

interface FileItemProps {
  file: ExcelFileInfo;
  isSelected: boolean;
  selectedSheet: string | null;
  onSelect: (fileName: string) => void;
  onSheetSelect: (sheetName: string | null) => void;
}

const FileItem: React.FC<FileItemProps> = ({
  file,
  isSelected,
  selectedSheet,
  onSelect,
  onSheetSelect
}) => {
  const [isExpanded, setIsExpanded] = useState(isSelected);

  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) {
      return `${Math.round(sizeInMB * 1024)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  const formatDate = (date: Date): string => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <div className={`border rounded-lg transition-all ${
      isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:border-gray-300'
    }`}>
      <div
        className="p-3 cursor-pointer"
        onClick={() => {
          onSelect(file.name);
          setIsExpanded(!isExpanded);
        }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <FileSpreadsheet className={`w-5 h-5 flex-shrink-0 ${
              isSelected ? 'text-blue-600' : 'text-green-600'
            }`} />

            <div className="flex-1 min-w-0">
              <div className="font-medium text-sm truncate" title={file.name}>
                {file.name}
              </div>
              <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                <div className="flex items-center gap-1">
                  <File className="w-3 h-3" />
                  <span>{file.total_sheets} sheet{file.total_sheets !== 1 ? 's' : ''}</span>
                </div>
                <div className="flex items-center gap-1">
                  <HardDrive className="w-3 h-3" />
                  <span>{formatFileSize(file.file_size_mb)}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  <span>{formatDate(file.last_modified)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isSelected && (
              <CheckCircle className="w-4 h-4 text-blue-600" />
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(!isExpanded);
              }}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${
                  isExpanded ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Sheet Selection */}
      {isExpanded && file.sheets.length > 0 && (
        <div className="border-t border-gray-200 p-3 bg-gray-50">
          <div className="text-xs font-medium text-gray-700 mb-2">
            Select Sheet:
          </div>
          <div className="space-y-1">
            <button
              onClick={() => onSheetSelect(null)}
              className={`w-full text-left px-2 py-1 text-xs rounded transition-colors ${
                !selectedSheet
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              All Sheets
            </button>
            {file.sheets.map((sheet) => (
              <button
                key={sheet}
                onClick={() => onSheetSelect(sheet)}
                className={`w-full text-left px-2 py-1 text-xs rounded transition-colors truncate ${
                  selectedSheet === sheet
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
                title={sheet}
              >
                📊 {sheet}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* File Statistics */}
      {isExpanded && (
        <div className="border-t border-gray-200 p-3 bg-gray-50">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-gray-500">Total Rows:</span>
              <span className="ml-1 font-medium">{file.total_rows.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500">Total Columns:</span>
              <span className="ml-1 font-medium">{file.total_columns.toLocaleString()}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const FileSelector: React.FC<FileSelectorProps> = ({
  files,
  selectedFile,
  selectedSheet,
  onFileSelect,
  onSheetSelect,
  onFileUpload,
  onRefresh,
  isUploading = false,
  uploadProgress = 0,
  className = '',
  allowMultipleFiles = true,
  maxFileSize = 50
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((fileName: string) => {
    if (selectedFile === fileName) {
      onFileSelect(null);
      onSheetSelect(null);
    } else {
      onFileSelect(fileName);
      onSheetSelect(null);
    }
  }, [selectedFile, onFileSelect, onSheetSelect]);

  const validateFile = useCallback((file: File): string | null => {
    if (!file.name.match(/\.(xlsx?|csv)$/i)) {
      return 'Please select an Excel file (.xlsx, .xls) or CSV file';
    }

    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size must be less than ${maxFileSize}MB`;
    }

    return null;
  }, [maxFileSize]);

  const handleFileUpload = useCallback(async (file: File) => {
    const error = validateFile(file);
    if (error) {
      setUploadError(error);
      return;
    }

    try {
      setUploadError(null);
      await onFileUpload(file);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Upload failed');
    }
  }, [validateFile, onFileUpload]);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
    // Reset input value to allow re-uploading the same file
    e.target.value = '';
  }, [handleFileUpload]);

  const selectedFileData = files.find(f => f.name === selectedFile);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Excel Files</h3>
        <button
          onClick={onRefresh}
          disabled={isUploading}
          className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Refresh file list"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Upload Area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleFileDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
          dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileInput}
          className="hidden"
        />

        {isUploading ? (
          <div className="space-y-3">
            <Loader2 className="w-8 h-8 text-blue-600 mx-auto animate-spin" />
            <div className="text-sm text-gray-600">Uploading file...</div>
            {uploadProgress > 0 && (
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <Upload className="w-8 h-8 text-gray-400 mx-auto" />
            <div className="text-sm text-gray-600">
              Drag and drop an Excel file here, or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-blue-600 hover:text-blue-800 underline"
              >
                browse files
              </button>
            </div>
            <div className="text-xs text-gray-500">
              Supports .xlsx, .xls, and .csv files up to {maxFileSize}MB
            </div>
          </div>
        )}
      </div>

      {/* Upload Error */}
      {uploadError && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2 text-red-800">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{uploadError}</span>
          </div>
        </div>
      )}

      {/* File List */}
      {files.length > 0 ? (
        <div className="space-y-3">
          <div className="text-sm text-gray-600">
            {files.length} file{files.length !== 1 ? 's' : ''} available
          </div>
          {files.map((file) => (
            <FileItem
              key={file.name}
              file={file}
              isSelected={selectedFile === file.name}
              selectedSheet={selectedSheet}
              onSelect={handleFileSelect}
              onSheetSelect={onSheetSelect}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <FileSpreadsheet className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <div className="text-sm">No Excel files uploaded yet</div>
          <div className="text-xs text-gray-400 mt-1">
            Upload your first file to get started
          </div>
        </div>
      )}

      {/* Clear Selection */}
      {selectedFile && (
        <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-2 text-blue-800">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm">
              Selected: {selectedFile}
              {selectedSheet && ` → ${selectedSheet}`}
            </span>
          </div>
          <button
            onClick={() => {
              onFileSelect(null);
              onSheetSelect(null);
            }}
            className="text-blue-600 hover:text-blue-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};