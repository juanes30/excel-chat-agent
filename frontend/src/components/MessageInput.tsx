/**
 * MessageInput component for sending chat queries with Excel file filtering
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Mic, MicOff, Paperclip, Settings, X } from 'lucide-react';
import { QueryFilters } from '../types/chat.types';

interface MessageInputProps {
  onSendMessage: (message: string, filters?: QueryFilters) => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  className?: string;
  showFilters?: boolean;
  availableFiles?: string[];
  availableSheets?: string[];
  selectedFile?: string | null;
  selectedSheet?: string | null;
  onFileSelect?: (file: string | null) => void;
  onSheetSelect?: (sheet: string | null) => void;
  isConnected?: boolean;
}

interface VoiceRecognition {
  start: () => void;
  stop: () => void;
  supported: boolean;
}

export const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = "Ask a question about your Excel data...",
  maxLength = 1000,
  className = '',
  showFilters = true,
  availableFiles = [],
  availableSheets = [],
  selectedFile,
  selectedSheet,
  onFileSelect,
  onSheetSelect,
  isConnected = true
}) => {
  const [message, setMessage] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [maxResults, setMaxResults] = useState(5);
  const [includeStatistics, setIncludeStatistics] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<any>(null);

  // Voice recognition setup
  const voiceRecognition: VoiceRecognition = React.useMemo(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      return { start: () => {}, stop: () => {}, supported: false };
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setMessage(prev => prev + transcript);
      setIsRecording(false);
    };

    recognition.onerror = () => {
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;

    return {
      start: () => {
        setIsRecording(true);
        recognition.start();
      },
      stop: () => {
        setIsRecording(false);
        recognition.stop();
      },
      supported: true
    };
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();

    if (!message.trim() || disabled || !isConnected) {
      return;
    }

    const filters: QueryFilters = {
      fileFilter: selectedFile || undefined,
      sheetFilter: selectedSheet || undefined,
      maxResults,
      includeStatistics
    };

    onSendMessage(message.trim(), filters);
    setMessage('');
    setShowAdvanced(false);

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [message, disabled, isConnected, selectedFile, selectedSheet, maxResults, includeStatistics, onSendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }, [handleSubmit]);

  const handleVoiceToggle = useCallback(() => {
    if (isRecording) {
      voiceRecognition.stop();
    } else {
      voiceRecognition.start();
    }
  }, [isRecording, voiceRecognition]);

  const isSubmitDisabled = !message.trim() || disabled || !isConnected;
  const remainingChars = maxLength - message.length;

  return (
    <div className={`bg-white border-t border-gray-200 ${className}`}>
      {/* Filters Section */}
      {showFilters && showAdvanced && (
        <div className="p-4 border-b border-gray-100 bg-gray-50">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-700">Query Filters</h3>
            <button
              onClick={() => setShowAdvanced(false)}
              className="text-gray-400 hover:text-gray-600 focus:outline-none"
              aria-label="Close filters"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* File Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Excel File
              </label>
              <select
                value={selectedFile || ''}
                onChange={(e) => onFileSelect?.(e.target.value || null)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All files</option>
                {availableFiles.map((file) => (
                  <option key={file} value={file}>
                    {file}
                  </option>
                ))}
              </select>
            </div>

            {/* Sheet Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Sheet
              </label>
              <select
                value={selectedSheet || ''}
                onChange={(e) => onSheetSelect?.(e.target.value || null)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={!selectedFile}
              >
                <option value="">All sheets</option>
                {availableSheets.map((sheet) => (
                  <option key={sheet} value={sheet}>
                    {sheet}
                  </option>
                ))}
              </select>
            </div>

            {/* Max Results */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Results
              </label>
              <select
                value={maxResults}
                onChange={(e) => setMaxResults(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value={3}>3 results</option>
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={20}>20 results</option>
              </select>
            </div>

            {/* Statistics Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Options
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includeStatistics}
                  onChange={(e) => setIncludeStatistics(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700">Include statistics</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Current Filters Display */}
      {showFilters && (selectedFile || selectedSheet) && (
        <div className="px-4 py-2 bg-blue-50 border-b border-blue-100">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-blue-700">Filtering:</span>
            {selectedFile && (
              <span className="bg-blue-200 text-blue-800 px-2 py-1 rounded">
                📄 {selectedFile}
              </span>
            )}
            {selectedSheet && (
              <span className="bg-blue-200 text-blue-800 px-2 py-1 rounded">
                📊 {selectedSheet}
              </span>
            )}
            <button
              onClick={() => {
                onFileSelect?.(null);
                onSheetSelect?.(null);
              }}
              className="text-blue-600 hover:text-blue-800 ml-auto"
            >
              Clear filters
            </button>
          </div>
        </div>
      )}

      {/* Input Section */}
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex items-end gap-3">
          {/* Main Input Area */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value.slice(0, maxLength))}
              onKeyDown={handleKeyDown}
              placeholder={isConnected ? placeholder : "Connecting..."}
              disabled={disabled || !isConnected}
              rows={1}
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
              style={{ minHeight: '52px' }}
            />

            {/* Character Count */}
            {message.length > maxLength * 0.8 && (
              <div className={`absolute bottom-2 right-2 text-xs ${remainingChars < 0 ? 'text-red-500' : 'text-gray-400'}`}>
                {remainingChars}
              </div>
            )}
          </div>

          {/* Voice Recording Button */}
          {voiceRecognition.supported && (
            <button
              type="button"
              onClick={handleVoiceToggle}
              disabled={disabled || !isConnected}
              className={`p-3 rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                isRecording
                  ? 'bg-red-600 text-white border-red-600 hover:bg-red-700 focus:ring-red-500'
                  : 'bg-gray-100 text-gray-600 border-gray-300 hover:bg-gray-200 focus:ring-gray-500'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
            >
              {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
          )}

          {/* Settings Button */}
          {showFilters && (
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              disabled={disabled}
              className={`p-3 rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed ${
                showAdvanced
                  ? 'bg-blue-600 text-white border-blue-600 hover:bg-blue-700'
                  : 'bg-gray-100 text-gray-600 border-gray-300 hover:bg-gray-200'
              }`}
              aria-label="Toggle advanced filters"
            >
              <Settings className="w-5 h-5" />
            </button>
          )}

          {/* Send Button */}
          <button
            type="submit"
            disabled={isSubmitDisabled}
            className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>

        {/* Connection Status */}
        {!isConnected && (
          <div className="mt-2 text-sm text-amber-600 flex items-center gap-1">
            <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></div>
            <span>Reconnecting to server...</span>
          </div>
        )}

        {/* Keyboard Shortcut Hint */}
        <div className="mt-2 text-xs text-gray-500">
          Press Enter to send, Shift+Enter for new line
        </div>
      </form>
    </div>
  );
};