/**
 * Main ChatContainer component that integrates all chat functionality
 */

import React, { useState, useEffect, useCallback } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { ConnectionStatus } from './ConnectionStatus';
import { FileSelector } from './FileSelector';
import { useChatContext } from '../context/ChatContext';
import {
  Menu,
  X,
  Settings,
  BarChart3,
  HelpCircle,
  Maximize2,
  Minimize2,
  Download,
  Trash2
} from 'lucide-react';

interface ChatContainerProps {
  className?: string;
  websocketUrl?: string;
  autoConnect?: boolean;
  maxMessages?: number;
  enableFileUpload?: boolean;
  showSidebar?: boolean;
  theme?: 'light' | 'dark';
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  className = '',
  enableFileUpload = true,
  showSidebar: defaultShowSidebar = true,
  theme = 'light'
}) => {
  const {
    state,
    sendQuery,
    clearMessages,
    reconnect,
    uploadFile,
    refreshFiles,
    retryLastQuery,
    setSelectedFile,
    setSelectedSheet
  } = useChatContext();

  const [showSidebar, setShowSidebar] = useState(defaultShowSidebar);
  const [showSettings, setShowSettings] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<'files' | 'stats' | 'help'>('files');

  // Responsive sidebar handling
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setShowSidebar(false);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Check on mount

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleSendQuery = useCallback(async (message: string, filters?: any) => {
    try {
      await sendQuery(message, filters);
    } catch (error) {
      console.error('Failed to send query:', error);
    }
  }, [sendQuery]);

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      await uploadFile(file);
    } catch (error) {
      console.error('File upload failed:', error);
      throw error;
    }
  }, [uploadFile]);

  const exportMessages = useCallback(() => {
    const data = {
      exported_at: new Date().toISOString(),
      messages: state.messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp.toISOString()
      })),
      session_info: {
        sessionId: state.sessionId,
        connectionState: state.connectionState,
        totalMessages: state.messages.length,
        selectedFile: state.selectedFile,
        selectedSheet: state.selectedSheet
      }
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [state]);

  const getAvailableSheets = useCallback(() => {
    if (!state.selectedFile) return [];
    const file = state.files.find(f => f.name === state.selectedFile);
    return file?.sheets || [];
  }, [state.selectedFile, state.files]);

  const renderSidebarContent = () => {
    switch (sidebarTab) {
      case 'files':
        return enableFileUpload ? (
          <FileSelector
            files={state.files}
            selectedFile={state.selectedFile}
            selectedSheet={state.selectedSheet}
            onFileSelect={setSelectedFile}
            onSheetSelect={setSelectedSheet}
            onFileUpload={handleFileUpload}
            onRefresh={refreshFiles}
            isUploading={state.isUploading}
            uploadProgress={state.uploadProgress}
          />
        ) : (
          <div className="p-4 text-center text-gray-500">
            <HelpCircle className="w-8 h-8 mx-auto mb-2" />
            <p className="text-sm">File upload is disabled</p>
          </div>
        );

      case 'stats':
        return (
          <div className="p-4 space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Statistics</h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {state.messages.filter(m => m.type === 'user').length}
                </div>
                <div className="text-sm text-blue-800">Queries</div>
              </div>

              <div className="bg-green-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {state.messages.filter(m => m.type === 'assistant').length}
                </div>
                <div className="text-sm text-green-800">Responses</div>
              </div>

              <div className="bg-purple-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {state.files.length}
                </div>
                <div className="text-sm text-purple-800">Files</div>
              </div>

              <div className="bg-yellow-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">
                  {state.reconnectAttempts}
                </div>
                <div className="text-sm text-yellow-800">Reconnects</div>
              </div>
            </div>

            {state.performanceMetrics && (
              <div className="space-y-3">
                <h4 className="font-medium text-gray-700">Performance</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Avg Latency:</span>
                    <span className="font-medium">
                      {Math.round(state.performanceMetrics.averageLatency)}ms
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Messages/sec:</span>
                    <span className="font-medium">
                      {state.performanceMetrics.messagesPerSecond.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Tokens:</span>
                    <span className="font-medium">
                      {state.performanceMetrics.totalTokens.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case 'help':
        return (
          <div className="p-4 space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Help & Tips</h3>

            <div className="space-y-3 text-sm">
              <div>
                <h4 className="font-medium text-gray-700 mb-1">Getting Started</h4>
                <p className="text-gray-600">
                  Upload an Excel file using the file selector, then ask questions about your data.
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Sample Questions</h4>
                <ul className="text-gray-600 space-y-1 ml-4">
                  <li>• "What's the average sales by region?"</li>
                  <li>• "Show me the top 10 customers"</li>
                  <li>• "Which products have declining sales?"</li>
                  <li>• "Create a summary of Q4 data"</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Keyboard Shortcuts</h4>
                <ul className="text-gray-600 space-y-1 ml-4">
                  <li>• Enter: Send message</li>
                  <li>• Shift+Enter: New line</li>
                  <li>• Ctrl+/: Show help</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Features</h4>
                <ul className="text-gray-600 space-y-1 ml-4">
                  <li>• Real-time streaming responses</li>
                  <li>• File and sheet filtering</li>
                  <li>• Voice input support</li>
                  <li>• Auto-reconnection</li>
                </ul>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className={`flex h-screen bg-gray-100 ${className} ${theme === 'dark' ? 'dark' : ''}`}>
      {/* Sidebar */}
      {showSidebar && (
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Sidebar Header */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold text-gray-900">Excel Chat Agent</h2>
              <button
                onClick={() => setShowSidebar(false)}
                className="md:hidden p-1 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Sidebar Tabs */}
            <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setSidebarTab('files')}
                className={`flex-1 px-3 py-1 text-sm rounded-md transition-colors ${
                  sidebarTab === 'files'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Files
              </button>
              <button
                onClick={() => setSidebarTab('stats')}
                className={`flex-1 px-3 py-1 text-sm rounded-md transition-colors ${
                  sidebarTab === 'stats'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Stats
              </button>
              <button
                onClick={() => setSidebarTab('help')}
                className={`flex-1 px-3 py-1 text-sm rounded-md transition-colors ${
                  sidebarTab === 'help'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Help
              </button>
            </div>
          </div>

          {/* Sidebar Content */}
          <div className="flex-1 overflow-y-auto">
            {renderSidebarContent()}
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {!showSidebar && (
                <button
                  onClick={() => setShowSidebar(true)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <Menu className="w-5 h-5" />
                </button>
              )}

              <h1 className="text-xl font-semibold text-gray-900">
                Chat
                {state.selectedFile && (
                  <span className="text-base font-normal text-gray-600 ml-2">
                    → {state.selectedFile}
                    {state.selectedSheet && ` → ${state.selectedSheet}`}
                  </span>
                )}
              </h1>
            </div>

            <div className="flex items-center gap-2">
              {/* Connection Status */}
              <ConnectionStatus
                connectionState={state.connectionState}
                reconnectAttempts={state.reconnectAttempts}
                maxReconnectAttempts={state.maxReconnectAttempts}
                onReconnect={reconnect}
                compact
              />

              {/* Action Buttons */}
              <div className="flex items-center gap-1">
                {state.messages.length > 0 && (
                  <>
                    <button
                      onClick={exportMessages}
                      className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                      title="Export chat"
                    >
                      <Download className="w-4 h-4" />
                    </button>

                    <button
                      onClick={clearMessages}
                      className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Clear messages"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </>
                )}

                <button
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                  title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-4 h-4" />
                  ) : (
                    <Maximize2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {state.lastError && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-red-800 text-sm">{state.lastError}</span>
                <button
                  onClick={retryLastQuery}
                  className="text-red-600 hover:text-red-800 text-sm underline"
                >
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Messages */}
        <MessageList
          messages={state.messages}
          isTyping={state.isTyping}
          className="flex-1"
          autoScroll
          showTimestamps
        />

        {/* Input */}
        <MessageInput
          onSendMessage={handleSendQuery}
          disabled={state.isTyping}
          placeholder={
            state.selectedFile
              ? `Ask about ${state.selectedFile}...`
              : "Upload a file and ask questions about your data..."
          }
          showFilters
          availableFiles={state.files.map(f => f.name)}
          availableSheets={getAvailableSheets()}
          selectedFile={state.selectedFile}
          selectedSheet={state.selectedSheet}
          onFileSelect={setSelectedFile}
          onSheetSelect={setSelectedSheet}
          isConnected={state.isConnected}
        />
      </div>
    </div>
  );
};