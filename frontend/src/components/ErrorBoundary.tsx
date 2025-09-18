/**
 * Error Boundary component for catching and handling React errors gracefully
 */

import React, { Component, ReactNode, ErrorInfo } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug, Mail } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
  isRetrying: boolean;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  maxRetries?: number;
  showDetails?: boolean;
  enableReporting?: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private retryTimeoutRef: NodeJS.Timeout | null = null;

  constructor(props: ErrorBoundaryProps) {
    super(props);

    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRetrying: false
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Log error to console
    console.error('Error Boundary caught an error:', error);
    console.error('Error Info:', errorInfo);

    // Call custom error handler
    this.props.onError?.(error, errorInfo);

    // Report to error tracking service if enabled
    if (this.props.enableReporting) {
      this.reportError(error, errorInfo);
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutRef) {
      clearTimeout(this.retryTimeoutRef);
    }
  }

  reportError = (error: Error, errorInfo: ErrorInfo) => {
    // This would integrate with error reporting services like Sentry, LogRocket, etc.
    console.log('Reporting error:', {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
  };

  handleRetry = () => {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;

    if (retryCount >= maxRetries) {
      return;
    }

    this.setState({
      isRetrying: true
    });

    // Add delay before retry to prevent rapid retries
    this.retryTimeoutRef = setTimeout(() => {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: retryCount + 1,
        isRetrying: false
      });
    }, 1000);
  };

  handleReload = () => {
    window.location.reload();
  };

  handleReportBug = () => {
    const { error, errorInfo } = this.state;
    const errorReport = {
      error: error?.message || 'Unknown error',
      stack: error?.stack,
      componentStack: errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    const mailBody = encodeURIComponent(`
Error Report:
${JSON.stringify(errorReport, null, 2)}

Please describe what you were doing when this error occurred:
[Your description here]
    `);

    window.open(`mailto:support@example.com?subject=Error Report&body=${mailBody}`);
  };

  getErrorSeverity = (error: Error): 'low' | 'medium' | 'high' => {
    const message = error.message.toLowerCase();

    if (message.includes('network') || message.includes('fetch') || message.includes('websocket')) {
      return 'medium';
    }

    if (message.includes('chunk') || message.includes('loading')) {
      return 'low';
    }

    return 'high';
  };

  getErrorSuggestion = (error: Error): string => {
    const message = error.message.toLowerCase();

    if (message.includes('network') || message.includes('websocket')) {
      return 'Check your internet connection and try again.';
    }

    if (message.includes('chunk') || message.includes('loading')) {
      return 'There may be a temporary loading issue. Please try refreshing the page.';
    }

    if (message.includes('permission') || message.includes('unauthorized')) {
      return 'You may not have permission to perform this action.';
    }

    return 'An unexpected error occurred. Please try refreshing the page or contact support.';
  };

  render() {
    const { hasError, error, errorInfo, retryCount, isRetrying } = this.state;
    const { children, fallback, maxRetries = 3, showDetails = false } = this.props;

    if (hasError) {
      if (fallback) {
        return fallback;
      }

      const severity = error ? this.getErrorSeverity(error) : 'high';
      const suggestion = error ? this.getErrorSuggestion(error) : '';
      const canRetry = retryCount < maxRetries;

      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
            {/* Error Icon */}
            <div className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center ${
              severity === 'high' ? 'bg-red-100' :
              severity === 'medium' ? 'bg-yellow-100' : 'bg-blue-100'
            }`}>
              <AlertTriangle className={`w-8 h-8 ${
                severity === 'high' ? 'text-red-600' :
                severity === 'medium' ? 'text-yellow-600' : 'text-blue-600'
              }`} />
            </div>

            {/* Error Message */}
            <div className="text-center mb-6">
              <h1 className="text-xl font-semibold text-gray-900 mb-2">
                Something went wrong
              </h1>
              <p className="text-gray-600 text-sm mb-4">
                {suggestion}
              </p>

              {error && (
                <div className="text-left bg-gray-50 rounded-lg p-3 mb-4">
                  <p className="text-sm text-gray-800 font-medium mb-1">
                    Error Details:
                  </p>
                  <p className="text-xs text-gray-600 font-mono break-all">
                    {error.message}
                  </p>
                </div>
              )}

              {retryCount > 0 && (
                <p className="text-xs text-gray-500 mb-4">
                  Retry attempts: {retryCount}/{maxRetries}
                </p>
              )}
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              {canRetry && (
                <button
                  onClick={this.handleRetry}
                  disabled={isRetrying}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {isRetrying ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Retrying...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      <span>Try Again</span>
                    </>
                  )}
                </button>
              )}

              <button
                onClick={this.handleReload}
                className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors flex items-center justify-center gap-2"
              >
                <Home className="w-4 h-4" />
                <span>Reload Page</span>
              </button>

              <button
                onClick={this.handleReportBug}
                className="w-full bg-orange-600 text-white py-2 px-4 rounded-lg hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 transition-colors flex items-center justify-center gap-2"
              >
                <Bug className="w-4 h-4" />
                <span>Report Issue</span>
              </button>
            </div>

            {/* Technical Details */}
            {showDetails && error && errorInfo && (
              <details className="mt-6">
                <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800">
                  Technical Details
                </summary>
                <div className="mt-3 p-3 bg-gray-50 rounded border text-xs">
                  <div className="mb-3">
                    <h4 className="font-medium text-gray-800 mb-1">Error Stack:</h4>
                    <pre className="text-gray-600 whitespace-pre-wrap break-all">
                      {error.stack}
                    </pre>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-800 mb-1">Component Stack:</h4>
                    <pre className="text-gray-600 whitespace-pre-wrap">
                      {errorInfo.componentStack}
                    </pre>
                  </div>
                </div>
              </details>
            )}

            {/* Footer */}
            <div className="mt-6 text-center">
              <p className="text-xs text-gray-500">
                If this problem persists, please contact support
              </p>
              <div className="mt-2">
                <button
                  onClick={() => window.open('mailto:support@example.com')}
                  className="text-xs text-blue-600 hover:text-blue-800 underline inline-flex items-center gap-1"
                >
                  <Mail className="w-3 h-3" />
                  <span>Contact Support</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}