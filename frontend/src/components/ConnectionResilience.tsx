/**
 * ConnectionResilience component for advanced connection recovery and monitoring
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketState } from '../types/chat.types';
import { NetworkErrorHandler } from './NetworkErrorHandler';
import {
  Activity,
  Shield,
  Zap,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Wifi
} from 'lucide-react';

interface ConnectionMetrics {
  connectionUptime: number;
  totalReconnections: number;
  averageLatency: number;
  lastSuccessfulConnection: Date | null;
  connectionQuality: 'poor' | 'fair' | 'good' | 'excellent';
  packetsLost: number;
  messagesSent: number;
  messagesReceived: number;
}

interface ConnectionResilienceProps {
  connectionState: WebSocketState;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  isOnline: boolean;
  lastError?: string | null;
  onRetry: () => void;
  onClearError: () => void;
  onConnectionQualityChange?: (quality: ConnectionMetrics['connectionQuality']) => void;
  children: React.ReactNode;
  enableAdvancedMonitoring?: boolean;
  enablePredictiveReconnection?: boolean;
  className?: string;
}

const initialMetrics: ConnectionMetrics = {
  connectionUptime: 0,
  totalReconnections: 0,
  averageLatency: 0,
  lastSuccessfulConnection: null,
  connectionQuality: 'poor',
  packetsLost: 0,
  messagesSent: 0,
  messagesReceived: 0
};

export const ConnectionResilience: React.FC<ConnectionResilienceProps> = ({
  connectionState,
  reconnectAttempts,
  maxReconnectAttempts,
  isOnline,
  lastError,
  onRetry,
  onClearError,
  onConnectionQualityChange,
  children,
  enableAdvancedMonitoring = true,
  enablePredictiveReconnection = true,
  className = ''
}) => {
  const [metrics, setMetrics] = useState<ConnectionMetrics>(initialMetrics);
  const [showMetrics, setShowMetrics] = useState(false);
  const [healthScore, setHealthScore] = useState(0);
  const [predictions, setPredictions] = useState<{
    nextFailureProbability: number;
    recommendedAction: string;
  } | null>(null);

  const metricsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const connectionStartTimeRef = useRef<Date | null>(null);
  const latencyHistoryRef = useRef<number[]>([]);
  const qualityHistoryRef = useRef<ConnectionMetrics['connectionQuality'][]>([]);

  // Update connection metrics
  useEffect(() => {
    if (connectionState === WebSocketState.CONNECTED) {
      if (!connectionStartTimeRef.current) {
        connectionStartTimeRef.current = new Date();
        setMetrics(prev => ({
          ...prev,
          lastSuccessfulConnection: new Date()
        }));
      }

      // Start metrics collection
      if (enableAdvancedMonitoring && !metricsIntervalRef.current) {
        metricsIntervalRef.current = setInterval(updateMetrics, 1000);
      }
    } else {
      connectionStartTimeRef.current = null;
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
        metricsIntervalRef.current = null;
      }
    }

    return () => {
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, [connectionState, enableAdvancedMonitoring]);

  // Track reconnection attempts
  useEffect(() => {
    if (reconnectAttempts > 0) {
      setMetrics(prev => ({
        ...prev,
        totalReconnections: prev.totalReconnections + 1
      }));
    }
  }, [reconnectAttempts]);

  const updateMetrics = useCallback(() => {
    if (connectionState !== WebSocketState.CONNECTED || !connectionStartTimeRef.current) {
      return;
    }

    const now = new Date();
    const uptime = now.getTime() - connectionStartTimeRef.current.getTime();

    // Simulate latency measurement (in a real app, this would come from actual ping measurements)
    const simulatedLatency = Math.random() * 100 + 20; // 20-120ms
    latencyHistoryRef.current.push(simulatedLatency);
    if (latencyHistoryRef.current.length > 100) {
      latencyHistoryRef.current = latencyHistoryRef.current.slice(-100);
    }

    const averageLatency = latencyHistoryRef.current.reduce((a, b) => a + b, 0) / latencyHistoryRef.current.length;

    // Determine connection quality
    let quality: ConnectionMetrics['connectionQuality'] = 'poor';
    if (averageLatency < 50 && isOnline) {
      quality = 'excellent';
    } else if (averageLatency < 100 && isOnline) {
      quality = 'good';
    } else if (averageLatency < 200 && isOnline) {
      quality = 'fair';
    }

    qualityHistoryRef.current.push(quality);
    if (qualityHistoryRef.current.length > 60) {
      qualityHistoryRef.current = qualityHistoryRef.current.slice(-60);
    }

    setMetrics(prev => {
      const newMetrics = {
        ...prev,
        connectionUptime: uptime,
        averageLatency,
        connectionQuality: quality
      };

      // Notify about quality changes
      if (prev.connectionQuality !== quality) {
        onConnectionQualityChange?.(quality);
      }

      return newMetrics;
    });

    // Update health score
    updateHealthScore(quality, averageLatency, uptime);

    // Update predictions if enabled
    if (enablePredictiveReconnection) {
      updatePredictions(quality, averageLatency, reconnectAttempts);
    }
  }, [connectionState, isOnline, reconnectAttempts, onConnectionQualityChange, enablePredictiveReconnection]);

  const updateHealthScore = useCallback((
    quality: ConnectionMetrics['connectionQuality'],
    latency: number,
    uptime: number
  ) => {
    let score = 0;

    // Quality score (0-40 points)
    switch (quality) {
      case 'excellent': score += 40; break;
      case 'good': score += 30; break;
      case 'fair': score += 20; break;
      case 'poor': score += 10; break;
    }

    // Latency score (0-30 points)
    if (latency < 50) score += 30;
    else if (latency < 100) score += 20;
    else if (latency < 200) score += 10;

    // Uptime score (0-20 points)
    const uptimeMinutes = uptime / (1000 * 60);
    if (uptimeMinutes > 60) score += 20;
    else if (uptimeMinutes > 30) score += 15;
    else if (uptimeMinutes > 10) score += 10;
    else if (uptimeMinutes > 1) score += 5;

    // Stability score (0-10 points)
    if (reconnectAttempts === 0) score += 10;
    else if (reconnectAttempts < 3) score += 5;

    setHealthScore(Math.min(score, 100));
  }, [reconnectAttempts]);

  const updatePredictions = useCallback((
    quality: ConnectionMetrics['connectionQuality'],
    latency: number,
    attempts: number
  ) => {
    // Simple prediction algorithm (in production, this would be more sophisticated)
    let failureProbability = 0;
    let action = 'Connection is stable';

    if (quality === 'poor' || latency > 200) {
      failureProbability = 0.7;
      action = 'Consider switching networks or refreshing connection';
    } else if (quality === 'fair' || latency > 100) {
      failureProbability = 0.3;
      action = 'Monitor connection quality';
    } else if (attempts > 2) {
      failureProbability = 0.5;
      action = 'Unstable connection detected, consider troubleshooting';
    }

    setPredictions({
      nextFailureProbability: failureProbability,
      recommendedAction: action
    });
  }, []);

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-50';
    if (score >= 60) return 'text-yellow-600 bg-yellow-50';
    if (score >= 40) return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  const getQualityIcon = (quality: ConnectionMetrics['connectionQuality']) => {
    switch (quality) {
      case 'excellent': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'good': return <TrendingUp className="w-4 h-4 text-blue-600" />;
      case 'fair': return <Activity className="w-4 h-4 text-yellow-600" />;
      case 'poor': return <AlertTriangle className="w-4 h-4 text-red-600" />;
    }
  };

  const formatUptime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className={`relative ${className}`}>
      {/* Network Error Handler */}
      {lastError && (
        <div className="mb-4">
          <NetworkErrorHandler
            connectionState={connectionState}
            lastError={lastError}
            onRetry={onRetry}
            onClearError={onClearError}
            autoRetry={enablePredictiveReconnection}
          />
        </div>
      )}

      {/* Connection Metrics Panel */}
      {enableAdvancedMonitoring && (
        <div className="mb-4">
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
          >
            <Activity className="w-4 h-4" />
            <span>Connection Health: {healthScore}/100</span>
            <div className={`w-2 h-2 rounded-full ${healthScore >= 80 ? 'bg-green-500' : healthScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'}`} />
          </button>

          {showMetrics && (
            <div className="mt-2 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1 mb-1">
                    {getQualityIcon(metrics.connectionQuality)}
                    <span className="text-xs font-medium text-gray-700">Quality</span>
                  </div>
                  <div className="text-sm capitalize">{metrics.connectionQuality}</div>
                </div>

                <div className="text-center">
                  <div className="flex items-center justify-center gap-1 mb-1">
                    <Zap className="w-4 h-4 text-blue-600" />
                    <span className="text-xs font-medium text-gray-700">Latency</span>
                  </div>
                  <div className="text-sm">{Math.round(metrics.averageLatency)}ms</div>
                </div>

                <div className="text-center">
                  <div className="flex items-center justify-center gap-1 mb-1">
                    <Clock className="w-4 h-4 text-purple-600" />
                    <span className="text-xs font-medium text-gray-700">Uptime</span>
                  </div>
                  <div className="text-sm">{formatUptime(metrics.connectionUptime)}</div>
                </div>

                <div className="text-center">
                  <div className="flex items-center justify-center gap-1 mb-1">
                    <Shield className="w-4 h-4 text-orange-600" />
                    <span className="text-xs font-medium text-gray-700">Reconnects</span>
                  </div>
                  <div className="text-sm">{metrics.totalReconnections}</div>
                </div>
              </div>

              {/* Predictions */}
              {enablePredictiveReconnection && predictions && (
                <div className="border-t border-gray-200 pt-3">
                  <h4 className="text-xs font-medium text-gray-700 mb-2">Connection Analysis</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">Failure Risk:</span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              predictions.nextFailureProbability > 0.6 ? 'bg-red-500' :
                              predictions.nextFailureProbability > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${predictions.nextFailureProbability * 100}%` }}
                          />
                        </div>
                        <span className="font-medium">
                          {Math.round(predictions.nextFailureProbability * 100)}%
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-600">
                      <strong>Recommendation:</strong> {predictions.recommendedAction}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Main Content */}
      {children}
    </div>
  );
};