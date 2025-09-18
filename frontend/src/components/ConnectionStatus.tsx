/**
 * ConnectionStatus component for displaying WebSocket connection state and reconnection info
 */

import React from 'react';
import { WebSocketState } from '../types/chat.types';
import {
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle,
  Loader2,
  RefreshCw,
  Clock,
  Activity
} from 'lucide-react';

interface ConnectionStatusProps {
  connectionState: WebSocketState;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  nextReconnectDelay?: number;
  networkStatus?: {
    isOnline: boolean;
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  };
  onReconnect?: () => void;
  className?: string;
  compact?: boolean;
  showDetails?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  connectionState,
  reconnectAttempts,
  maxReconnectAttempts,
  nextReconnectDelay,
  networkStatus,
  onReconnect,
  className = '',
  compact = false,
  showDetails = false
}) => {
  const getStatusInfo = () => {
    switch (connectionState) {
      case WebSocketState.CONNECTED:
        return {
          icon: CheckCircle,
          text: 'Connected',
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200'
        };

      case WebSocketState.CONNECTING:
        return {
          icon: Loader2,
          text: 'Connecting...',
          color: 'text-blue-600',
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          animate: 'animate-spin'
        };

      case WebSocketState.RECONNECTING:
        return {
          icon: RefreshCw,
          text: `Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          animate: 'animate-spin'
        };

      case WebSocketState.DISCONNECTING:
        return {
          icon: Loader2,
          text: 'Disconnecting...',
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          animate: 'animate-spin'
        };

      case WebSocketState.ERROR:
        return {
          icon: AlertCircle,
          text: 'Connection Error',
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200'
        };

      case WebSocketState.DISCONNECTED:
      default:
        return {
          icon: WifiOff,
          text: 'Disconnected',
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200'
        };
    }
  };

  const statusInfo = getStatusInfo();
  const StatusIcon = statusInfo.icon;
  const isConnected = connectionState === WebSocketState.CONNECTED;
  const canReconnect = connectionState === WebSocketState.DISCONNECTED || connectionState === WebSocketState.ERROR;

  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <StatusIcon
          className={`w-4 h-4 ${statusInfo.color} ${statusInfo.animate || ''}`}
        />
        <span className={`text-sm ${statusInfo.color}`}>
          {statusInfo.text}
        </span>
        {canReconnect && onReconnect && (
          <button
            onClick={onReconnect}
            className="text-xs text-blue-600 hover:text-blue-800 underline focus:outline-none"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`p-3 rounded-lg border ${statusInfo.bgColor} ${statusInfo.borderColor} ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StatusIcon
            className={`w-5 h-5 ${statusInfo.color} ${statusInfo.animate || ''}`}
          />
          <div>
            <div className={`font-medium ${statusInfo.color}`}>
              {statusInfo.text}
            </div>

            {showDetails && (
              <div className="text-xs text-gray-600 mt-1">
                {connectionState === WebSocketState.RECONNECTING && nextReconnectDelay && (
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>Next attempt in {Math.ceil(nextReconnectDelay / 1000)}s</span>
                  </div>
                )}

                {connectionState === WebSocketState.ERROR && reconnectAttempts >= maxReconnectAttempts && (
                  <div className="text-red-600">
                    Max reconnection attempts reached
                  </div>
                )}

                {networkStatus && (
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex items-center gap-1">
                      {networkStatus.isOnline ? (
                        <Wifi className="w-3 h-3 text-green-500" />
                      ) : (
                        <WifiOff className="w-3 h-3 text-red-500" />
                      )}
                      <span>{networkStatus.isOnline ? 'Online' : 'Offline'}</span>
                    </div>

                    {networkStatus.effectiveType && (
                      <div className="flex items-center gap-1">
                        <Activity className="w-3 h-3" />
                        <span>{networkStatus.effectiveType}</span>
                      </div>
                    )}

                    {networkStatus.rtt && (
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{networkStatus.rtt}ms</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {canReconnect && onReconnect && (
          <button
            onClick={onReconnect}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
          >
            Reconnect
          </button>
        )}
      </div>

      {/* Connection Quality Indicator */}
      {isConnected && networkStatus && (
        <div className="mt-3 pt-3 border-t border-green-200">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <span>Connection Quality</span>
            <div className="flex items-center gap-1">
              {networkStatus.downlink && networkStatus.downlink > 1 ? (
                <>
                  <div className="w-2 h-3 bg-green-400 rounded"></div>
                  <div className="w-2 h-4 bg-green-500 rounded"></div>
                  <div className="w-2 h-5 bg-green-600 rounded"></div>
                  <span className="ml-1 text-green-600">Excellent</span>
                </>
              ) : networkStatus.downlink && networkStatus.downlink > 0.5 ? (
                <>
                  <div className="w-2 h-3 bg-yellow-400 rounded"></div>
                  <div className="w-2 h-4 bg-yellow-500 rounded"></div>
                  <div className="w-2 h-5 bg-gray-300 rounded"></div>
                  <span className="ml-1 text-yellow-600">Good</span>
                </>
              ) : (
                <>
                  <div className="w-2 h-3 bg-red-400 rounded"></div>
                  <div className="w-2 h-4 bg-gray-300 rounded"></div>
                  <div className="w-2 h-5 bg-gray-300 rounded"></div>
                  <span className="ml-1 text-red-600">Poor</span>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Reconnection Progress */}
      {connectionState === WebSocketState.RECONNECTING && (
        <div className="mt-3 pt-3 border-t border-yellow-200">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-2">
            <span>Reconnection Progress</span>
            <span>{reconnectAttempts}/{maxReconnectAttempts}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
              style={{
                width: `${(reconnectAttempts / maxReconnectAttempts) * 100}%`
              }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};