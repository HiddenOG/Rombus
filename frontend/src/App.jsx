import React from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { useSession } from './hooks/useSession';

/**
 * App — Slim orchestrator composing Sidebar and ChatPanel.
 */
const App = () => {
  const {
    sessionId,
    currentStage,
    summary,
    chatHistory,
    isProcessing,
    error,
    chatEndRef,
    handleUpload,
    handleChat,
    handleAction,
    clearError,
  } = useSession();

  return (
    <ErrorBoundary>
      <div className="flex h-screen overflow-hidden bg-gray-50 font-sans text-gray-900 antialiased">
        <Sidebar
          sessionId={sessionId}
          currentStage={currentStage}
          summary={summary}
          onFileUpload={handleUpload}
          isProcessing={isProcessing}
        />
        <ChatPanel
          chatHistory={chatHistory}
          currentStage={currentStage}
          sessionId={sessionId}
          isProcessing={isProcessing}
          error={error}
          chatEndRef={chatEndRef}
          onChat={handleChat}
          onAction={handleAction}
          onClearError={clearError}
        />
      </div>
    </ErrorBoundary>
  );
};

export default App;
