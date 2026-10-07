import React, { useState } from 'react';
import { Brain, Loader2, Send, AlertCircle, X } from 'lucide-react';
import { StageProgress } from './StageProgress';
import { MessageBubble } from './MessageBubble';

/**
 * ChatPanel — main chat area with messages and input.
 */
export function ChatPanel({
    chatHistory,
    currentStage,
    sessionId,
    isProcessing,
    error,
    chatEndRef,
    onChat,
    onAction,
    onClearError,
}) {
    const [inputMessage, setInputMessage] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!inputMessage.trim() || isProcessing) return;
        const msg = inputMessage;
        setInputMessage('');
        onChat(msg);
    };

    return (
        <div className="flex-1 flex flex-col overflow-hidden bg-white relative min-w-0">
            {/* Header — fixed height, never scrolls */}
            <div className="shrink-0 h-14 border-b border-gray-200 flex items-center justify-between px-6 bg-white z-10">
                <div className="flex items-center gap-3">
                    <div
                        className={`w-2.5 h-2.5 rounded-full transition-colors ${isProcessing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'
                            }`}
                    />
                    <span className="font-semibold text-gray-700 text-base">AI Copilot</span>
                </div>
                <StageProgress currentStage={currentStage} />
            </div>

            {/* Messages — THIS is the scrollable area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-5 bg-gradient-to-b from-gray-50/50 to-white custom-scrollbar">
                {chatHistory.length === 0 && <EmptyState />}

                {chatHistory.map((msg, i) => (
                    <MessageBubble
                        key={i}
                        message={msg}
                        onAction={onAction}
                        isProcessing={isProcessing}
                    />
                ))}
                <div ref={chatEndRef} />
            </div>

            {/* Error display */}
            {error && (
                <div className="shrink-0 mx-6 mb-2 p-3 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 text-red-700 text-sm animate-in">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span className="flex-1">{error}</span>
                    <button
                        onClick={onClearError}
                        className="hover:bg-red-100 rounded-full p-1 transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            )}

            {/* Input Area — fixed at bottom, never scrolls */}
            <div className="shrink-0 p-4 bg-white border-t border-gray-200">
                <form onSubmit={handleSubmit} className="relative">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder={
                            sessionId
                                ? 'Ask about the data or discuss the next step...'
                                : 'Upload a dataset to start'
                        }
                        disabled={!sessionId || isProcessing}
                        className="w-full pl-5 pr-14 py-3.5 bg-gray-100 border-none rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all outline-none text-base placeholder:text-gray-400"
                    />
                    <button
                        type="submit"
                        disabled={!sessionId || isProcessing || !inputMessage.trim()}
                        className="absolute right-1.5 top-1.5 bottom-1.5 px-4 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                        {isProcessing ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Send className="w-5 h-5" />
                        )}
                    </button>
                </form>
                <div className="flex justify-center mt-2">
                    <p className="text-xs text-gray-400 tracking-wide">
                        ASEMIC Workflow • Statistical Evidence • Human-in-the-Loop
                    </p>
                </div>
            </div>
        </div>
    );
}

function EmptyState() {
    return (
        <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto py-20">
            <div className="w-24 h-24 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-indigo-50">
                <Brain className="w-12 h-12 text-indigo-600" />
            </div>
            <h2 className="text-3xl font-bold text-gray-800 mb-3">AutoML Copilot</h2>
            <p className="text-gray-500 text-base leading-relaxed">
                AI-powered data science assistant. Upload a dataset and I'll guide you through
                the ASEMIC workflow — from data acquisition to model communication.
            </p>
            <div className="flex flex-wrap gap-2.5 mt-6 justify-center">
                {['Upload CSV', 'Clean Data', 'Explore Patterns', 'Train Models', 'Get Predictions'].map(step => (
                    <span key={step} className="text-xs bg-gray-100 text-gray-500 px-3 py-1.5 rounded-full font-medium">
                        {step}
                    </span>
                ))}
            </div>
        </div>
    );
}
