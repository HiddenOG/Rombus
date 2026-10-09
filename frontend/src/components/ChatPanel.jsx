import React, { useState } from 'react';
import { Loader2, ArrowUp, AlertCircle, X, Lock, LineChart, Wand2 } from 'lucide-react';
import { Logo } from './Logo';
import { StageProgress } from './StageProgress';
import { MessageBubble } from './MessageBubble';

/**
 * ChatPanel — conversation with the copilot, plus the input box.
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
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
            {/* Header */}
            <header className="shrink-0 h-16 px-6 flex items-center justify-between gap-4 border-b border-line bg-surface/70 backdrop-blur">
                <div className="flex items-center gap-2.5 text-sm">
                    <span className="relative flex h-2.5 w-2.5">
                        {isProcessing && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-60" />}
                        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${isProcessing ? 'bg-brand-500' : 'bg-emerald-500'}`} />
                    </span>
                    <span className="font-semibold text-ink">Copilot</span>
                    <span className="text-ink-3">{isProcessing ? 'thinking…' : 'ready'}</span>
                </div>
                <StageProgress currentStage={currentStage} />
            </header>

            {/* Conversation */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                <div className="mx-auto max-w-3xl px-6 py-8 space-y-6">
                    {chatHistory.length === 0 && <EmptyState />}
                    {chatHistory.map((msg, i) => (
                        <MessageBubble key={i} message={msg} onAction={onAction} isProcessing={isProcessing} />
                    ))}
                    {isProcessing && chatHistory.length > 0 && <ThinkingIndicator />}
                    <div ref={chatEndRef} />
                </div>
            </div>

            {/* Input */}
            <div className="shrink-0 px-6 pb-5 pt-2">
                <div className="mx-auto max-w-3xl">
                    {error && (
                        <div className="mb-3 flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800 animate-in">
                            <AlertCircle className="w-4 h-4 shrink-0" />
                            <span className="flex-1">{error}</span>
                            <button onClick={onClearError} aria-label="Dismiss error" className="rounded-full p-1 hover:bg-red-100">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    )}
                    <form
                        onSubmit={handleSubmit}
                        className="flex items-center gap-2 rounded-2xl border border-line bg-surface p-2 pl-4 shadow-[0_2px_12px_rgba(28,25,23,0.06)] transition-shadow focus-within:border-brand-300 focus-within:ring-4 focus-within:ring-brand-100"
                    >
                        <input
                            type="text"
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            placeholder={sessionId ? 'Ask about your data or the next step…' : 'Upload a dataset to start'}
                            disabled={!sessionId || isProcessing}
                            className="flex-1 bg-transparent py-2 text-[15px] text-ink outline-none placeholder:text-ink-3 disabled:cursor-not-allowed"
                        />
                        <button
                            type="submit"
                            aria-label="Send"
                            disabled={!sessionId || isProcessing || !inputMessage.trim()}
                            className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500 text-white transition-colors hover:bg-brand-600 disabled:bg-surface-2 disabled:text-ink-3 disabled:cursor-not-allowed"
                        >
                            {isProcessing ? <Loader2 className="w-5 h-5 animate-spin" /> : <ArrowUp className="w-5 h-5" />}
                        </button>
                    </form>
                    <p className="mt-2 text-center text-xs text-ink-3">
                        The copilot only suggests. Nothing changes until you press Run.
                    </p>
                </div>
            </div>
        </main>
    );
}

function ThinkingIndicator() {
    return (
        <div className="flex items-center gap-3 pl-11 text-sm text-ink-3">
            <span className="flex gap-1">
                {[0, 150, 300].map(delay => (
                    <span
                        key={delay}
                        className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-400"
                        style={{ animationDelay: `${delay}ms` }}
                    />
                ))}
            </span>
            Thinking…
        </div>
    );
}

function EmptyState() {
    const points = [
        { icon: Lock, title: 'Honest scores', text: 'A test set is locked away before you see any statistics.' },
        { icon: Wand2, title: 'You stay in control', text: 'The AI suggests each step; you decide what runs.' },
        { icon: LineChart, title: 'Evidence first', text: 'Every suggestion is backed by your data’s numbers.' },
    ];

    return (
        <div className="pt-12 text-center">
            <Logo size={72} className="mx-auto mb-6 drop-shadow-md" />
            <h2 className="text-3xl font-bold tracking-tight text-ink">Your data science copilot</h2>
            <p className="mx-auto mt-3 max-w-md text-[15px] leading-relaxed text-ink-2">
                Upload a spreadsheet and build a prediction model step by step, with an AI that explains
                every move.
            </p>
            <div className="mt-10 grid gap-3 text-left sm:grid-cols-3">
                {points.map(({ icon: Icon, title, text }) => (
                    <div key={title} className="rounded-2xl border border-line bg-surface p-4">
                        <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
                            <Icon className="h-4 w-4 text-brand-600" />
                        </div>
                        <p className="text-sm font-semibold text-ink">{title}</p>
                        <p className="mt-1 text-xs leading-relaxed text-ink-3">{text}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
