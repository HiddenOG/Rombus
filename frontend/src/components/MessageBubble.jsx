import React from 'react';
import { Sparkles } from 'lucide-react';
import { EvidenceCard } from './EvidenceCard';
import { ActionButton } from './ActionButton';

/**
 * MessageBubble — individual chat message display.
 */
export function MessageBubble({ message, onAction, isProcessing }) {
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';
    const isAI = message.role === 'ai';

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div
                className={`max-w-[80%] p-4 ${isUser
                    ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-2xl rounded-tr-sm shadow-lg shadow-indigo-100'
                    : isSystem
                        ? 'bg-gray-100 text-gray-600 rounded-lg text-sm italic px-4 py-3'
                        : 'bg-white border border-gray-200/80 shadow-sm rounded-2xl rounded-tl-sm'
                    }`}
            >
                {/* AI role label */}
                {isAI && (
                    <div className="flex items-center gap-2 mb-2.5">
                        <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center">
                            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                        </div>
                        <span className="text-xs font-bold uppercase tracking-widest text-indigo-500">
                            Copilot
                        </span>
                    </div>
                )}

                {/* Message content */}
                <div className="whitespace-pre-wrap text-base leading-relaxed">
                    {message.content}
                </div>

                {/* Evidence rendering */}
                {message.evidence && <EvidenceCard evidence={message.evidence} />}

                {/* Suggested actions */}
                {message.suggestions?.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-gray-100">
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2.5">
                            Suggested Actions
                        </p>
                        <div className="flex flex-wrap gap-2">
                            {message.suggestions.map((action, j) => (
                                <ActionButton
                                    key={j}
                                    action={action}
                                    onExecute={onAction}
                                    isProcessing={isProcessing}
                                />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
