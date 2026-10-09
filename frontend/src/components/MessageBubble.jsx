import React from 'react';
import Markdown from 'react-markdown';
import { CircleCheck } from 'lucide-react';
import { Logo } from './Logo';
import { EvidenceCard } from './EvidenceCard';
import { ActionButton } from './ActionButton';

/**
 * MessageBubble — one entry in the conversation.
 */
export function MessageBubble({ message, onAction, isProcessing }) {
    if (message.role === 'user') {
        return (
            <div className="flex justify-end animate-in">
                <div className="max-w-[75%] rounded-2xl rounded-br-md bg-brand-100 px-4 py-2.5 text-[15px] leading-relaxed text-ink">
                    {message.content}
                </div>
            </div>
        );
    }

    if (message.role === 'system') {
        return (
            <div className="flex justify-center animate-in">
                <div className="flex max-w-[85%] items-start gap-2 rounded-full border border-line bg-surface px-3.5 py-1.5 text-xs text-ink-2">
                    <CircleCheck className="w-3.5 h-3.5 mt-px shrink-0 text-emerald-600" />
                    <span className="whitespace-pre-wrap">{message.content}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="flex gap-3 animate-in">
            <Logo size={30} className="mt-1 shrink-0" />
            <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-line bg-surface px-4 py-3 shadow-[0_1px_2px_rgba(28,25,23,0.04)]">
                <div className="prose-chat text-[15px] leading-relaxed text-ink-2">
                    <Markdown>{message.content}</Markdown>
                </div>

                {message.evidence && <EvidenceCard evidence={message.evidence} />}

                {message.suggestions?.length > 0 && (
                    <div className="mt-4 space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-wider text-ink-3">Suggested next steps</p>
                        {message.suggestions.map((action, j) => (
                            <ActionButton
                                key={j}
                                action={action}
                                onExecute={onAction}
                                isProcessing={isProcessing}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
