import React from 'react';
import { Play, ArrowRight, ShieldCheck } from 'lucide-react';

const humanize = (name) => name.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

function formatValue(value) {
    if (Array.isArray(value)) return value.join(', ');
    if (value && typeof value === 'object') {
        return Object.entries(value).map(([k, v]) => `${k}=${v}`).join(', ');
    }
    return String(value);
}

/**
 * ActionButton — a suggested action as a card: what runs, with which settings, and why.
 */
export function ActionButton({ action, onExecute, isProcessing }) {
    const isTransition = action.tool === 'transition_stage';
    const Icon = isTransition ? ArrowRight : Play;
    const args = Object.entries(action.arguments || {});

    return (
        <div className="flex items-start gap-3 rounded-xl border border-line bg-surface p-3 transition-colors hover:border-brand-200">
            <div className="min-w-0 flex-1">
                <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-ink">
                    {humanize(action.tool)}
                    {action.verified && (
                        <span
                            title="A built-in step: tested, replayed exactly on the test rows and in the exported script"
                            className="flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-800"
                        >
                            <ShieldCheck className="h-3 w-3" /> Verified step
                        </span>
                    )}
                </p>
                {args.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {args.map(([key, value]) => (
                            <span
                                key={key}
                                className="rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2"
                            >
                                {humanize(key)}: <span className="text-ink">{formatValue(value)}</span>
                            </span>
                        ))}
                    </div>
                )}
                {action.explanation && (
                    <p className="mt-1.5 text-xs leading-relaxed text-ink-3">{action.explanation}</p>
                )}
            </div>
            <button
                onClick={() => onExecute(action)}
                disabled={isProcessing}
                className="shrink-0 flex items-center gap-1.5 rounded-lg bg-brand-500 px-3 py-1.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
                <Icon className="w-3.5 h-3.5" />
                {isTransition ? 'Next' : 'Run'}
            </button>
        </div>
    );
}
