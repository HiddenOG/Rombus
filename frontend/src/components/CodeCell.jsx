import React from 'react';
import { Play, Sparkles, AlertTriangle, Lock } from 'lucide-react';
import { CODE_KINDS, CODE_TEMPLATES } from '../lib/codeCells';
import { PythonEditor } from './PythonEditor';

/**
 * CodeCell — an unrun code cell, like Jupyter's: write (or read what the assistant wrote), pick its
 * kind, and run it. Once it runs it becomes a result cell in the same place, code still editable.
 */
export function CodeCell({ cell, isProcessing, canRun, onChange, onRun }) {
    const source = cell.source ?? '';

    const setKind = (kind) => {
        // An untouched starting template follows the kind; real code is kept as it is
        const untouched = !source.trim() || source === CODE_TEMPLATES[cell.kind];
        onChange(cell, { kind, ...(untouched ? { source: CODE_TEMPLATES[kind] } : {}), error: null });
    };

    return (
        <div className="flex gap-3 animate-in">
            <div className="hidden w-10 shrink-0 pt-3 text-right font-mono text-[11px] text-ink-3 sm:block">[ ]</div>
            <div className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-line bg-surface focus-within:border-brand-300">
                <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2">
                    <div role="radiogroup" aria-label="Kind of cell" className="flex rounded-lg bg-surface-2 p-0.5">
                        {Object.entries(CODE_KINDS).map(([kind, { label, hint }]) => (
                            <button
                                key={kind}
                                type="button"
                                role="radio"
                                aria-checked={cell.kind === kind}
                                title={hint}
                                onClick={() => setKind(kind)}
                                className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                                    cell.kind === kind ? 'bg-surface text-ink shadow-sm' : 'text-ink-3 hover:text-ink-2'
                                }`}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                    {cell.byAi && (
                        <span className="flex items-center gap-1 rounded-full border border-brand-200 bg-brand-50 px-2 py-0.5 text-[10px] font-semibold text-brand-800">
                            <Sparkles className="h-3 w-3" /> AI-written · not run yet
                        </span>
                    )}
                    <span className="flex-1" />
                    <button
                        type="button"
                        onClick={() => onRun(cell)}
                        disabled={isProcessing || !canRun || !source.trim()}
                        className="flex items-center gap-1.5 rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-brand-600 disabled:opacity-50"
                    >
                        <Play className="h-3.5 w-3.5" /> Run
                    </button>
                </div>

                <PythonEditor
                    value={source}
                    onChange={code => onChange(cell, { source: code, error: null })}
                    onRun={() => { if (!isProcessing && canRun) onRun(cell); }}
                />
                <div className="border-t border-line" />

                {cell.problem && !cell.error && (
                    <Notice tone="amber" icon={AlertTriangle}>A trial run of this code found {cell.problem}. Fix it before running, or ask the assistant to.</Notice>
                )}
                {cell.error && <Notice tone="red" icon={AlertTriangle}>{cell.error}</Notice>}
                {!canRun && (
                    <Notice tone="gray" icon={Lock}>
                        Running your own code is switched off on this server (it needs a sandbox). It runs on your own
                        machine: see the README.
                    </Notice>
                )}

                <p className="border-t border-line bg-surface-2/50 px-4 py-1.5 text-[11px] text-ink-3">
                    {CODE_KINDS[cell.kind]?.hint}. <kbd className="font-mono">Shift</kbd>+<kbd className="font-mono">Enter</kbd> runs.
                </p>
            </div>
        </div>
    );
}

function Notice({ tone, icon: Icon, children }) {
    const tones = {
        red: 'border-red-200 bg-red-50 text-red-800',
        amber: 'border-amber-200 bg-amber-50 text-amber-900',
        gray: 'border-line bg-surface-2 text-ink-2',
    };
    return (
        <div className={`mx-3 mb-3 flex gap-2 rounded-lg border px-3 py-2 text-xs leading-relaxed ${tones[tone]}`}>
            <Icon className="mt-px h-3.5 w-3.5 shrink-0" />
            <span className="whitespace-pre-wrap">{children}</span>
        </div>
    );
}
