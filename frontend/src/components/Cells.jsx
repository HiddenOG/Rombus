import React, { useState } from 'react';
import Markdown from 'react-markdown';
import { Pencil, Copy, Trash2, RotateCw, Info, StickyNote, AlertTriangle, Loader2, Ban } from 'lucide-react';
import { Logo } from './Logo';
import { ActionButton } from './ActionButton';
import { EvidenceCard } from './EvidenceCard';
import { StepEditor } from './StepEditor';

const humanize = (s) => s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

const KIND = {
    learn: { label: 'Changes data · learns', className: 'bg-brand-50 text-brand-700 border-brand-200' },
    fixed: { label: 'Changes data', className: 'bg-brand-50 text-brand-700 border-brand-200' },
    train_only: { label: 'Training rows only', className: 'bg-amber-50 text-amber-800 border-amber-200' },
    action: { label: 'Action', className: 'bg-surface-2 text-ink-2 border-line' },
    view: { label: 'View', className: 'bg-surface-2 text-ink-2 border-line' },
};

/** Gutter with the cell number, like Jupyter's [3] */
function Gutter({ number }) {
    return (
        <div className="w-10 shrink-0 pt-3 text-right font-mono text-[11px] text-ink-3">
            {number != null ? `[${number}]` : ''}
        </div>
    );
}

export function NoteCell({ cell }) {
    return (
        <div className="flex gap-3 animate-in">
            <Gutter />
            <div className="flex flex-1 items-start gap-2 rounded-xl px-3 py-2 text-sm text-ink-2">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-3" />
                <div className="prose-chat"><Markdown>{cell.text}</Markdown></div>
            </div>
        </div>
    );
}

export function AskCell({ cell, number, onRun, isProcessing }) {
    return (
        <div className="flex gap-3 animate-in">
            <Gutter number={number} />
            <div className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-line bg-surface">
                <div className="border-b border-line bg-brand-50/50 px-4 py-2.5 text-[15px] text-ink">
                    {cell.prompt}
                </div>
                <div className="flex gap-3 px-4 py-3">
                    <Logo size={24} className="mt-0.5 shrink-0" />
                    <div className="min-w-0 flex-1">
                        {cell.status === 'thinking' ? (
                            <p className="flex items-center gap-2 text-sm text-ink-3">
                                <Loader2 className="h-4 w-4 animate-spin" /> Thinking…
                            </p>
                        ) : (
                            <div className={`prose-chat text-[15px] leading-relaxed ${cell.status === 'error' ? 'text-red-700' : 'text-ink-2'}`}>
                                <Markdown>{cell.reply || ''}</Markdown>
                            </div>
                        )}
                        {cell.suggestions?.length > 0 && (
                            <div className="mt-3 space-y-2">
                                {cell.suggestions.map((action, i) => (
                                    <ActionButton key={i} action={action} onExecute={a => onRun(a, cell.id)} isProcessing={isProcessing} />
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export function RunCell({ cell, number, tool, step, summary, isProcessing, onEdit, onCopy, onDelete, onRerun }) {
    const [editing, setEditing] = useState(false);
    const kind = tool?.kind || 'view';
    const isStep = Boolean(cell.stepId);
    const removed = isStep && !step; // undone or deleted elsewhere
    // Only views can be safely re-run; actions (choose target, train) are never one-click repeats
    const isView = kind === 'view';
    const stale = isView && cell.version && summary?.data_version && cell.version !== summary.data_version;
    const args = step?.args ?? cell.args ?? {};
    const columns = summary?.columns?.map(c => c.name) || [];

    const save = async (newArgs) => {
        if (await onEdit(cell, newArgs)) setEditing(false);
    };

    return (
        <div className={`flex gap-3 animate-in ${removed ? 'opacity-60' : ''}`}>
            <Gutter number={number} />
            <div className="min-w-0 flex-1 rounded-2xl border border-line bg-surface">
                {/* Header: what ran, its kind, and the cell's buttons */}
                <div className="flex items-start justify-between gap-3 px-4 pt-3">
                    <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-semibold text-ink">{humanize(cell.tool)}</span>
                            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${KIND[kind]?.className}`}>
                                {KIND[kind]?.label}
                            </span>
                            {removed && (
                                <span className="flex items-center gap-1 rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[10px] font-semibold text-ink-3">
                                    <Ban className="h-3 w-3" /> No longer in the recipe
                                </span>
                            )}
                        </div>
                        {Object.keys(args).length > 0 && (
                            <div className="mt-1.5 flex flex-wrap gap-1.5">
                                {Object.entries(args).map(([k, v]) => (
                                    <span key={k} className="rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2">
                                        {humanize(k)}: <span className="text-ink">{Array.isArray(v) ? v.join(', ') : typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5">
                        {isStep && !removed && (
                            <IconButton label="Edit settings" onClick={() => setEditing(e => !e)} disabled={isProcessing}>
                                <Pencil className="h-3.5 w-3.5" />
                            </IconButton>
                        )}
                        {isView && (
                            <IconButton label="Rerun on current data" onClick={() => onRerun(cell)} disabled={isProcessing}>
                                <RotateCw className="h-3.5 w-3.5" />
                            </IconButton>
                        )}
                        {kind !== 'action' && (
                            <IconButton label="Copy (run again as a new cell)" onClick={() => onCopy(cell)} disabled={isProcessing || removed}>
                                <Copy className="h-3.5 w-3.5" />
                            </IconButton>
                        )}
                        <IconButton label={isStep && !removed ? 'Delete (removes the step from the recipe)' : 'Delete cell'} onClick={() => onDelete(cell)} disabled={isProcessing} danger>
                            <Trash2 className="h-3.5 w-3.5" />
                        </IconButton>
                    </div>
                </div>

                <div className="space-y-3 px-4 pb-4 pt-2">
                    {stale && (
                        <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                            <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                            <span className="flex-1">Out of date: the data changed after this was made.</span>
                            <button onClick={() => onRerun(cell)} disabled={isProcessing} className="font-semibold underline-offset-2 hover:underline">
                                Rerun
                            </button>
                        </div>
                    )}

                    {editing && tool && (
                        <StepEditor
                            tool={tool}
                            initialArgs={args}
                            columns={columns}
                            busy={isProcessing}
                            onSave={save}
                            onCancel={() => setEditing(false)}
                        />
                    )}

                    <p className="whitespace-pre-wrap text-sm text-ink-2">{step?.message || cell.message}</p>

                    {step && kind === 'learn' && <StickyNoteView learned={step.learned} />}
                    {kind === 'train_only' && !removed && (
                        <p className="text-xs text-ink-3">Applied to the training rows only; skipped for the test set and new customers.</p>
                    )}

                    {cell.evidence && <EvidenceCard evidence={cell.evidence} />}
                </div>
            </div>
        </div>
    );
}

function IconButton({ label, onClick, disabled, danger, children }) {
    return (
        <button
            type="button"
            title={label}
            aria-label={label}
            onClick={onClick}
            disabled={disabled}
            className={`rounded-lg p-1.5 text-ink-3 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                danger ? 'hover:bg-red-50 hover:text-red-700' : 'hover:bg-surface-2 hover:text-ink'
            }`}
        >
            {children}
        </button>
    );
}

/** The step's sticky note: what it learned from the training rows, in plain words. */
function StickyNoteView({ learned }) {
    const rows = [];
    for (const [col, value] of Object.entries(learned?.fill || {})) rows.push([col, `blanks → ${value}`]);
    for (const [col, classes] of Object.entries(learned?.classes || {})) {
        rows.push([col, `${classes.length} categories: ${classes.slice(0, 6).join(', ')}${classes.length > 6 ? '…' : ''}`]);
    }
    for (const [col, s] of Object.entries(learned?.scale || {})) rows.push([col, `average ${s.mean}, spread ${s.std}`]);
    if (!rows.length) return null;

    return (
        <div className="rounded-xl border border-amber-200 bg-amber-50/70 px-3.5 py-2.5">
            <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-amber-900">
                <StickyNote className="h-3.5 w-3.5" /> Learned from the training rows (reused for test rows and new customers)
            </p>
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-xs">
                {rows.map(([col, text]) => (
                    <React.Fragment key={col + text}>
                        <dt className="font-mono font-medium text-ink">{col}</dt>
                        <dd className="font-mono text-ink-2">{text}</dd>
                    </React.Fragment>
                ))}
            </dl>
        </div>
    );
}
