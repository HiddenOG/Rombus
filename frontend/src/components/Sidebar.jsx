import React, { useCallback, useState } from 'react';
import { Upload, FileSpreadsheet, Layers, Sparkles, Undo2, Lock, Target, Boxes, NotebookPen, Plus, Trash2, PanelLeftClose, ChevronRight } from 'lucide-react';
import { Logo } from './Logo';
import { ConfirmDialog } from './ConfirmDialog';
import { modelName } from '../lib/modelNames';
import { AccountPanel } from './AccountPanel';

/**
 * Sidebar — brand, dataset overview and column evidence panel.
 */
export function Sidebar({ open, onClose, sessionId, summary, onFileUpload, isProcessing, notebooks = [], onOpenNotebook, onNewNotebook, onDeleteNotebook, account }) {
    return (
        <>
            {/* Phones/tablets: the sidebar is a drawer over the page, with a backdrop that fades in and out */}
            <div
                className={`fixed inset-0 z-30 bg-ink/30 transition-opacity duration-500 ease-in-out motion-reduce:transition-none lg:hidden ${
                    open ? 'opacity-100' : 'pointer-events-none opacity-0'
                }`}
                onClick={onClose}
                aria-hidden="true"
            />
            {/* Opening and closing glide: phones slide the drawer in and out; computers shrink the column's
                width to nothing while the notebook widens into the space. The contents keep their width
                (they're clipped, not squeezed), and a closed sidebar can't be tabbed into (inert). */}
            <aside
                aria-label="Notebooks and dataset"
                inert={!open}
                className={`fixed inset-y-0 left-0 z-40 shrink-0 overflow-hidden bg-surface transition-[transform,width] duration-500 ease-in-out motion-reduce:transition-none lg:static lg:z-auto lg:translate-x-0 ${
                    open ? 'translate-x-0 lg:w-80' : '-translate-x-full lg:w-0'
                }`}
            >
            <div className="flex h-full w-[85vw] max-w-80 flex-col border-r border-line lg:w-80">
            {/* Brand */}
            <div className="shrink-0 h-16 px-5 flex items-center gap-3 border-b border-line">
                <Logo size={36} className="drop-shadow-sm" />
                <div className="min-w-0 flex-1 leading-tight">
                    <h1 className="text-[15px] font-bold text-ink">Rhombus</h1>
                    <p className="text-xs text-ink-3">Data science you can trust</p>
                </div>
                <button onClick={onClose} title="Close sidebar" aria-label="Close sidebar" className="rounded-lg p-1.5 text-ink-3 hover:bg-surface-2 hover:text-ink">
                    <PanelLeftClose className="h-4.5 w-4.5" />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-6 custom-scrollbar">
                <NotebookList
                    notebooks={notebooks}
                    currentId={sessionId}
                    onOpen={onOpenNotebook}
                    onNew={onNewNotebook}
                    onDelete={onDeleteNotebook}
                    disabled={isProcessing}
                />
                {!sessionId ? (
                    <UploadCard onFileUpload={onFileUpload} isProcessing={isProcessing} />
                ) : (
                    <>
                        <DatasetOverview summary={summary} />
                        <ColumnList summary={summary} />
                    </>
                )}
            </div>
            <AccountPanel account={account} />
            </div>
            </aside>
        </>
    );
}

function timeAgo(seconds) {
    const mins = Math.round((Date.now() / 1000 - seconds) / 60);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins} min ago`;
    const hours = Math.round(mins / 60);
    return hours < 24 ? `${hours} h ago` : `${Math.round(hours / 24)} d ago`;
}

const FOLD_KEY = 'rhombus.notebooks.folded';
const SEARCH_FROM = 6; // a search box once there are more notebooks than this

function NotebookList({ notebooks, currentId, onOpen, onNew, onDelete, disabled }) {
    const [pending, setPending] = useState(null); // the notebook waiting for "are you sure?"
    const confirmDelete = (n) => setPending(n);
    const cancel = useCallback(() => setPending(null), []);
    // Many notebooks: fold the list away, or search it. It scrolls inside its own box either way,
    // so the dataset cards below never get pushed out of sight
    const [folded, setFolded] = useState(() => {
        try { return localStorage.getItem(FOLD_KEY) === '1'; } catch { return false; }
    });
    const [query, setQuery] = useState('');
    const toggle = () => setFolded(f => {
        try { localStorage.setItem(FOLD_KEY, f ? '0' : '1'); } catch { /* not remembered */ }
        return !f;
    });
    const words = query.trim().toLowerCase();
    const shown = words ? notebooks.filter(n => `${n.filename} ${n.target || ''}`.toLowerCase().includes(words)) : notebooks;

    return (
        <section>
            <div className="mb-2.5 flex items-center justify-between">
                <button
                    type="button"
                    onClick={toggle}
                    aria-expanded={!folded}
                    className="-mt-2.5 flex items-center gap-1 rounded-lg py-1 pr-1.5 text-left hover:text-ink"
                    title={folded ? 'Show notebooks' : 'Hide notebooks'}
                >
                    <ChevronRight className={`h-3.5 w-3.5 text-ink-3 transition-transform ${folded ? '' : 'rotate-90'}`} />
                    <SectionTitle icon={NotebookPen}>
                        Notebooks{notebooks.length > 0 && <span className="ml-1 font-normal text-ink-3">· {notebooks.length}</span>}
                    </SectionTitle>
                </button>
                <button
                    onClick={onNew}
                    disabled={disabled}
                    className="-mt-2.5 flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-brand-700 hover:bg-brand-50 disabled:opacity-50"
                >
                    <Plus className="h-3.5 w-3.5" /> New
                </button>
            </div>
            {folded ? null : notebooks.length === 0 ? (
                <p className="text-xs text-ink-3">Your notebooks will appear here.</p>
            ) : (
                <>
                {notebooks.length > SEARCH_FROM && (
                    <input
                        type="search"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        placeholder={`Search ${notebooks.length} notebooks`}
                        aria-label="Search notebooks"
                        className="mb-1.5 w-full rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-xs text-ink outline-none focus:border-brand-300"
                    />
                )}
                {shown.length === 0 && <p className="px-1 text-xs text-ink-3">No notebook matches “{query}”.</p>}
                <ul className="max-h-44 space-y-0.5 overflow-y-auto custom-scrollbar">
                    {shown.map(n => (
                        <li key={n.session_id} className="group relative">
                            <button
                                onClick={() => onOpen(n.session_id)}
                                disabled={disabled}
                                className={`w-full rounded-lg px-2.5 py-1.5 pr-8 text-left transition-colors ${
                                    n.session_id === currentId ? 'bg-brand-50 text-brand-800' : 'text-ink-2 hover:bg-surface-2'
                                }`}
                            >
                                <span className="block truncate text-sm font-medium">{n.filename}</span>
                                <span className="block truncate text-[11px] text-ink-3">
                                    {n.target ? `predicting ${n.target} · ` : ''}{timeAgo(n.updated_at)}
                                </span>
                            </button>
                            <button
                                onClick={() => confirmDelete(n)}
                                disabled={disabled}
                                title="Delete notebook"
                                aria-label={`Delete notebook ${n.filename}`}
                                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-ink-3 transition-opacity hover:bg-red-50 hover:text-red-700 focus:opacity-100 pointer-fine:opacity-0 pointer-fine:group-hover:opacity-100"
                            >
                                <Trash2 className="h-3.5 w-3.5" />
                            </button>
                        </li>
                    ))}
                </ul>
                </>
            )}

            <ConfirmDialog
                open={Boolean(pending)}
                title="Delete this notebook?"
                confirmLabel="Delete notebook"
                busy={disabled}
                onCancel={cancel}
                onConfirm={() => {
                    onDelete(pending.session_id);
                    setPending(null);
                }}
            >
                <p>
                    <strong className="text-ink">{pending?.filename}</strong>
                    {pending?.target ? <> (predicting <span className="font-mono">{pending.target}</span>)</> : null} will be removed
                    for good: the uploaded data, the locked test rows, the recipe and every cell.
                </p>
                <p className="mt-2 font-medium text-red-700">This can't be undone.</p>
            </ConfirmDialog>
        </section>
    );
}

function UploadCard({ onFileUpload, isProcessing }) {
    return (
        <label
            className={`block rounded-2xl border-2 border-dashed border-brand-200 bg-brand-50/60 p-6 text-center transition-colors ${
                isProcessing ? 'cursor-wait opacity-70' : 'cursor-pointer hover:border-brand-400 hover:bg-brand-50'
            }`}
        >
            <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-surface border border-brand-100 flex items-center justify-center">
                <Upload className="w-5 h-5 text-brand-600" />
            </div>
            <p className="text-sm font-semibold text-ink">{isProcessing ? 'Uploading…' : 'Upload a dataset'}</p>
            <p className="text-xs text-ink-3 mt-1">CSV or Excel, up to 20 MB. Pick several to stack them into one table.</p>
            <span className="inline-block mt-4 rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-600">
                Choose files
            </span>
            <input
                type="file"
                multiple
                className="hidden"
                disabled={isProcessing}
                onChange={(e) => {
                    if (e.target.files.length) onFileUpload(e.target.files);
                }}
                accept=".csv,.xlsx,.xls"
            />
        </label>
    );
}

function DatasetOverview({ summary }) {
    return (
        <section>
            <SectionTitle icon={FileSpreadsheet}>Dataset</SectionTitle>
            <div className="rounded-2xl border border-line bg-surface p-4 space-y-4">
                <p className="text-sm font-semibold text-ink truncate" title={summary?.filename}>
                    {summary?.filename}
                </p>

                <div className="grid grid-cols-2 gap-2">
                    <StatTile
                        label={summary?.is_split ? 'Training rows' : 'Rows'}
                        value={summary?.shape?.[0]?.toLocaleString()}
                    />
                    <StatTile label="Columns" value={summary?.shape?.[1]} />
                    {summary?.is_split && (
                        <StatTile
                            icon={Lock}
                            label="Locked test rows"
                            value={summary.test_rows.toLocaleString()}
                            note={`${Math.round(summary.test_size * 100)}% set aside`}
                            wide
                        />
                    )}
                </div>

                <div className="space-y-2 text-sm">
                    <InfoRow icon={Target} label="Target">
                        {summary?.target ? (
                            <span className="font-semibold text-brand-700">{summary.target}</span>
                        ) : summary?.clean_only ? (
                            <span className="font-medium text-ink-2">None yet: exploring</span>
                        ) : (
                            <span className="text-ink-3">Not chosen yet</span>
                        )}
                    </InfoRow>
                    {summary?.sealed_rows > 0 && (
                        <InfoRow icon={Lock} label="Sealed">
                            <span
                                className="font-medium text-ink-2"
                                title="Never shown: if you predict a column later, these rows are its test set, so its score is honest"
                            >
                                {summary.sealed_rows.toLocaleString()} rows ({Math.round(summary.sealed_share * 100)}%)
                            </span>
                        </InfoRow>
                    )}
                    {summary?.problem_type && (
                        <InfoRow icon={Boxes} label="Problem">
                            <span className="font-medium text-ink capitalize">{summary.problem_type}</span>
                        </InfoRow>
                    )}
                    {summary?.trained_models?.length > 0 && (
                        <InfoRow icon={Sparkles} label="Models">
                            <span className="text-right text-xs font-medium text-ink">{summary.trained_models.map(modelName).join(', ')}</span>
                        </InfoRow>
                    )}
                </div>

                {summary?.can_undo && (
                    <div className="flex items-center gap-1.5 rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs font-medium text-ink-2">
                        <Undo2 className="w-3.5 h-3.5" />
                        {summary.undo_count} undo step{summary.undo_count > 1 ? 's' : ''} available
                    </div>
                )}
            </div>
        </section>
    );
}

function ColumnList({ summary }) {
    return (
        <section>
            <SectionTitle icon={Layers}>Columns · {summary?.columns?.length}</SectionTitle>
            {!summary?.is_split && !summary?.clean_only && (
                <p className="mb-3 flex gap-2 rounded-xl bg-brand-50 border border-brand-100 px-3 py-2.5 text-xs leading-relaxed text-ink-2">
                    <Lock className="w-3.5 h-3.5 mt-0.5 shrink-0 text-brand-600" />
                    Statistics stay hidden until you choose the column to predict and lock away the test rows.
                </p>
            )}
            <div className="space-y-2">
                {summary?.columns?.map(col => (
                    <ColumnCard key={col.name} col={col} isTarget={col.name === summary?.target} />
                ))}
            </div>
        </section>
    );
}

function SectionTitle({ icon: Icon, children }) {
    return (
        <h3 className="mb-2.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-3">
            <Icon className="w-3.5 h-3.5" /> {children}
        </h3>
    );
}

function StatTile({ label, value, note, icon: Icon, wide = false }) {
    return (
        <div className={`rounded-xl bg-surface-2 px-3 py-2.5 ${wide ? 'col-span-2' : ''}`}>
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-ink-3">
                {Icon && <Icon className="w-3 h-3" />}
                {label}
            </div>
            <div className="mt-0.5 flex items-baseline gap-2">
                <span className="text-lg font-bold text-ink">{value ?? '—'}</span>
                {note && <span className="text-xs text-ink-3">{note}</span>}
            </div>
        </div>
    );
}

function InfoRow({ icon: Icon, label, children }) {
    return (
        <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-1.5 text-ink-3">
                <Icon className="w-3.5 h-3.5" /> {label}
            </span>
            <span className="truncate">{children}</span>
        </div>
    );
}

function ColumnCard({ col, isTarget }) {
    const missingPct = (col.missing_ratio * 100).toFixed(1);
    const hasMissing = col.missing_ratio > 0;
    const isSkewed = col.skewness != null && Math.abs(col.skewness) > 1;

    return (
        <div
            className={`rounded-xl border p-3 text-sm transition-colors ${
                isTarget ? 'border-brand-200 bg-brand-50/50' : 'border-line bg-surface hover:border-[#ddd5cb]'
            }`}
        >
            <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-ink truncate" title={col.name}>
                    {col.name}
                </span>
                <span className="shrink-0 rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-2">
                    {col.dtype}
                </span>
            </div>
            {col.missing_ratio != null && (
                <div className="mt-1.5 grid grid-cols-2 gap-x-2 gap-y-0.5 text-xs text-ink-3">
                    <span className={hasMissing ? 'font-medium text-red-600' : ''}>Missing {missingPct}%</span>
                    <span>Unique {col.unique_count}</span>
                    {col.skewness != null && (
                        <span className={isSkewed ? 'font-medium text-amber-700' : ''}>
                            Skew {col.skewness.toFixed(2)}
                        </span>
                    )}
                    {col.mean != null && <span>Mean {col.mean.toFixed(2)}</span>}
                </div>
            )}
            {hasMissing && (
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-2">
                    <div
                        className="h-full rounded-full bg-red-400"
                        style={{ width: `${Math.min(col.missing_ratio * 100, 100)}%` }}
                    />
                </div>
            )}
        </div>
    );
}
