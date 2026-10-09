import React from 'react';
import { Upload, FileSpreadsheet, Layers, Sparkles, Undo2, Lock, Target, Boxes } from 'lucide-react';
import { Logo } from './Logo';

/**
 * Sidebar — brand, dataset overview and column evidence panel.
 */
export function Sidebar({ sessionId, summary, onFileUpload, isProcessing }) {
    return (
        <aside className="w-80 shrink-0 bg-surface border-r border-line flex flex-col overflow-hidden">
            {/* Brand */}
            <div className="shrink-0 h-16 px-5 flex items-center gap-3 border-b border-line">
                <Logo size={36} className="drop-shadow-sm" />
                <div className="leading-tight">
                    <h1 className="text-[15px] font-bold text-ink">AutoML Copilot</h1>
                    <p className="text-xs text-ink-3">Data science you can trust</p>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-6 custom-scrollbar">
                {!sessionId ? (
                    <UploadCard onFileUpload={onFileUpload} isProcessing={isProcessing} />
                ) : (
                    <>
                        <DatasetOverview summary={summary} />
                        <ColumnList summary={summary} />
                    </>
                )}
            </div>
        </aside>
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
            <p className="text-xs text-ink-3 mt-1">CSV or Excel, up to 50 MB</p>
            <span className="inline-block mt-4 rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-600">
                Choose file
            </span>
            <input
                type="file"
                className="hidden"
                disabled={isProcessing}
                onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) onFileUpload(file);
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
                        ) : (
                            <span className="text-ink-3">Not chosen yet</span>
                        )}
                    </InfoRow>
                    {summary?.problem_type && (
                        <InfoRow icon={Boxes} label="Problem">
                            <span className="font-medium text-ink capitalize">{summary.problem_type}</span>
                        </InfoRow>
                    )}
                    {summary?.trained_models?.length > 0 && (
                        <InfoRow icon={Sparkles} label="Models">
                            <span className="font-mono text-xs text-ink uppercase">{summary.trained_models.join(', ')}</span>
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
            {!summary?.is_split && (
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
