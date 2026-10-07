import React from 'react';
import { Upload, Database, Eye, Layers, Zap, Activity, Undo2 } from 'lucide-react';

/**
 * Sidebar — dataset overview and column evidence panel.
 */
export function Sidebar({ sessionId, currentStage, summary, onFileUpload, isProcessing }) {
    return (
        <div className="w-72 bg-white border-r border-gray-200 flex flex-col overflow-hidden">
            {/* Header */}
            <div className="shrink-0 p-4 border-b border-gray-200 bg-gradient-to-br from-indigo-600 to-indigo-700 text-white">
                <div className="flex items-center gap-2.5 mb-2">
                    <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                        <Zap className="w-5 h-5" />
                    </div>
                    <div>
                        <h1 className="text-base font-bold leading-tight">AutoML Copilot</h1>
                        <p className="text-xs text-indigo-200 font-medium">AI-Powered Data Science</p>
                    </div>
                </div>
                <div className="flex items-center gap-2 mt-3 bg-white/10 rounded-lg px-3 py-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    <span className="text-xs font-bold uppercase tracking-widest">{currentStage}</span>
                </div>
            </div>

            {/* Content — scrollable */}
            <div className="flex-1 overflow-y-auto p-4 space-y-5 custom-scrollbar">
                {!sessionId ? (
                    <div className="text-center py-12">
                        <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                            <Database className="w-8 h-8 text-gray-300" />
                        </div>
                        <p className="text-sm text-gray-500 mb-4">Upload a dataset to begin</p>
                        <label className="inline-flex items-center gap-2 bg-indigo-600 text-white px-5 py-2.5 rounded-xl cursor-pointer hover:bg-indigo-700 transition-all text-sm font-medium shadow-lg shadow-indigo-100 hover:shadow-indigo-200">
                            <Upload className="w-4 h-4" />
                            Upload CSV / Excel
                            <input
                                type="file"
                                className="hidden"
                                onChange={(e) => {
                                    const file = e.target.files[0];
                                    if (file) onFileUpload(file);
                                }}
                                accept=".csv,.xlsx,.xls"
                            />
                        </label>
                    </div>
                ) : (
                    <>
                        {/* Dataset Overview */}
                        <div>
                            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                                <Eye className="w-3.5 h-3.5" /> Dataset Overview
                            </h3>
                            <div className="bg-gray-50 rounded-xl p-3 space-y-2 text-sm">
                                <DataRow label="File" value={summary?.filename} />
                                <DataRow label="Rows" value={summary?.shape?.[0]?.toLocaleString()} mono />
                                <DataRow label="Columns" value={summary?.shape?.[1]} mono />
                                <DataRow
                                    label="Target"
                                    value={summary?.target || 'Not Set'}
                                    highlight={!!summary?.target}
                                />
                                <DataRow
                                    label="Problem"
                                    value={summary?.problem_type || 'N/A'}
                                />
                                {summary?.can_undo && (
                                    <div className="flex items-center gap-1 text-amber-600 text-xs font-medium pt-1">
                                        <Undo2 className="w-3.5 h-3.5" />
                                        {summary.undo_count} undo step{summary.undo_count > 1 ? 's' : ''} available
                                    </div>
                                )}
                                {summary?.trained_models?.length > 0 && (
                                    <DataRow
                                        label="Models"
                                        value={summary.trained_models.join(', ').toUpperCase()}
                                        highlight
                                    />
                                )}
                            </div>
                        </div>

                        {/* Column Evidence */}
                        <div>
                            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                                <Layers className="w-3.5 h-3.5" /> Columns ({summary?.columns?.length})
                            </h3>
                            <div className="space-y-2">
                                {summary?.columns?.map(col => (
                                    <ColumnCard key={col.name} col={col} />
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

function DataRow({ label, value, mono = false, highlight = false }) {
    return (
        <div className="flex justify-between items-center">
            <span className="text-gray-500 text-sm">{label}</span>
            <span
                className={`text-sm font-semibold truncate max-w-[130px] ${highlight ? 'text-indigo-600' : 'text-gray-800'
                    } ${mono ? 'font-mono' : ''}`}
                title={value}
            >
                {value}
            </span>
        </div>
    );
}

function ColumnCard({ col }) {
    const missingPct = (col.missing_ratio * 100).toFixed(1);
    const hasMissing = col.missing_ratio > 0;
    const isSkewed = col.skewness != null && Math.abs(col.skewness) > 1;

    return (
        <div className="p-2.5 bg-white border border-gray-100 rounded-xl shadow-sm text-sm hover:shadow-md transition-shadow">
            <div className="flex justify-between mb-1.5">
                <span className="font-bold truncate max-w-[120px] text-gray-800" title={col.name}>
                    {col.name}
                </span>
                <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full font-mono">
                    {col.dtype}
                </span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-xs text-gray-500">
                <span className={hasMissing ? 'text-red-500 font-medium' : ''}>
                    Missing: {missingPct}%
                </span>
                <span>Unique: {col.unique_count}</span>
                {col.skewness != null && (
                    <span className={isSkewed ? 'text-orange-500 font-medium' : ''}>
                        Skew: {col.skewness.toFixed(2)}
                    </span>
                )}
                {col.mean != null && (
                    <span>Mean: {col.mean.toFixed(2)}</span>
                )}
            </div>
            {/* Missing bar */}
            {hasMissing && (
                <div className="mt-1.5 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-red-400 rounded-full"
                        style={{ width: `${Math.min(col.missing_ratio * 100, 100)}%` }}
                    />
                </div>
            )}
        </div>
    );
}
