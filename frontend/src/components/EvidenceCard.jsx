import React from 'react';
import { Trophy } from 'lucide-react';
import { ChartRenderer } from './ChartRenderer';

/**
 * EvidenceCard — renders rich evidence data (tables, stats, charts, model results).
 */
export function EvidenceCard({ evidence }) {
    if (!evidence) return null;

    return (
        <div className="mt-4 max-w-full space-y-3">
            {evidence.chart_type && <ChartRenderer evidence={evidence} />}

            {evidence.preview && <PreviewTable data={evidence.preview} />}

            {evidence.describe && <DescribeTable data={evidence.describe} />}

            {evidence.info && (
                <pre className="overflow-x-auto rounded-xl border border-line bg-surface-2 p-3 font-mono text-[11px] leading-relaxed text-ink-2">
                    {evidence.info}
                </pre>
            )}

            {evidence.results && <ModelResults results={evidence.results} bestModel={evidence.best_model} />}

            {evidence.prediction && (
                <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wider text-brand-700">Prediction</div>
                    <div className="mt-1 text-2xl font-bold text-ink">{evidence.prediction}</div>
                    {evidence.model_used && (
                        <div className="mt-1 text-xs text-ink-3">
                            Model: <span className="font-mono uppercase">{evidence.model_used}</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

const tableShell = 'max-h-80 overflow-auto rounded-xl border border-line bg-surface';
const th = 'sticky top-0 z-10 bg-surface-2 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-ink-3 whitespace-nowrap';

function PreviewTable({ data }) {
    if (!data?.length) return null;
    const headers = Object.keys(data[0] || {});

    return (
        <div className={tableShell}>
            <table className="min-w-full text-xs">
                <thead>
                    <tr>{headers.map(h => <th key={h} className={th}>{h}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-line">
                    {data.map((row, i) => (
                        <tr key={i} className="hover:bg-brand-50/40">
                            {headers.map(h => {
                                const value = row[h];
                                const isBlank = value === null || value === undefined || value === '';
                                return (
                                    <td
                                        key={h}
                                        className={`whitespace-nowrap px-3 py-2 tabular-nums ${isBlank ? 'italic text-ink-3' : 'text-ink'}`}
                                    >
                                        {isBlank ? 'blank' : String(value)}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function DescribeTable({ data }) {
    if (!data) return null;
    const columns = Object.keys(data);
    if (!columns.length) return null;
    const stats = Object.keys(data[columns[0]] || {});

    return (
        <div className={tableShell}>
            <table className="min-w-full text-xs">
                <thead>
                    <tr>
                        <th className={th}>Statistic</th>
                        {columns.map(h => <th key={h} className={th}>{h}</th>)}
                    </tr>
                </thead>
                <tbody className="divide-y divide-line">
                    {stats.map(stat => (
                        <tr key={stat} className="hover:bg-brand-50/40">
                            <td className="whitespace-nowrap px-3 py-1.5 font-semibold text-ink-2">{stat}</td>
                            {columns.map(col => {
                                const v = data[col][stat];
                                return (
                                    <td key={col} className="whitespace-nowrap px-3 py-1.5 font-mono tabular-nums text-ink">
                                        {v == null ? '–' : typeof v === 'number' ? v.toFixed(2) : String(v)}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function ModelResults({ results, bestModel }) {
    if (!results?.length) return null;

    return (
        <div className="space-y-2">
            {results.map((r, i) => {
                const isBest = r.model === bestModel;
                return (
                    <div
                        key={i}
                        className={`rounded-xl border p-3.5 ${isBest ? 'border-brand-300 bg-brand-50/60' : 'border-line bg-surface'}`}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                                <span className="font-mono text-sm font-semibold uppercase text-ink">{r.model}</span>
                                {isBest && (
                                    <span className="flex items-center gap-1 rounded-full bg-brand-500 px-2 py-0.5 text-[10px] font-bold uppercase text-white">
                                        <Trophy className="w-3 h-3" /> Best
                                    </span>
                                )}
                            </div>
                            <div className="font-mono text-sm font-semibold text-ink">
                                {r.accuracy != null && `Accuracy ${r.accuracy}`}
                                {r.r2_score != null && `R² ${r.r2_score}`}
                                {r.error && <span className="font-sans text-xs font-medium text-red-700">{r.error}</span>}
                            </div>
                        </div>

                        {(r.cv_mean != null || r.f1_score != null || r.mae != null) && (
                            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-ink-3">
                                {r.cv_mean != null && <span>CV {r.cv_mean} ± {r.cv_std}</span>}
                                {r.f1_score != null && <span>F1 {r.f1_score}</span>}
                                {r.mae != null && <span>MAE {r.mae}</span>}
                                {r.rmse != null && <span>RMSE {r.rmse}</span>}
                            </div>
                        )}

                        {r.best_params && (
                            <div className="mt-1.5 font-mono text-[11px] text-ink-3">
                                Params {JSON.stringify(r.best_params)}
                            </div>
                        )}

                        {r.report && (
                            <pre className="mt-2.5 overflow-x-auto rounded-lg bg-surface-2 p-2.5 font-mono text-[10px] leading-relaxed text-ink-2">
                                {r.report}
                            </pre>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
