import React from 'react';
import { Trophy, GraduationCap } from 'lucide-react';
import { modelName } from '../lib/modelNames';
import { ChartRenderer } from './ChartRenderer';

/**
 * EvidenceCard — renders rich evidence data (tables, stats, charts, model results).
 */
export function EvidenceCard({ evidence }) {
    if (!evidence) return null;

    return (
        <div className="mt-4 max-w-full space-y-3">
            {evidence.chart_type && <ChartRenderer evidence={evidence} />}

            {evidence.preview && <PreviewTable data={evidence.preview} columns={evidence.preview_columns} />}

            {evidence.output && (
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-surface-2 p-3 font-mono text-[12px] leading-relaxed text-ink">
                    {evidence.output}
                </pre>
            )}

            {evidence.images?.map((src, i) => (
                <img key={i} src={src} alt={`Chart ${i + 1} from your code`} className="max-w-full rounded-xl border border-line bg-surface" />
            ))}

            {evidence.describe && <DescribeTable data={evidence.describe} />}

            {evidence.info && (
                <pre className="overflow-x-auto rounded-xl border border-line bg-surface-2 p-3 font-mono text-[11px] leading-relaxed text-ink-2">
                    {evidence.info}
                </pre>
            )}

            {evidence.results && <ModelResults results={evidence.results} bestModel={evidence.best_model} />}

            {evidence.exam && <ExamCard exam={evidence.exam} />}

            {evidence.prediction && (
                <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wider text-brand-700">Prediction</div>
                    <div className="mt-1 text-2xl font-bold text-ink">{evidence.prediction}</div>
                    {evidence.model_used && (
                        <div className="mt-1 text-xs text-ink-3">
                            Model: <span className="font-medium text-ink-2">{modelName(evidence.model_used)}</span>
                        </div>
                    )}
                    {evidence.probabilities && (
                        <div className="mt-3 space-y-1.5">
                            <p className="text-xs font-semibold text-ink-2">How sure the model is</p>
                            {Object.entries(evidence.probabilities)
                                .sort((a, b) => b[1] - a[1])
                                .map(([label, p]) => (
                                    <div key={label} className="flex items-center gap-2 text-xs">
                                        <span className="w-20 truncate font-mono text-ink" title={label}>{label}</span>
                                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface">
                                            <div
                                                className={`h-full rounded-full ${label === evidence.prediction ? 'bg-brand-500' : 'bg-ink-3/40'}`}
                                                style={{ width: `${p * 100}%` }}
                                            />
                                        </div>
                                        <span className="w-12 text-right font-mono tabular-nums text-ink-2">{(p * 100).toFixed(0)}%</span>
                                    </div>
                                ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

const tableShell = 'max-h-80 overflow-auto rounded-xl border border-line bg-surface';
const th = 'sticky top-0 z-10 bg-surface-2 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-ink-3 whitespace-nowrap';

function PreviewTable({ data, columns }) {
    if (!data?.length) return null;
    // The server's order: the browser moves number-like names ("1", "2") to the front of a row's keys
    const headers = columns?.length ? columns : Object.keys(data[0] || {});

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

const asScore = (value, metric) => (value == null ? '–' : metric === 'accuracy' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3));

function ExamCard({ exam }) {
    const rows = [
        { label: `Model: ${modelName(exam.model)}`, value: exam.score, className: 'bg-brand-500' },
        { label: `Just guessing: ${exam.baseline_label}`, value: exam.baseline, className: 'bg-ink-3/50' },
        ...(exam.practice != null ? [{ label: 'Practice score (during training)', value: exam.practice, className: 'bg-brand-200' }] : []),
    ];
    const max = Math.max(1, ...rows.map(r => r.value ?? 0));

    return (
        <div className="space-y-4 rounded-xl border border-brand-200 bg-surface p-4">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-brand-700">
                        <GraduationCap className="h-3.5 w-3.5" /> Final exam · honest score
                    </p>
                    <p className="mt-1 text-4xl font-bold tabular-nums text-ink">{asScore(exam.score, exam.metric)}</p>
                    <p className="mt-0.5 text-sm text-ink-2">
                        {exam.metric} on <strong>{exam.test_rows}</strong> locked rows the model never saw and no step learned from
                    </p>
                </div>
                <span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${exam.look > 1 ? 'bg-amber-100 text-amber-900' : 'bg-emerald-50 text-emerald-800'}`}>
                    Look #{exam.look}
                </span>
            </div>

            <div className="space-y-2">
                {rows.map(r => (
                    <div key={r.label}>
                        <div className="mb-0.5 flex justify-between text-xs">
                            <span className="text-ink-2">{r.label}</span>
                            <span className="font-mono font-semibold tabular-nums text-ink">{asScore(r.value, exam.metric)}</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                            <div className={`h-full rounded-full ${r.className}`} style={{ width: `${Math.max(0, (r.value ?? 0) / max) * 100}%` }} />
                        </div>
                    </div>
                ))}
            </div>

            {exam.no_better_than_guessing && (
                <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-800">
                    ⚠ No better than just guessing. The score only looks good because one answer is very common:
                    {exam.baseline_label} scores the same. This model hasn't learned anything useful yet.
                </p>
            )}

            {exam.look > 1 && (
                <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                    This is look #{exam.look} at the test rows. Each extra look makes the score less trustworthy: you start
                    choosing whatever happens to work on these particular rows.
                </p>
            )}

            {exam.confusion && (
                <div>
                    <p className="mb-1.5 text-xs font-semibold text-ink-2">Where it was right and wrong (rows: actual, columns: predicted)</p>
                    <table className="text-xs">
                        <thead>
                            <tr>
                                <th />
                                {exam.labels.map(l => <th key={l} className="px-3 py-1 font-mono font-medium text-ink-3">{l}</th>)}
                            </tr>
                        </thead>
                        <tbody>
                            {exam.confusion.map((row, i) => (
                                <tr key={exam.labels[i]}>
                                    <td className="pr-3 text-right font-mono font-medium text-ink-3">{exam.labels[i]}</td>
                                    {row.map((count, j) => (
                                        <td key={j} className="p-0.5">
                                            <div className={`min-w-12 rounded-md px-3 py-1.5 text-center font-mono font-semibold tabular-nums ${
                                                i === j ? 'bg-emerald-50 text-emerald-800' : count ? 'bg-red-50 text-red-800' : 'bg-surface-2 text-ink-3'
                                            }`}>
                                                {count}
                                            </div>
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {exam.mae != null && (
                <p className="text-xs text-ink-2">
                    Average error: <span className="font-mono font-semibold text-ink">{exam.mae}</span> (guessing the average:{' '}
                    <span className="font-mono">{exam.baseline_mae}</span>)
                </p>
            )}
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
                                <span className="text-sm font-semibold text-ink" title={r.model}>{modelName(r.model)}</span>
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
