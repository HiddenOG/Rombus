import React from 'react';
import { ChartRenderer } from './ChartRenderer';

/**
 * EvidenceCard — renders rich evidence data (tables, stats, charts, model results).
 */
export function EvidenceCard({ evidence }) {
    if (!evidence) return null;

    return (
        <div className="mt-4 max-w-full space-y-3">
            {/* Chart rendering (histogram, boxplot, scatter, heatmap, feature importance) */}
            {evidence.chart_type && <ChartRenderer evidence={evidence} />}

            {/* Data preview table */}
            {evidence.preview && <PreviewTable data={evidence.preview} />}

            {/* Descriptive statistics */}
            {evidence.describe && <DescribeTable data={evidence.describe} />}

            {/* df.info() output */}
            {evidence.info && (
                <div className="p-3 bg-gray-900 text-green-400 font-mono text-[10px] rounded-lg whitespace-pre overflow-x-auto leading-relaxed">
                    {evidence.info}
                </div>
            )}

            {/* Model training results */}
            {evidence.results && <ModelResults results={evidence.results} bestModel={evidence.best_model} />}

            {/* Prediction result */}
            {evidence.prediction && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                    <div className="text-xs font-bold text-emerald-700 mb-1">Prediction</div>
                    <div className="text-lg font-bold text-emerald-800">{evidence.prediction}</div>
                    {evidence.model_used && (
                        <div className="text-[10px] text-emerald-600 mt-1">Model: {evidence.model_used}</div>
                    )}
                </div>
            )}

            {/* Correlation matrix (backward compat — old text format) */}
            {evidence.correlation && !evidence.chart_type && (
                <div className="p-2 bg-white border rounded-lg text-[10px]">
                    <p className="font-bold mb-1">Correlation Matrix</p>
                    {Object.keys(evidence.correlation).slice(0, 5).map(c => (
                        <div key={c} className="truncate">
                            <span className="font-bold">{c}:</span>{' '}
                            {Object.entries(evidence.correlation[c])
                                .slice(0, 3)
                                .map(([k, v]) => `${k}(${v.toFixed(2)})`)
                                .join(', ')}
                            ...
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function PreviewTable({ data }) {
    if (!data?.length) return null;
    const headers = Object.keys(data[0] || {});

    return (
        <div className="border rounded-lg overflow-auto max-h-80 shadow-inner bg-white">
            <table className="min-w-full divide-y divide-gray-200 text-xs">
                <thead className="bg-gray-50 sticky top-0 z-10">
                    <tr>
                        {headers.map(h => (
                            <th key={h} className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider border-r text-[10px]">
                                {h}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {data.map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50 transition-colors">
                            {headers.map((h, j) => (
                                <td key={j} className="px-3 py-2 whitespace-nowrap border-r text-gray-700">
                                    {String(row[h] ?? '')}
                                </td>
                            ))}
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
        <div className="border rounded-lg overflow-auto max-h-80 shadow-inner bg-white">
            <p className="bg-gray-100 px-3 py-1.5 font-bold text-[10px] uppercase sticky top-0 z-10 border-b tracking-wider text-gray-500">
                Descriptive Statistics
            </p>
            <table className="min-w-full divide-y divide-gray-200 text-[10px]">
                <thead className="bg-gray-50">
                    <tr>
                        <th className="px-2 py-1 text-left font-medium text-gray-500 uppercase border-r">Stat</th>
                        {columns.map(h => (
                            <th key={h} className="px-2 py-1 text-left font-medium text-gray-500 uppercase border-r">{h}</th>
                        ))}
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {stats.map(stat => (
                        <tr key={stat} className="hover:bg-gray-50">
                            <td className="px-2 py-1 font-bold bg-gray-50 border-r text-gray-600">{stat}</td>
                            {columns.map(col => (
                                <td key={col} className="px-2 py-1 border-r font-mono">
                                    {data[col][stat] != null
                                        ? typeof data[col][stat] === 'number'
                                            ? data[col][stat].toFixed(2)
                                            : String(data[col][stat])
                                        : '-'}
                                </td>
                            ))}
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
            {results.map((r, i) => (
                <div
                    key={i}
                    className={`p-3 rounded-xl border ${r.model === bestModel
                            ? 'bg-indigo-50 border-indigo-200 ring-1 ring-indigo-300'
                            : 'bg-gray-50 border-gray-200'
                        }`}
                >
                    <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-2">
                            <span className="font-bold text-xs text-gray-800 uppercase">{r.model}</span>
                            {r.model === bestModel && (
                                <span className="text-[9px] bg-indigo-600 text-white px-1.5 py-0.5 rounded-full font-bold">BEST</span>
                            )}
                        </div>
                        <div className="text-xs font-mono font-bold text-indigo-700">
                            {r.accuracy != null && `Acc: ${r.accuracy}`}
                            {r.r2_score != null && `R²: ${r.r2_score}`}
                            {r.error && <span className="text-red-500">{r.error}</span>}
                        </div>
                    </div>

                    {(r.cv_mean != null || r.f1_score != null || r.mae != null) && (
                        <div className="flex gap-3 text-[10px] text-gray-500 mt-1">
                            {r.cv_mean != null && <span>CV: {r.cv_mean} ±{r.cv_std}</span>}
                            {r.f1_score != null && <span>F1: {r.f1_score}</span>}
                            {r.mae != null && <span>MAE: {r.mae}</span>}
                            {r.rmse != null && <span>RMSE: {r.rmse}</span>}
                        </div>
                    )}

                    {r.best_params && (
                        <div className="text-[10px] text-gray-400 mt-1 font-mono">
                            Params: {JSON.stringify(r.best_params)}
                        </div>
                    )}

                    {r.report && (
                        <pre className="mt-2 text-[9px] font-mono text-gray-600 bg-white p-2 rounded-sm border overflow-x-auto">
                            {r.report}
                        </pre>
                    )}
                </div>
            ))}
        </div>
    );
}
