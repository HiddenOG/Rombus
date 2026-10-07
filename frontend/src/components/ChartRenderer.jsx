import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, Cell, Legend,
} from 'recharts';

const COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#4f46e5', '#4338ca', '#3730a3'];

/**
 * ChartRenderer — renders chart-ready evidence data using Recharts.
 */
export function ChartRenderer({ evidence }) {
    if (!evidence?.chart_type) return null;

    switch (evidence.chart_type) {
        case 'histogram':
        case 'bar':
            return <HistogramChart data={evidence.chart_data} title={evidence.title} xKey={evidence.x_key} yKey={evidence.y_key} />;
        case 'boxplot':
            return <BoxPlotDisplay data={evidence.chart_data} title={evidence.title} />;
        case 'heatmap':
            return <CorrelationDisplay data={evidence.correlation} columns={evidence.columns} title={evidence.title} />;
        case 'scatter':
            return <ScatterPlot data={evidence.chart_data} x={evidence.x} y={evidence.y} hue={evidence.hue} title={evidence.title} />;
        default:
            return null;
    }
}

function HistogramChart({ data, title, xKey = 'bin', yKey = 'count' }) {
    if (!data?.length) return null;

    // For feature importance, use feature/importance keys
    const actualXKey = data[0]?.feature ? 'feature' : xKey;
    const actualYKey = data[0]?.importance ? 'importance' : yKey;

    return (
        <div className="mt-3 bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
            {title && <p className="text-xs font-bold text-gray-600 mb-3 uppercase tracking-wider">{title}</p>}
            <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis
                        dataKey={actualXKey}
                        tick={{ fontSize: 10, fill: '#6b7280' }}
                        angle={-30}
                        textAnchor="end"
                        height={60}
                    />
                    <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#1f2937',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            fontSize: '11px',
                        }}
                    />
                    <Bar dataKey={actualYKey} radius={[4, 4, 0, 0]}>
                        {data.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}

function BoxPlotDisplay({ data, title }) {
    if (!data) return null;

    const stats = [
        { label: 'Min', value: data.min },
        { label: 'Q1', value: data.q1 },
        { label: 'Median', value: data.median },
        { label: 'Q3', value: data.q3 },
        { label: 'Max', value: data.max },
    ];

    // Compute IQR and range for the visual
    const range = data.max - data.min;
    const getPos = (val) => ((val - data.min) / (range || 1)) * 100;

    return (
        <div className="mt-3 bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
            {title && <p className="text-xs font-bold text-gray-600 mb-3 uppercase tracking-wider">{title}</p>}

            {/* Visual box plot */}
            <div className="relative h-12 mx-8 mb-4">
                {/* Whisker line */}
                <div
                    className="absolute top-1/2 -translate-y-1/2 h-0.5 bg-gray-400"
                    style={{ left: `${getPos(data.min)}%`, right: `${100 - getPos(data.max)}%` }}
                />
                {/* Box */}
                <div
                    className="absolute top-1 bottom-1 bg-indigo-100 border-2 border-indigo-400 rounded"
                    style={{ left: `${getPos(data.q1)}%`, width: `${getPos(data.q3) - getPos(data.q1)}%` }}
                />
                {/* Median line */}
                <div
                    className="absolute top-0.5 bottom-0.5 w-0.5 bg-indigo-700"
                    style={{ left: `${getPos(data.median)}%` }}
                />
                {/* Min/max whisker caps */}
                <div className="absolute top-2 bottom-2 w-0.5 bg-gray-400" style={{ left: `${getPos(data.min)}%` }} />
                <div className="absolute top-2 bottom-2 w-0.5 bg-gray-400" style={{ left: `${getPos(data.max)}%` }} />
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-5 gap-2 text-center">
                {stats.map(s => (
                    <div key={s.label} className="bg-gray-50 rounded-lg p-2">
                        <div className="text-[10px] text-gray-400 uppercase font-bold">{s.label}</div>
                        <div className="text-xs font-mono font-bold text-gray-700">{s.value?.toFixed(2)}</div>
                    </div>
                ))}
            </div>

            {data.outlier_count > 0 && (
                <p className="mt-2 text-[10px] text-orange-600 font-medium">
                    ⚠ {data.outlier_count} outlier{data.outlier_count > 1 ? 's' : ''} detected
                </p>
            )}
        </div>
    );
}

function CorrelationDisplay({ data, columns, title }) {
    if (!data || !columns) return null;

    const getColor = (val) => {
        if (val >= 0.7) return 'bg-indigo-600 text-white';
        if (val >= 0.4) return 'bg-indigo-300 text-indigo-900';
        if (val >= 0) return 'bg-indigo-50 text-indigo-700';
        if (val >= -0.4) return 'bg-red-50 text-red-700';
        if (val >= -0.7) return 'bg-red-300 text-red-900';
        return 'bg-red-600 text-white';
    };

    return (
        <div className="mt-3 bg-white rounded-xl border border-gray-100 p-4 shadow-sm overflow-auto">
            {title && <p className="text-xs font-bold text-gray-600 mb-3 uppercase tracking-wider">{title}</p>}
            <table className="text-[10px]">
                <thead>
                    <tr>
                        <th className="px-1 py-1"></th>
                        {columns.map(c => (
                            <th key={c} className="px-1 py-1 font-medium text-gray-500 max-w-[60px] truncate" title={c}>
                                {c.slice(0, 8)}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {columns.map(row => (
                        <tr key={row}>
                            <td className="px-1 py-1 font-medium text-gray-500 max-w-[60px] truncate" title={row}>
                                {row.slice(0, 8)}
                            </td>
                            {columns.map(col => (
                                <td key={col} className="px-1 py-0.5">
                                    <div className={`px-1.5 py-0.5 rounded text-center font-mono font-bold ${getColor(data[row]?.[col] || 0)}`}>
                                        {(data[row]?.[col] || 0).toFixed(2)}
                                    </div>
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function ScatterPlot({ data, x, y, hue, title }) {
    if (!data?.length) return null;

    // Group by hue if provided
    const groups = {};
    if (hue) {
        data.forEach(d => {
            const key = String(d[hue]);
            if (!groups[key]) groups[key] = [];
            groups[key].push(d);
        });
    }

    return (
        <div className="mt-3 bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
            {title && <p className="text-xs font-bold text-gray-600 mb-3 uppercase tracking-wider">{title}</p>}
            <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey={x} name={x} tick={{ fontSize: 10, fill: '#6b7280' }} />
                    <YAxis dataKey={y} name={y} tick={{ fontSize: 10, fill: '#6b7280' }} />
                    <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        contentStyle={{
                            backgroundColor: '#1f2937',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            fontSize: '11px',
                        }}
                    />
                    {hue && Object.keys(groups).length > 0 ? (
                        <>
                            <Legend />
                            {Object.entries(groups).map(([key, groupData], i) => (
                                <Scatter key={key} name={key} data={groupData} fill={COLORS[i % COLORS.length]} />
                            ))}
                        </>
                    ) : (
                        <Scatter data={data} fill="#6366f1" />
                    )}
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
}
