import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, Legend,
} from 'recharts';
import { SERIES, GROUPS, OTHER, GRID, AXIS, tooltipStyle } from '../lib/chartColors';

const axisTick = { fontSize: 11, fill: AXIS };

/**
 * ChartRenderer — renders chart-ready evidence data using Recharts.
 */
export function ChartRenderer({ evidence }) {
    if (!evidence?.chart_type) return null;

    switch (evidence.chart_type) {
        case 'histogram':
        case 'bar':
            return (
                <BarChartCard
                    data={evidence.chart_data}
                    title={evidence.title}
                    xKey={evidence.x_key}
                    yKey={evidence.y_key}
                    categorical={evidence.chart_type === 'bar'}
                />
            );
        case 'boxplot':
            return <BoxPlotCard data={evidence.chart_data} title={evidence.title} />;
        case 'heatmap':
            return <CorrelationCard data={evidence.correlation} columns={evidence.columns} title={evidence.title} />;
        case 'scatter':
            return <ScatterCard data={evidence.chart_data} x={evidence.x} y={evidence.y} hue={evidence.hue} title={evidence.title} />;
        case 'missing_map':
            return <MissingMapCard evidence={evidence} />;
        default:
            return null;
    }
}

function ChartCard({ title, children, footer }) {
    return (
        <figure className="rounded-xl border border-line bg-surface p-4">
            {title && <figcaption className="mb-3 text-sm font-semibold text-ink">{title}</figcaption>}
            {children}
            {footer}
        </figure>
    );
}

// Histogram bars touch (they cover a continuous range); category bars stand apart, at a sensible width
function BarChartCard({ data, title, xKey = 'bin', yKey = 'count', categorical = false }) {
    if (!data?.length) return null;

    // Feature importance uses feature/importance keys
    const actualXKey = data[0]?.feature ? 'feature' : xKey;
    const actualYKey = data[0]?.importance != null ? 'importance' : yKey;

    return (
        <ChartCard title={title}>
            <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 4 }} barCategoryGap={categorical ? '25%' : 2}>
                    <CartesianGrid vertical={false} stroke={GRID} />
                    <XAxis
                        dataKey={actualXKey}
                        tick={axisTick}
                        tickLine={false}
                        axisLine={{ stroke: GRID }}
                        angle={-30}
                        textAnchor="end"
                        height={56}
                        interval="preserveStartEnd"
                    />
                    <YAxis tick={axisTick} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(235, 104, 52, 0.08)' }} />
                    <Bar dataKey={actualYKey} fill={SERIES} radius={[4, 4, 0, 0]} maxBarSize={categorical ? 56 : undefined} />
                </BarChart>
            </ResponsiveContainer>
        </ChartCard>
    );
}

function BoxPlotCard({ data, title }) {
    if (!data) return null;

    const stats = [
        { label: 'Min', value: data.min },
        { label: 'Q1', value: data.q1 },
        { label: 'Median', value: data.median },
        { label: 'Q3', value: data.q3 },
        { label: 'Max', value: data.max },
    ];

    const range = data.max - data.min;
    const pos = (val) => ((val - data.min) / (range || 1)) * 100;

    return (
        <ChartCard title={title}>
            <div className="relative mx-6 mb-5 h-12">
                <div
                    className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-3"
                    style={{ left: `${pos(data.min)}%`, right: `${100 - pos(data.max)}%` }}
                />
                <div
                    className="absolute top-1.5 bottom-1.5 rounded-md border-2"
                    style={{
                        left: `${pos(data.q1)}%`,
                        width: `${pos(data.q3) - pos(data.q1)}%`,
                        borderColor: SERIES,
                        background: 'rgba(235, 104, 52, 0.12)',
                    }}
                />
                <div className="absolute top-0.5 bottom-0.5 w-0.5 bg-ink" style={{ left: `${pos(data.median)}%` }} />
                <div className="absolute top-3 bottom-3 w-px bg-ink-3" style={{ left: `${pos(data.min)}%` }} />
                <div className="absolute top-3 bottom-3 w-px bg-ink-3" style={{ left: `${pos(data.max)}%` }} />
            </div>

            <div className="grid grid-cols-5 gap-2 text-center">
                {stats.map(s => (
                    <div key={s.label} className="rounded-lg bg-surface-2 px-2 py-1.5">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-3">{s.label}</div>
                        <div className="font-mono text-xs font-semibold tabular-nums text-ink">{s.value?.toFixed(2)}</div>
                    </div>
                ))}
            </div>

            {data.outlier_count > 0 && (
                <p className="mt-2.5 text-xs font-medium text-amber-800">
                    ⚠ {data.outlier_count} outlier{data.outlier_count > 1 ? 's' : ''} beyond the whiskers
                </p>
            )}
        </ChartCard>
    );
}

// Diverging scale: blue = move in opposite directions, gray = no relationship, orange = move together
function correlationCell(val) {
    if (val >= 0.7) return 'bg-orange-600 text-white';
    if (val >= 0.4) return 'bg-orange-300 text-ink';
    if (val >= 0.1) return 'bg-orange-100 text-ink';
    if (val > -0.1) return 'bg-surface-2 text-ink-3';
    if (val > -0.4) return 'bg-blue-100 text-ink';
    if (val > -0.7) return 'bg-blue-300 text-ink';
    return 'bg-blue-700 text-white';
}

function CorrelationCard({ data, columns, title }) {
    if (!data || !columns) return null;

    const legend = (
        <div className="mt-3 flex items-center gap-2 text-[11px] text-ink-3">
            <span>−1 opposite</span>
            <span className="flex overflow-hidden rounded">
                {['bg-blue-700', 'bg-blue-300', 'bg-blue-100', 'bg-surface-2', 'bg-orange-100', 'bg-orange-300', 'bg-orange-600'].map(c => (
                    <span key={c} className={`h-2.5 w-6 ${c}`} />
                ))}
            </span>
            <span>+1 together</span>
        </div>
    );

    return (
        <ChartCard title={title} footer={legend}>
            <div className="overflow-auto">
                <table className="text-[11px]">
                    <thead>
                        <tr>
                            <th />
                            {columns.map(c => (
                                <th key={c} className="max-w-18 truncate px-1 py-1 font-medium text-ink-3" title={c}>
                                    {c.slice(0, 9)}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {columns.map(row => (
                            <tr key={row}>
                                <td className="max-w-22.5 truncate pr-2 text-right font-medium text-ink-3" title={row}>
                                    {row}
                                </td>
                                {columns.map(col => {
                                    const v = data[row]?.[col] ?? 0;
                                    // A column against itself is always 1.00: show it, but keep it quiet
                                    const isSelf = row === col;
                                    return (
                                        <td key={col} className="p-0.5">
                                            <div
                                                className={`rounded-md px-1.5 py-1 text-center font-mono font-semibold tabular-nums ${
                                                    isSelf ? 'bg-surface text-ink-3/60 border border-dashed border-line' : correlationCell(v)
                                                }`}
                                                title={isSelf ? `${row} with itself` : `${row} × ${col}: ${v.toFixed(2)}`}
                                            >
                                                {isSelf ? '—' : v.toFixed(2)}
                                            </div>
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </ChartCard>
    );
}

/**
 * Where the blanks are, like seaborn's heatmap(df.isnull()): one column per data column, one thin
 * strip per row (a sample, in order). Two states only: blank (brand color) or filled (neutral).
 * Each column's blank share is written under its name, so nobody has to count cells.
 */
function MissingMapCard({ evidence }) {
    const { columns = [], missing_share: share = {}, rows = [], row_numbers: rowNumbers = [], total_rows: total, title } = evidence;
    if (!columns.length) return null;
    const grid = { gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))` };

    const legend = (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-3">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm" style={{ background: SERIES }} /> blank</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-surface-2" /> filled</span>
            <span>{rows.length < total ? `${rows.length} of ${total.toLocaleString()} training rows, evenly spaced` : `all ${total} training rows`}</span>
        </div>
    );

    return (
        <ChartCard title={title} footer={legend}>
            <div className="overflow-x-auto">
                <div style={{ minWidth: `${Math.max(columns.length * 28, 240)}px` }}>
                    <div className="grid gap-px" style={grid}>
                        {columns.map(c => (
                            <div key={c} className="flex min-w-0 flex-col items-center gap-0.5 pb-1.5" title={`${c}: ${(share[c] * 100).toFixed(1)}% blank`}>
                                <span className="w-full truncate text-center text-[10px] font-medium text-ink-2">{c}</span>
                                <span className={`text-[10px] tabular-nums ${share[c] > 0 ? 'font-semibold text-ink' : 'text-ink-3'}`}>
                                    {share[c] > 0 ? `${(share[c] * 100).toFixed(share[c] < 0.01 ? 1 : 0)}%` : '0'}
                                </span>
                            </div>
                        ))}
                    </div>
                    <div className="overflow-hidden rounded-md border border-line">
                        {rows.map((cells, i) => (
                            <div key={rowNumbers[i] ?? i} className="grid gap-px" style={grid}>
                                {cells.map((blank, j) => (
                                    <div
                                        key={j}
                                        className={blank ? '' : 'bg-surface-2'}
                                        style={{ height: 3, ...(blank ? { background: SERIES } : {}) }}
                                        title={blank ? `Row ${rowNumbers[i]}: ${columns[j]} is blank` : undefined}
                                    />
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </ChartCard>
    );
}

/** Keep the largest groups as their own color; fold the rest into "Other". */
function groupForColor(data, hue) {
    const groups = new Map();
    data.forEach(d => {
        const key = String(d[hue]);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(d);
    });
    const sorted = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
    const kept = sorted.slice(0, GROUPS.length).map(([name, rows], i) => ({ name, rows, color: GROUPS[i] }));
    const rest = sorted.slice(GROUPS.length).flatMap(([, rows]) => rows);
    if (rest.length) kept.push({ name: 'Other', rows: rest, color: OTHER });
    return kept;
}

function ScatterCard({ data, x, y, hue, title }) {
    if (!data?.length) return null;
    const groups = hue ? groupForColor(data, hue) : [];

    return (
        <ChartCard title={title}>
            <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
                    <CartesianGrid stroke={GRID} />
                    <XAxis dataKey={x} name={x} type="number" tick={axisTick} tickLine={false} axisLine={{ stroke: GRID }} />
                    <YAxis dataKey={y} name={y} type="number" tick={axisTick} tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ strokeDasharray: '3 3', stroke: AXIS }} contentStyle={tooltipStyle} />
                    {groups.length > 0 ? (
                        <>
                            <Legend iconType="circle" wrapperStyle={{ fontSize: 12, color: '#57534e' }} />
                            {groups.map(g => (
                                <Scatter key={g.name} name={g.name} data={g.rows} fill={g.color} fillOpacity={0.8} />
                            ))}
                        </>
                    ) : (
                        <Scatter data={data} fill={SERIES} fillOpacity={0.75} />
                    )}
                </ScatterChart>
            </ResponsiveContainer>
        </ChartCard>
    );
}
