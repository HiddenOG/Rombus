import React, { useState } from 'react';

const humanize = (s) => s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

const isColumnField = (spec) => /column/i.test(spec.description || '');

/**
 * StepEditor — a settings form generated from a tool's JSON schema.
 * Used to edit an existing step and to add a step by hand (no AI needed).
 */
export function StepEditor({ tool, initialArgs = {}, columns = [], onSave, onCancel, saveLabel = 'Save & replay', busy }) {
    const properties = Object.entries(tool.parameters?.properties || {}).filter(([name]) => name !== 'explanation');
    const required = new Set(tool.parameters?.required || []);
    const [values, setValues] = useState(() =>
        Object.fromEntries(properties.map(([name, spec]) => [name, toInput(initialArgs[name], spec)])),
    );
    const [problem, setProblem] = useState(null);

    const set = (name, value) => setValues(v => ({ ...v, [name]: value }));

    const submit = (e) => {
        e.preventDefault();
        try {
            const args = {};
            for (const [name, spec] of properties) {
                const value = fromInput(values[name], spec);
                if (value === undefined) {
                    if (required.has(name)) throw new Error(`"${humanize(name)}" is required.`);
                    continue;
                }
                args[name] = value;
            }
            setProblem(null);
            onSave(args);
        } catch (err) {
            setProblem(err.message);
        }
    };

    return (
        <form onSubmit={submit} className="space-y-3 rounded-xl border border-brand-200 bg-brand-50/40 p-3.5">
            {properties.length === 0 && <p className="text-sm text-ink-3">This step has no settings.</p>}
            {properties.map(([name, spec]) => (
                <Field
                    key={name}
                    name={name}
                    spec={spec}
                    required={required.has(name)}
                    value={values[name]}
                    columns={columns}
                    onChange={v => set(name, v)}
                />
            ))}
            {problem && <p className="text-xs font-medium text-red-700">{problem}</p>}
            <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={onCancel} className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-2">
                    Cancel
                </button>
                <button
                    type="submit"
                    disabled={busy}
                    className="rounded-lg bg-brand-500 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
                >
                    {saveLabel}
                </button>
            </div>
        </form>
    );
}

function Field({ name, spec, required, value, columns, onChange }) {
    const label = (
        <span className="mb-1 block text-xs font-semibold text-ink-2">
            {humanize(name)} {required ? <span className="text-brand-600">*</span> : <span className="font-normal text-ink-3">(optional)</span>}
        </span>
    );
    const input = 'w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-sm text-ink outline-none focus:border-brand-300 focus:ring-2 focus:ring-brand-100';

    if (spec.enum) {
        return (
            <label className="block">
                {label}
                <select className={input} value={value} onChange={e => onChange(e.target.value)}>
                    <option value="">{required ? 'Choose…' : 'Default'}</option>
                    {spec.enum.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
            </label>
        );
    }

    if (spec.type === 'array') {
        const options = spec.items?.enum || (isColumnField(spec) ? columns : null);
        if (options) {
            const chosen = new Set(value);
            const all = [...new Set([...options, ...value])];
            return (
                <fieldset>
                    {label}
                    <div className="flex flex-wrap gap-1.5">
                        {all.map(o => (
                            <button
                                type="button"
                                key={o}
                                onClick={() => onChange(chosen.has(o) ? value.filter(v => v !== o) : [...value, o])}
                                className={`rounded-lg border px-2 py-1 font-mono text-xs transition-colors ${
                                    chosen.has(o) ? 'border-brand-400 bg-brand-500 text-white' : 'border-line bg-surface text-ink-2 hover:border-brand-300'
                                }`}
                            >
                                {o}
                            </button>
                        ))}
                    </div>
                </fieldset>
            );
        }
    }

    if (spec.type === 'string' && isColumnField(spec)) {
        const all = [...new Set([...columns, ...(value ? [value] : [])])];
        return (
            <label className="block">
                {label}
                <select className={input} value={value} onChange={e => onChange(e.target.value)}>
                    <option value="">Choose a column…</option>
                    {all.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
            </label>
        );
    }

    if (spec.type === 'boolean') {
        return (
            <label className="flex items-center gap-2 text-sm text-ink-2">
                <input type="checkbox" checked={value} onChange={e => onChange(e.target.checked)} className="accent-orange-500" />
                {humanize(name)}
            </label>
        );
    }

    if (spec.format === 'code') {
        return (
            <label className="block">
                {label}
                <textarea
                    className={`${input} font-mono text-[12px] leading-[1.45]`}
                    rows={16}
                    spellCheck={false}
                    value={value}
                    onChange={e => onChange(e.target.value)}
                />
                <span className="mt-1 block text-xs text-ink-3">
                    Runs on your machine with full access, like a Jupyter cell. It will be marked "not verified".
                </span>
            </label>
        );
    }

    if (spec.type === 'object') {
        return (
            <label className="block">
                {label}
                <textarea
                    className={`${input} font-mono text-xs`}
                    rows={4}
                    value={value}
                    onChange={e => onChange(e.target.value)}
                    placeholder='{"column": value}'
                />
            </label>
        );
    }

    return (
        <label className="block">
            {label}
            <input
                className={input}
                type={spec.type === 'number' || spec.type === 'integer' ? 'number' : 'text'}
                step={spec.type === 'integer' ? 1 : 'any'}
                value={value}
                onChange={e => onChange(e.target.value)}
                placeholder={spec.description}
            />
        </label>
    );
}

/** Stored argument -> form value */
function toInput(value, spec) {
    if (spec.type === 'array') return Array.isArray(value) ? value : [];
    if (spec.type === 'boolean') return Boolean(value);
    if (spec.type === 'object') return value ? JSON.stringify(value, null, 2) : '';
    if (value == null && spec.default != null) return String(spec.default);
    return value == null ? '' : String(value);
}

/** Form value -> argument (undefined means "leave it out, use the default") */
function fromInput(value, spec) {
    if (spec.type === 'array') return value.length ? value : undefined;
    if (spec.type === 'boolean') return value;
    if (value === '') return undefined;
    if (spec.type === 'number' || spec.type === 'integer') {
        const n = Number(value);
        if (Number.isNaN(n)) throw new Error(`"${value}" isn't a number.`);
        return n;
    }
    if (spec.type === 'object') {
        try {
            return JSON.parse(value);
        } catch {
            throw new Error('Features must be valid JSON, like {"age": 30}.');
        }
    }
    return value;
}
