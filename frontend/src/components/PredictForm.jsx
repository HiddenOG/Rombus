import React, { useState } from 'react';

/**
 * PredictForm — one field per original column, prefilled with a typical value from the
 * training rows. The values go through the same recipe as the training data before the model sees them.
 */
export function PredictForm({ fields, models, onPredict, onCancel, busy }) {
    const [values, setValues] = useState(() => Object.fromEntries(fields.map(f => [f.name, f.typical ?? ''])));
    const [model, setModel] = useState('');

    const submit = (e) => {
        e.preventDefault();
        const features = {};
        for (const f of fields) {
            const v = values[f.name];
            if (v === '' || v == null) continue; // left blank: the recipe fills it like any blank
            features[f.name] = f.kind === 'number' ? Number(v) : v;
        }
        onPredict({ features, ...(model ? { algorithm: model } : {}) });
    };

    const input = 'w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-sm text-ink outline-none focus:border-brand-300 focus:ring-2 focus:ring-brand-100';

    return (
        <form onSubmit={submit} className="space-y-3">
            <p className="text-xs text-ink-3">
                Describe a new case. Fields start at a typical training value; clear one to leave it blank
                (the recipe fills blanks the same way it did for training).
            </p>
            <div className="grid max-h-72 grid-cols-1 gap-x-3 gap-y-2.5 overflow-y-auto pr-1 custom-scrollbar sm:grid-cols-2">
                {fields.map(f => (
                    <label key={f.name} className="block">
                        <span className="mb-1 block truncate text-xs font-semibold text-ink-2" title={f.name}>{f.name}</span>
                        {f.kind === 'number' ? (
                            <input
                                type="number"
                                step="any"
                                className={input}
                                value={values[f.name]}
                                placeholder={f.min != null ? `${f.min} – ${f.max}` : ''}
                                onChange={e => setValues(v => ({ ...v, [f.name]: e.target.value }))}
                            />
                        ) : (
                            <select
                                className={input}
                                value={values[f.name]}
                                onChange={e => setValues(v => ({ ...v, [f.name]: e.target.value }))}
                            >
                                <option value="">(blank)</option>
                                {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                            </select>
                        )}
                    </label>
                ))}
            </div>
            <div className="flex items-center justify-between gap-3 pt-1">
                <label className="flex items-center gap-2 text-xs text-ink-2">
                    Model
                    <select className="rounded-lg border border-line bg-surface px-2 py-1 text-xs" value={model} onChange={e => setModel(e.target.value)}>
                        <option value="">Best</option>
                        {models.map(m => <option key={m} value={m}>{m.toUpperCase()}</option>)}
                    </select>
                </label>
                <div className="flex gap-2">
                    <button type="button" onClick={onCancel} className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-2">
                        Cancel
                    </button>
                    <button
                        type="submit"
                        disabled={busy}
                        className="rounded-lg bg-brand-500 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
                    >
                        Predict
                    </button>
                </div>
            </div>
        </form>
    );
}
