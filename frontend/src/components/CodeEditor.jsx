import React, { useState } from 'react';
import { PythonEditor } from './PythonEditor';

const Key = ({ children }) => <kbd className="rounded border border-line bg-surface-2 px-1 font-mono text-[10px]">{children}</kbd>;

/**
 * CodeEditor — edit a step's code (or replace a view's), then save to run it. Shift+Enter saves and runs,
 * Escape cancels.
 */
export function CodeEditor({ initialCode, onSave, onCancel, busy, saveLabel = 'Save & replay' }) {
    const [code, setCode] = useState(initialCode);
    const save = () => { if (!busy) onSave(code); };

    return (
        <div className="space-y-2.5">
            <div className="overflow-hidden rounded-xl border border-brand-200 focus-within:border-brand-400 focus-within:ring-4 focus-within:ring-brand-100">
                <PythonEditor value={code} onChange={setCode} onRun={save} onEscape={onCancel} minLines={10} />
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
                <p className="text-xs text-ink-3">
                    <Key>Shift</Key>+<Key>Enter</Key> runs, <Key>Esc</Key> cancels.
                </p>
                <div className="flex shrink-0 gap-2">
                    <button type="button" onClick={onCancel} className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-2">
                        Cancel
                    </button>
                    <button
                        type="button"
                        disabled={busy}
                        onClick={save}
                        className="rounded-lg bg-brand-500 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
                    >
                        {saveLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
