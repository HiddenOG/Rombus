import React, { useState } from 'react';

const INDENT = '    ';

/**
 * CodeEditor — a plain monospace editor for step code. Tab indents, Shift+Tab dedents.
 */
export function CodeEditor({ initialCode, onSave, onCancel, busy, saveLabel = 'Save & replay' }) {
    const [code, setCode] = useState(initialCode);

    const onKeyDown = (e) => {
        if (e.key !== 'Tab') return;
        e.preventDefault();
        const el = e.target;
        const { selectionStart: start, selectionEnd: end } = el;
        const lineStart = code.lastIndexOf('\n', start - 1) + 1;
        if (e.shiftKey) {
            if (code.startsWith(INDENT, lineStart)) {
                setCode(code.slice(0, lineStart) + code.slice(lineStart + INDENT.length));
                requestAnimationFrame(() => el.setSelectionRange(Math.max(lineStart, start - INDENT.length), Math.max(lineStart, end - INDENT.length)));
            }
            return;
        }
        setCode(code.slice(0, start) + INDENT + code.slice(end));
        requestAnimationFrame(() => el.setSelectionRange(start + INDENT.length, start + INDENT.length));
    };

    return (
        <div className="space-y-2.5">
            <textarea
                value={code}
                onChange={e => setCode(e.target.value)}
                onKeyDown={onKeyDown}
                spellCheck={false}
                rows={Math.min(28, Math.max(10, code.split('\n').length + 1))}
                className="w-full resize-y rounded-xl border border-brand-200 bg-surface p-3.5 font-mono text-[12px] leading-[1.45] text-ink outline-none focus:border-brand-400 focus:ring-4 focus:ring-brand-100"
            />
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
                <p className="text-xs text-ink-3">
                    Runs on your machine with full access, like a Jupyter cell. Saving replays every later step.
                </p>
                <div className="flex shrink-0 gap-2">
                    <button type="button" onClick={onCancel} className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-2">
                        Cancel
                    </button>
                    <button
                        type="button"
                        disabled={busy}
                        onClick={() => onSave(code)}
                        className="rounded-lg bg-brand-500 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
                    >
                        {saveLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
