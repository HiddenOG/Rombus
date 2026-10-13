import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { PythonEditor } from './PythonEditor';

/**
 * CodeView — the Python behind a step (learn() reads the training rows, apply() uses what it learned),
 * read-only, coloured the same as the editor.
 */
export function CodeView({ code }) {
    const [copied, setCopied] = useState(false);

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            // Clipboard blocked: the code is still selectable by hand
        }
    };

    return (
        <div className="relative overflow-hidden rounded-xl border border-line">
            <button
                onClick={copy}
                className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-[11px] font-medium text-ink-2 hover:text-ink"
            >
                {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                {copied ? 'Copied' : 'Copy'}
            </button>
            <PythonEditor value={code} readOnly label="Code (read only)" minLines={1} maxHeight="24rem" />
        </div>
    );
}
