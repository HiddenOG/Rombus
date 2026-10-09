import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

const KEYWORDS = /\b(def|return|for|in|if|else|not|continue|import|as|and|or|None|True|False)\b/;
const TOKEN = /(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:def|return|for|in|if|else|not|continue|import|as|and|or|None|True|False)\b|\b\d+(?:\.\d+)?\b)/g;

/** Color one line of Python: comments, strings, keywords, numbers. Builds React nodes, no HTML injection. */
function highlight(line, key) {
    const parts = [];
    let last = 0;
    for (const match of line.matchAll(TOKEN)) {
        if (match.index > last) parts.push(line.slice(last, match.index));
        const t = match[0];
        const className = t.startsWith('#')
            ? 'text-ink-3 italic'
            : t.startsWith('"') || t.startsWith("'")
                ? 'text-emerald-700'
                : KEYWORDS.test(t)
                    ? 'text-brand-700 font-semibold'
                    : 'text-blue-700';
        parts.push(<span key={match.index} className={className}>{t}</span>);
        last = match.index + t.length;
    }
    if (last < line.length) parts.push(line.slice(last));
    return <div key={key} className="min-h-[1.4em]">{parts}</div>;
}

/**
 * CodeView — the Python behind a step: learn() reads the training rows, apply() uses what it learned.
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
        <div className="relative rounded-xl border border-line bg-surface-2">
            <button
                onClick={copy}
                className="absolute right-2 top-2 flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-[11px] font-medium text-ink-2 hover:text-ink"
            >
                {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                {copied ? 'Copied' : 'Copy'}
            </button>
            <pre className="max-h-96 overflow-auto p-3.5 pr-20 font-mono text-[12px] leading-[1.4] text-ink custom-scrollbar">
                {code.split('\n').map((line, i) => highlight(line, i))}
            </pre>
        </div>
    );
}
