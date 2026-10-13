import React, { Suspense, lazy } from 'react';

// The editor (CodeMirror, ~120 KB) loads the first time code is on screen, not with the page
const CodeMirrorEditor = lazy(() => import('./CodeMirrorEditor'));

/**
 * PythonEditor — the notebook's code editor: Python colours, line numbers, auto-indent, bracket
 * matching, undo/redo; Shift+Enter runs, Escape leaves. readOnly shows code the same way.
 * Until the editor has loaded, the code shows as plain text.
 */
export function PythonEditor(props) {
    const fallback = (
        <pre className="m-0 overflow-auto px-4 py-2.5 font-mono text-[12.5px] leading-[1.55] text-ink" style={{ maxHeight: props.maxHeight ?? '32rem' }}>
            {props.value}
        </pre>
    );
    return (
        <Suspense fallback={fallback}>
            <CodeMirrorEditor {...props} />
        </Suspense>
    );
}
