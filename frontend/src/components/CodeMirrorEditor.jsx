import React, { useEffect, useRef } from 'react';
import { EditorState, Prec } from '@codemirror/state';
import {
    EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { syntaxHighlighting, HighlightStyle, indentOnInput, bracketMatching, indentUnit } from '@codemirror/language';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { python } from '@codemirror/lang-python';
import { tags as t } from '@lezer/highlight';

/*
  Python colours in the app's palette (checked for contrast on white): keywords in the brand orange,
  strings green, numbers blue, function and class names violet, comments a muted grey.
*/
const colours = HighlightStyle.define([
    { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: '#c2410c', fontWeight: '600' },
    { tag: [t.string, t.special(t.string)], color: '#047857' },
    { tag: [t.number, t.bool, t.null], color: '#1d4ed8' },
    { tag: [t.function(t.variableName), t.function(t.propertyName), t.definition(t.function(t.variableName)), t.className], color: '#6d28d9' },
    { tag: t.comment, color: '#78716c', fontStyle: 'italic' },
    { tag: [t.self, t.special(t.variableName)], color: '#b45309' },
    { tag: [t.operator, t.punctuation, t.bracket], color: '#57534e' },
]);

const look = EditorView.theme({
    '&': { fontSize: '12.5px', backgroundColor: 'var(--color-surface)', color: 'var(--color-ink)' },
    '&.cm-focused': { outline: 'none' },
    '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.55' },
    '.cm-content': { padding: '10px 0', caretColor: 'var(--color-brand-600)' },
    '.cm-gutters': { backgroundColor: 'var(--color-surface-2)', color: 'var(--color-ink-3)', border: 'none', borderRight: '1px solid var(--color-line)' },
    '.cm-lineNumbers .cm-gutterElement': { padding: '0 10px 0 12px', minWidth: '32px' },
    '.cm-activeLine': { backgroundColor: 'rgb(255 247 237 / 0.6)' },
    '.cm-activeLineGutter': { backgroundColor: 'var(--color-brand-50)', color: 'var(--color-brand-700)' },
    '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': { backgroundColor: 'var(--color-orange-200) !important' },
    '.cm-matchingBracket': { backgroundColor: 'var(--color-brand-100)', outline: '1px solid var(--color-brand-300)' },
    '.cm-cursor': { borderLeftColor: 'var(--color-brand-600)', borderLeftWidth: '2px' },
});

/**
 * PythonEditor — the notebook's code editor (CodeMirror 6): Python colours, line numbers (errors say
 * "line 14"), auto-indent, bracket matching, undo/redo. Shift+Enter runs, Escape leaves.
 * With readOnly it's the same view without editing, so code looks the same everywhere.
 */
export default function CodeMirrorEditor({ value, onChange, onRun, onEscape, readOnly = false, label = 'Code', minLines = 4, maxHeight = '32rem' }) {
    const host = useRef(null);
    const view = useRef(null);
    // The latest callbacks, without rebuilding the editor every render
    const handlers = useRef({ onChange, onRun, onEscape });
    useEffect(() => { handlers.current = { onChange, onRun, onEscape }; });

    useEffect(() => {
        const keys = Prec.highest(keymap.of([
            { key: 'Shift-Enter', run: () => { handlers.current.onRun?.(); return Boolean(handlers.current.onRun); } },
            { key: 'Escape', run: () => { handlers.current.onEscape?.(); return Boolean(handlers.current.onEscape); } },
        ]));
        const editing = readOnly
            ? [EditorState.readOnly.of(true), EditorView.editable.of(false)]
            : [history(), drawSelection(), indentOnInput(), closeBrackets(), highlightActiveLine(), highlightActiveLineGutter(),
                keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab])];
        view.current = new EditorView({
            parent: host.current,
            state: EditorState.create({
                doc: value ?? '',
                extensions: [
                    keys, lineNumbers(), python(), syntaxHighlighting(colours), bracketMatching(), look,
                    indentUnit.of('    '), EditorState.tabSize.of(4), ...editing,
                    EditorView.contentAttributes.of({ 'aria-label': label, spellcheck: 'false' }),
                    EditorView.theme({ '.cm-content': { minHeight: `${minLines * 1.55}em` }, '.cm-scroller': { maxHeight } }),
                    EditorView.updateListener.of(update => {
                        if (update.docChanged) handlers.current.onChange?.(update.state.doc.toString());
                    }),
                ],
            }),
        });
        return () => view.current?.destroy();
        // Built once per mount; value changes from outside are applied below
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [readOnly]);

    // Code replaced from outside (a template for another kind of cell, a reverted step): show it
    useEffect(() => {
        const current = view.current?.state.doc.toString();
        if (view.current && value != null && value !== current) {
            view.current.dispatch({ changes: { from: 0, to: current.length, insert: value } });
        }
    }, [value]);

    return <div ref={host} className="overflow-hidden" />;
}
