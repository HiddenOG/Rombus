import React, { useCallback, useEffect, useState } from 'react';
import { Trash2, RotateCw, X } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';

/*
 * Cell selection UI (state lives in hooks/useCellSelection). Deleting recipe-step cells undoes
 * those steps together, as one change.
 */

const INTERACTIVE = 'button, a, input, textarea, select, label, summary, [role="menuitem"], [contenteditable="true"], pre, code';

/** Makes a cell selectable: click its background to select; selected cells get an orange outline. */
export function Selectable({ id, selection, children }) {
    const isSelected = selection.selected.includes(id);
    const onClick = (e) => {
        // Clicks on buttons, inputs, code or text being selected keep their normal meaning
        if (e.target.closest(INTERACTIVE) || window.getSelection()?.toString()) return;
        selection.click(id, e);
    };
    return (
        <div
            id={`cell-${id}`}
            onClick={onClick}
            aria-selected={isSelected}
            className={`rounded-2xl transition-shadow ${isSelected ? 'ring-2 ring-brand-400 ring-offset-2 ring-offset-canvas' : ''}`}
        >
            {children}
        </div>
    );
}

/** What can be done with the selected cells, plus keyboard shortcuts. */
export function SelectionBar({ nb, selection }) {
    const [confirming, setConfirming] = useState(false);
    const { selected } = selection;
    const chosen = nb.cells.filter(c => selected.includes(c.id));
    const steps = chosen.filter(c => c.stepId && nb.summary?.recipe?.some(s => s.id === c.stepId));
    const views = chosen.filter(c => c.type === 'run' && nb.tools[c.tool]?.kind === 'view');
    const cancel = useCallback(() => setConfirming(false), []);

    useEffect(() => {
        const onKey = (e) => {
            if (e.target.closest?.('input, textarea, select, [contenteditable="true"]') || confirming) return;
            if (e.key === 'Escape' && selected.length) selection.clear();
            else if (e.key === 'ArrowDown' && selected.length) { e.preventDefault(); selection.step(1); }
            else if (e.key === 'ArrowUp' && selected.length) { e.preventDefault(); selection.step(-1); }
            else if ((e.key === 'Delete' || e.key === 'Backspace') && selected.length && !nb.isProcessing) {
                e.preventDefault();
                setConfirming(true);
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [selected, selection, confirming, nb.isProcessing]);

    if (!selected.length) return null;

    return (
        <>
            <div className="mx-auto mb-2 flex max-w-3xl flex-wrap items-center gap-2 rounded-xl border border-brand-200 bg-brand-50 px-3 py-2 text-sm animate-in">
                <span className="font-semibold text-brand-800">{selected.length} selected</span>
                <span className="hidden text-xs text-ink-3 sm:inline">Shift/Ctrl + click to select more · ↑ ↓ · Delete · Esc</span>
                <span className="flex-1" />
                {views.length > 0 && (
                    <button
                        onClick={() => views.forEach(c => nb.rerun(c))}
                        disabled={nb.isProcessing}
                        className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-semibold text-ink-2 hover:bg-surface disabled:opacity-50"
                    >
                        <RotateCw className="h-3.5 w-3.5" /> Rerun {views.length === 1 ? 'view' : `${views.length} views`}
                    </button>
                )}
                <button
                    onClick={() => setConfirming(true)}
                    disabled={nb.isProcessing}
                    className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-semibold text-red-700 hover:bg-red-50 disabled:opacity-50"
                >
                    <Trash2 className="h-3.5 w-3.5" /> {steps.length ? 'Undo & delete' : 'Delete'}
                </button>
                <button onClick={selection.clear} aria-label="Clear selection" className="rounded-lg p-1 text-ink-3 hover:bg-surface hover:text-ink">
                    <X className="h-4 w-4" />
                </button>
            </div>

            <ConfirmDialog
                open={confirming}
                title={`Delete ${selected.length} cell${selected.length > 1 ? 's' : ''}?`}
                confirmLabel={steps.length ? `Undo ${steps.length} step${steps.length > 1 ? 's' : ''} & delete` : 'Delete'}
                busy={nb.isProcessing}
                onCancel={cancel}
                onConfirm={async () => {
                    setConfirming(false);
                    if (await nb.deleteCells(selected)) selection.clear();
                }}
            >
                {steps.length > 0 ? (
                    <p>
                        This undoes {steps.length === 1 ? 'a recipe step' : `${steps.length} recipe steps`} (
                        {steps.map(c => c.tool.replace(/_/g, ' ')).join(', ')}) in one go, and the rest of the recipe
                        replays from the raw data. If a remaining step depends on one of these, nothing changes and
                        you&apos;ll see why.
                    </p>
                ) : (
                    <p>These cells only show results or questions; your data and recipe don&apos;t change.</p>
                )}
            </ConfirmDialog>
        </>
    );
}
