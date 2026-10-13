import React, { useCallback, useEffect, useState } from 'react';
import { Trash2, RotateCw, X, Undo2 } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';

/*
 * Cell selection UI (state lives in hooks/useCellSelection). Deleting recipe-step cells undoes
 * those steps together, as one change.
 */

const INTERACTIVE = 'button, a, input, textarea, select, label, summary, [role="menuitem"], [contenteditable="true"], pre, code';

/**
 * Makes a cell selectable: click its background to select. A selected cell gets a thin orange bar on
 * its left (like Jupyter's active cell); when it's the only one selected, its actions appear right
 * on it: Undo step (for cells that changed the data) and Delete.
 */
export function Selectable({ cell, nb, selection, children }) {
    const [confirming, setConfirming] = useState(false);
    const isSelected = selection.selected.includes(cell.id);
    const only = isSelected && selection.selected.length === 1;
    const isStep = Boolean(cell.stepId && nb.summary?.recipe?.some(s => s.id === cell.stepId));
    const cancel = useCallback(() => setConfirming(false), []);

    const onClick = (e) => {
        // Clicks on buttons, inputs, code or text being selected keep their normal meaning
        if (e.target.closest(INTERACTIVE) || window.getSelection()?.toString()) return;
        selection.click(cell.id, e);
    };
    return (
        <div id={`cell-${cell.id}`} onClick={onClick} aria-selected={isSelected} className="relative">
            {isSelected && <span aria-hidden="true" className="absolute -left-2 top-1 bottom-1 w-1 rounded-full bg-brand-500" />}
            {only && (
                <div className="mb-1.5 flex justify-end gap-1.5 animate-in">
                    {isStep && (
                        <CellAction onClick={() => nb.undoStep(cell)} disabled={nb.isProcessing} title="Remove this step from the recipe and rebuild the data without it. The cell stays, greyed out.">
                            <Undo2 className="h-3.5 w-3.5" /> Undo step
                        </CellAction>
                    )}
                    <CellAction onClick={() => setConfirming(true)} disabled={nb.isProcessing} danger>
                        <Trash2 className="h-3.5 w-3.5" /> Delete
                    </CellAction>
                </div>
            )}
            <div className={`rounded-2xl transition-colors ${isSelected ? 'bg-brand-50/50' : ''}`}>{children}</div>

            <ConfirmDialog
                open={confirming}
                title="Delete this cell?"
                confirmLabel={isStep ? 'Undo step & delete' : 'Delete'}
                busy={nb.isProcessing}
                onCancel={cancel}
                onConfirm={() => { setConfirming(false); nb.deleteCell(cell); selection.clear(); }}
            >
                <p>
                    {isStep
                        ? 'This also undoes its step: the data is rebuilt without it. To keep the cell for reference, use Undo step instead.'
                        : "This cell only shows a result or a question; your data and recipe don't change."}
                </p>
            </ConfirmDialog>
        </div>
    );
}

function CellAction({ children, danger, ...props }) {
    return (
        <button
            type="button"
            {...props}
            className={`flex items-center gap-1.5 rounded-lg border bg-surface px-2.5 py-1 text-xs font-semibold shadow-sm transition-colors disabled:opacity-50 ${
                danger ? 'border-red-200 text-red-700 hover:bg-red-50' : 'border-line text-ink-2 hover:border-brand-300 hover:text-brand-700'
            }`}
        >
            {children}
        </button>
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
            {/* One cell: its actions sit on the cell itself. Several: this bar acts on all of them. */}
            {selected.length > 1 && (
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
            )}

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
