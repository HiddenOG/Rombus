import React, { useEffect, useRef, useState } from 'react';
import { Cpu, ChevronDown, Play, RotateCcw, FastForward, Eraser } from 'lucide-react';
import { ConfirmDialog } from './ConfirmDialog';

/**
 * KernelMenu — Jupyter-style notebook controls: run all, restart, restart & run all, clear outputs.
 */
export function KernelMenu({ nb }) {
    const [open, setOpen] = useState(false);
    const [confirm, setConfirm] = useState(null); // 'restart' | 'restartRun'
    const ref = useRef(null);
    const disabled = !nb.sessionId || nb.isProcessing;

    // Close when clicking anywhere outside the menu
    useEffect(() => {
        if (!open) return undefined;
        const onClick = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [open]);

    const choose = (fn) => () => { setOpen(false); fn(); };
    const items = [
        { icon: Play, label: 'Run all', text: 'Re-run every chart and retrain models on the current data.', onClick: choose(() => nb.runAll(false)) },
        { icon: RotateCcw, label: 'Restart', text: "Clear trained models and the AI's memory; rebuild the data from scratch.", onClick: choose(() => setConfirm('restart')) },
        { icon: FastForward, label: 'Restart & run all', text: 'Restart, then re-run everything top to bottom. Proves the notebook is reproducible.', onClick: choose(() => setConfirm('restartRun')) },
        { icon: Eraser, label: 'Clear outputs', text: 'Hide all results. Re-run a cell to bring them back.', onClick: choose(nb.clearOutputs) },
    ];

    return (
        <div ref={ref} className="relative">
            <button
                onClick={() => setOpen(o => !o)}
                disabled={disabled}
                aria-haspopup="menu"
                aria-expanded={open}
                className="flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-semibold text-ink-2 hover:border-brand-300 hover:text-brand-700 disabled:opacity-50"
            >
                <Cpu className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Kernel</span>
                <ChevronDown className="h-3 w-3" />
            </button>
            {open && (
                <div role="menu" className="absolute right-0 top-full z-30 mt-1.5 w-72 rounded-xl border border-line bg-surface p-1.5 shadow-lg animate-in">
                    {items.map(({ icon: Icon, label, text, onClick }) => (
                        <button key={label} role="menuitem" onClick={onClick} className="flex w-full gap-2.5 rounded-lg px-2.5 py-2 text-left hover:bg-surface-2">
                            <Icon className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" />
                            <span>
                                <span className="block text-sm font-semibold text-ink">{label}</span>
                                <span className="block text-xs leading-relaxed text-ink-3">{text}</span>
                            </span>
                        </button>
                    ))}
                    <p className="mt-1 border-t border-line px-2.5 pt-2 text-[11px] leading-relaxed text-ink-3">
                        The target choice and the final exam are never re-run automatically.
                    </p>
                </div>
            )}

            <ConfirmDialog
                open={Boolean(confirm)}
                title={confirm === 'restartRun' ? 'Restart and run everything?' : 'Restart the notebook?'}
                confirmLabel={confirm === 'restartRun' ? 'Restart & run all' : 'Restart'}
                busy={nb.isProcessing}
                onCancel={() => setConfirm(null)}
                onConfirm={() => {
                    const runAfter = confirm === 'restartRun';
                    setConfirm(null);
                    if (runAfter) nb.runAll(true);
                    else nb.restart();
                }}
            >
                <p>
                    Trained models and the AI&apos;s conversation memory will be cleared, and the training rows rebuilt from
                    the raw data by replaying your recipe. Your cells and recipe stay.
                </p>
                {confirm === 'restartRun' && <p className="mt-2">Then every chart is re-run and every model retrained, top to bottom.</p>}
            </ConfirmDialog>
        </div>
    );
}
