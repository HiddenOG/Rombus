import React, { useEffect, useRef } from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * ConfirmDialog — "are you sure?" for actions that can't be undone.
 * Cancel has focus by default (so Enter never deletes by accident); Escape or a click outside cancels.
 */
export function ConfirmDialog({ open, title, children, confirmLabel = 'Delete', onConfirm, onCancel, busy }) {
    const cancelRef = useRef(null);

    useEffect(() => {
        if (!open) return undefined;
        cancelRef.current?.focus();
        const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open, onCancel]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-ink/40 p-4 backdrop-blur-[2px] animate-in sm:items-center" onClick={onCancel}>
            <div
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="confirm-title"
                aria-describedby="confirm-body"
                onClick={e => e.stopPropagation()}
                className="w-full max-w-md rounded-2xl border border-line bg-surface p-5 shadow-xl"
            >
                <div className="flex gap-3.5">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-50">
                        <AlertTriangle className="h-5 w-5 text-red-600" />
                    </div>
                    <div className="min-w-0">
                        <h2 id="confirm-title" className="text-base font-semibold text-ink">{title}</h2>
                        <div id="confirm-body" className="mt-1.5 text-sm leading-relaxed text-ink-2">{children}</div>
                    </div>
                </div>
                <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                    <button
                        ref={cancelRef}
                        onClick={onCancel}
                        className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-ink-2 hover:bg-surface-2 focus:outline-none focus:ring-2 focus:ring-brand-200"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={onConfirm}
                        disabled={busy}
                        className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-200 disabled:opacity-50"
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
