import React, { useEffect, useRef } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Overlay } from './Overlay';

// danger: can't be undone (red, warning sign). caution: worth a pause but safe (brand orange).
// Keeping red for real danger only means people still take it seriously when they see it.
const TONES = {
    danger: { icon: AlertTriangle, badge: 'bg-red-50 text-red-600',
              button: 'bg-red-600 hover:bg-red-700 focus:ring-red-200' },
    caution: { icon: RotateCcw, badge: 'bg-brand-50 text-brand-600',
               button: 'bg-brand-500 hover:bg-brand-600 focus:ring-brand-200' },
};

/**
 * ConfirmDialog — "are you sure?" before an action.
 * Cancel has focus by default (so Enter never confirms by accident); Escape or a click outside cancels.
 */
export function ConfirmDialog({ open, title, children, confirmLabel = 'Delete', onConfirm, onCancel, busy, tone = 'danger' }) {
    const { icon: Icon, badge, button } = TONES[tone];
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
        <Overlay onClose={onCancel}>
            <div
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="confirm-title"
                aria-describedby="confirm-body"
                onClick={e => e.stopPropagation()}
                className="w-full max-w-md rounded-2xl border border-line bg-surface p-5 shadow-xl"
            >
                <div className="flex gap-3.5">
                    <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${badge}`}>
                        <Icon className="h-5 w-5" />
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
                        className={`rounded-lg px-4 py-2 text-sm font-semibold text-white focus:outline-none focus:ring-2 disabled:opacity-50 ${button}`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </Overlay>
    );
}
