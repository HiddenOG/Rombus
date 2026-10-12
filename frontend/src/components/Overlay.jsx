import React from 'react';
import { createPortal } from 'react-dom';

/**
 * Overlay: a dimmed full-screen layer for dialogs, rendered straight into <body>.
 *
 * Why the portal: `position: fixed` is relative to the window ONLY if no ancestor has a transform,
 * filter or backdrop-filter. The notebook header (backdrop-blur) and the sliding sidebar (transform)
 * both have one, so a dialog rendered inside them was centred in that small box instead of the screen.
 */
export function Overlay({ onClose, children }) {
    return createPortal(
        <div
            className="fixed inset-0 z-50 flex items-end justify-center overflow-y-auto bg-ink/40 p-4 backdrop-blur-[2px] animate-in sm:items-center"
            onClick={onClose}
        >
            {children}
        </div>,
        document.body,
    );
}
