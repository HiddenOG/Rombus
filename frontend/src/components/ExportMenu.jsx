import React, { useEffect, useRef, useState } from 'react';
import { Download, ChevronDown, FileSpreadsheet, Table2, FileCode2, Boxes } from 'lucide-react';
import { downloadExport } from '../lib/api';

/**
 * ExportMenu — download the cleaned data or the whole recipe as a runnable Python script.
 */
export function ExportMenu({ nb }) {
    const [open, setOpen] = useState(false);
    const [busy, setBusy] = useState(false);
    const ref = useRef(null);
    const locked = Boolean(nb.summary?.is_split);
    const trained = Boolean(nb.summary?.trained_models?.length);
    const disabled = !nb.sessionId || nb.isProcessing || busy;

    useEffect(() => {
        if (!open) return undefined;
        const onClick = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [open]);

    const download = (what, format) => async () => {
        setOpen(false);
        setBusy(true);
        try {
            await downloadExport(nb.sessionId, what, format);
        } catch (err) {
            nb.showError(err.message);
        } finally {
            setBusy(false);
        }
    };

    const items = [
        { icon: Table2, label: 'Cleaned training rows', text: 'The data after your recipe, as CSV.', onClick: download('train', 'csv') },
        { icon: FileSpreadsheet, label: 'Cleaned training rows (Excel)', text: 'Same, as an .xlsx file.', onClick: download('train', 'xlsx') },
        {
            icon: Table2, label: 'All rows, cleaned', disabled: !locked,
            text: 'Training and test rows (marked in a "split" column). Test rows use the training sticky notes.',
            onClick: download('all', 'csv'),
        },
        {
            icon: FileCode2, label: 'Pipeline as Python script', disabled: !locked,
            text: 'Your whole recipe as one runnable .py file: split, learn, apply, save. Reproduces this notebook.',
            onClick: download('script'),
        },
        {
            icon: Boxes, label: 'Trained model (.pkl)', disabled: !trained,
            hint: 'Train a model first.',
            text: 'The best model, plus the recipe and its sticky notes to prepare new rows. Load it with pickle.',
            onClick: download('model'),
        },
    ];

    return (
        <div ref={ref} className="relative">
            <button
                onClick={() => setOpen(o => !o)}
                disabled={disabled}
                aria-haspopup="menu"
                aria-expanded={open}
                title="Export"
                className="flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-semibold text-ink-2 hover:border-brand-300 hover:text-brand-700 disabled:opacity-50"
            >
                <Download className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{busy ? 'Exporting…' : 'Export'}</span>
                <ChevronDown className="h-3 w-3" />
            </button>
            {open && (
                <div role="menu" className="absolute right-0 top-full z-30 mt-1.5 w-80 max-w-[90vw] rounded-xl border border-line bg-surface p-1.5 shadow-lg animate-in">
                    {items.map(({ icon: Icon, label, text, hint = 'Choose the target first.', onClick, disabled: off }) => (
                        <button
                            key={label}
                            role="menuitem"
                            onClick={onClick}
                            disabled={off}
                            className="flex w-full gap-2.5 rounded-lg px-2.5 py-2 text-left hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            <Icon className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" />
                            <span>
                                <span className="block text-sm font-semibold text-ink">{label}</span>
                                <span className="block text-xs leading-relaxed text-ink-3">{off ? hint : text}</span>
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
