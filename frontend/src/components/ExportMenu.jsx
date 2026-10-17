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
    const cleanOnly = Boolean(nb.summary?.clean_only);
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

    // Clean-only notebooks: no test rows and no model, so the export is simply the whole cleaned table
    // With a sealed envelope, the visible rows are what's being explored; every row opens the envelope
    const sealed = nb.summary?.sealed_rows || 0;
    const items = cleanOnly ? [
        {
            icon: Table2, label: sealed ? 'Cleaned data (visible rows)' : 'Cleaned data',
            text: sealed ? 'The rows you’re exploring, after your steps, as CSV.' : 'Every row, after your steps, as CSV.',
            onClick: download('train', 'csv'),
        },
        { icon: FileSpreadsheet, label: sealed ? 'Cleaned data (visible rows, Excel)' : 'Cleaned data (Excel)', text: 'Same, as an .xlsx file.', onClick: download('train', 'xlsx') },
        ...(sealed ? [{
            icon: Table2, label: 'Every row, the sealed ones too',
            text: `Opens the envelope: the ${sealed.toLocaleString()} sealed rows count as seen, so a later prediction’s score will say so.`,
            onClick: download('all', 'csv'),
        }] : []),
        {
            icon: FileCode2, label: 'Cleaning as a Python script',
            text: 'Your steps as one runnable .py file that cleans the original file the same way.',
            onClick: download('script'),
        },
    ] : [
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
        <div ref={ref} className="sm:relative">
            <button
                onClick={() => setOpen(o => !o)}
                disabled={disabled}
                aria-haspopup="menu"
                aria-expanded={open}
                aria-label="Export"
                title="Export"
                className="flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-semibold text-ink-2 hover:border-brand-300 hover:text-brand-700 disabled:opacity-50"
            >
                <Download className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{busy ? 'Exporting…' : 'Export'}</span>
                <ChevronDown className="h-3 w-3" />
            </button>
            {open && (
                <div role="menu" className="absolute inset-x-3 top-full z-30 mt-1.5 rounded-xl border border-line bg-surface p-1.5 shadow-lg animate-in sm:inset-x-auto sm:right-0 sm:w-80">
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
