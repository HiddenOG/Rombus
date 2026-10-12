/**
 * useCellSelection — Jupyter-style cell selection.
 *   click             select that cell
 *   Ctrl/⌘ + click    add or remove a cell
 *   Shift + click     select a range
 *   ↑ / ↓             move the selection (wired up in SelectionBar, with Delete and Esc)
 */

import { useCallback, useState } from 'react';

export function useCellSelection(cells) {
    const [selected, setSelected] = useState([]);
    const [anchor, setAnchor] = useState(null);
    const ids = cells.map(c => c.id);

    // Ignore cells that no longer exist (deleted, or another notebook opened)
    const present = selected.filter(id => ids.includes(id));

    const click = useCallback((id, e) => {
        if (e.shiftKey && anchor && ids.includes(anchor)) {
            const [a, b] = [ids.indexOf(anchor), ids.indexOf(id)].sort((x, y) => x - y);
            setSelected(ids.slice(a, b + 1));
        } else if (e.ctrlKey || e.metaKey) {
            setSelected(s => (s.includes(id) ? s.filter(x => x !== id) : [...s, id]));
            setAnchor(id);
        } else {
            setSelected(s => (s.length === 1 && s[0] === id ? [] : [id]));
            setAnchor(id);
        }
    }, [anchor, ids]);

    const step = useCallback((direction) => {
        const current = present.length ? ids.indexOf(present[present.length - 1]) : -1;
        const next = ids[Math.min(ids.length - 1, Math.max(0, current + direction))];
        if (next) {
            setSelected([next]);
            setAnchor(next);
            document.getElementById(`cell-${next}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }, [ids, present]);

    const clear = useCallback(() => setSelected([]), []);

    return { selected: present, click, step, clear };
}
