/**
 * useNotebook — state for the notebook: the session, its cells, and every cell action.
 *
 * Cell types:
 *   note  { text }                                  upload/status notes
 *   ask   { prompt, reply, suggestions, status }    a question to the copilot and its answer
 *   run   { tool, args, message, evidence, stepId, version }
 *         an executed action; stepId links it to its recipe step (data-changing actions only),
 *         version records the data it was made from so views can be flagged when out of date.
 *
 * Cells are kept in localStorage per notebook (session), so a refresh doesn't lose them.
 * The recipe on the server stays the source of truth for what each step does.
 */

import { useCallback, useEffect, useState } from 'react';
import * as api from '../lib/api';

const STORAGE_KEY = 'automl.notebooks.v1';

function loadStore() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY)) || { current: null, cells: {} };
    } catch {
        return { current: null, cells: {} };
    }
}

function saveStore(store) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    } catch {
        // Storage full or blocked: the notebook still works, it just won't survive a refresh
    }
}

const newId = () => Math.random().toString(36).slice(2, 10);

export function useNotebook() {
    const [sessionId, setSessionId] = useState(null);
    const [summary, setSummary] = useState(null);
    const [cells, setCells] = useState([]);
    const [tools, setTools] = useState({});
    const [notebooks, setNotebooks] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState(null);

    // Persist cells and the open notebook whenever they change. Skipped while no notebook is
    // open, so the first render can't wipe "last open notebook" before it's restored.
    useEffect(() => {
        if (!sessionId) return;
        const store = loadStore();
        store.current = sessionId;
        store.cells[sessionId] = cells;
        saveStore(store);
    }, [sessionId, cells]);

    const refreshNotebooks = useCallback(async () => {
        try {
            setNotebooks((await api.listSessions()).sessions);
        } catch {
            // The list is a convenience; ignore failures
        }
    }, []);

    const openNotebook = useCallback(async (id) => {
        setError(null);
        try {
            const data = await api.getSession(id);
            setSessionId(id);
            setSummary(data.summary);
            setCells(loadStore().cells[id] || []);
        } catch (err) {
            setError(`Couldn't open that notebook: ${err.message}`);
        }
    }, []);

    // First load: tool schemas, the notebook list, and the notebook that was open last time
    useEffect(() => {
        (async () => {
            try {
                const { tools: list } = await api.getTools();
                setTools(Object.fromEntries(list.map(t => [t.name, t])));
            } catch (err) {
                setError(err.message);
            }
            refreshNotebooks();
            const { current } = loadStore();
            if (current) openNotebook(current);
        })();
    }, [openNotebook, refreshNotebooks]);

    const newNotebook = useCallback(() => {
        saveStore({ ...loadStore(), current: null });
        setSessionId(null);
        setSummary(null);
        setCells([]);
        setError(null);
    }, []);

    const addCell = (cell) => setCells(prev => [...prev, { id: newId(), ...cell }]);
    const patchCell = (id, patch) => setCells(prev => prev.map(c => (c.id === id ? { ...c, ...patch } : c)));

    // Run a step through `fn`, with the busy flag and error banner handled once
    const busy = async (fn) => {
        setIsProcessing(true);
        setError(null);
        try {
            return await fn();
        } catch (err) {
            setError(err.message);
            return null;
        } finally {
            setIsProcessing(false);
        }
    };

    const upload = (file) => busy(async () => {
        const data = await api.uploadFile(file);
        setSessionId(data.session_id);
        setSummary(data.summary);
        setCells([{
            id: newId(),
            type: 'note',
            text: `Uploaded **${file.name}**: ${data.summary.shape[0]} rows × ${data.summary.shape[1]} columns. ` +
                'Start by choosing the column to predict. Ask the copilot, or use **Add step → Select target**.',
        }]);
        refreshNotebooks();
    });

    const ask = (prompt) => busy(async () => {
        const id = newId();
        setCells(prev => [...prev, { id, type: 'ask', prompt, status: 'thinking' }]);
        try {
            const data = await api.sendChat(sessionId, prompt);
            patchCell(id, { status: 'done', reply: data.response, suggestions: data.suggested_actions });
        } catch (err) {
            patchCell(id, { status: 'error', reply: err.message });
        }
    });

    const run = (action, fromCellId = null) => busy(async () => {
        const data = await api.executeAction(sessionId, action.tool, action.arguments || {});
        if (!data.success) {
            setError(data.message || 'Action failed.');
            return;
        }
        setSummary(data.new_state_summary);
        addCell({
            type: 'run',
            tool: action.tool,
            args: data.step?.args ?? action.arguments ?? {},
            message: data.message,
            evidence: data.evidence,
            stepId: data.step?.id ?? null,
            version: data.new_state_summary?.data_version,
            from: fromCellId,
        });
        refreshNotebooks();
    });

    // Views (charts, tables) are re-executed in place against the current data
    const rerun = (cell) => busy(async () => {
        const data = await api.executeAction(sessionId, cell.tool, cell.args || {});
        if (!data.success) {
            setError(data.message || 'Rerun failed.');
            return;
        }
        setSummary(data.new_state_summary);
        patchCell(cell.id, { message: data.message, evidence: data.evidence, version: data.new_state_summary?.data_version });
    });

    const editStep = (cell, args) => busy(async () => {
        const data = await api.updateStep(sessionId, cell.stepId, args);
        setSummary(data.new_state_summary);
        const step = data.new_state_summary.recipe.find(s => s.id === cell.stepId);
        patchCell(cell.id, { args: step?.args ?? args, message: step?.message ?? data.message });
        addCell({ type: 'note', text: data.message });
        return true;
    });

    // Saving code turns a built-in step into a custom code step (same place in the recipe)
    const editCode = (cell, code) => busy(async () => {
        const data = await api.updateStepCode(sessionId, cell.stepId, code);
        setSummary(data.new_state_summary);
        patchCell(cell.id, { tool: 'custom_code' });
        addCell({ type: 'note', text: data.message });
        return true;
    });

    const revertCell = (cell) => busy(async () => {
        const data = await api.revertStep(sessionId, cell.stepId);
        setSummary(data.new_state_summary);
        const step = data.new_state_summary.recipe.find(s => s.id === cell.stepId);
        patchCell(cell.id, { tool: step?.tool ?? cell.tool, args: step?.args ?? cell.args });
        addCell({ type: 'note', text: data.message });
    });

    const deleteCell = (cell) => busy(async () => {
        const inRecipe = cell.stepId && summary?.recipe?.some(s => s.id === cell.stepId);
        if (inRecipe) {
            const data = await api.deleteStep(sessionId, cell.stepId);
            setSummary(data.new_state_summary);
            addCell({ type: 'note', text: data.message });
        }
        setCells(prev => prev.filter(c => c.id !== cell.id));
    });

    const copyCell = (cell) => run({ tool: cell.tool, arguments: cell.args }, cell.id);

    return {
        sessionId,
        summary,
        cells,
        tools,
        notebooks,
        isProcessing,
        error,
        clearError: () => setError(null),
        upload,
        ask,
        run,
        rerun,
        editStep,
        editCode,
        revertCell,
        deleteCell,
        copyCell,
        openNotebook,
        newNotebook,
    };
}
