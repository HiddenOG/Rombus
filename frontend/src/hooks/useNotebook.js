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
 * Cells are saved on the server with the notebook, so any browser can reopen it. The browser
 * only remembers which notebook was open last (and holds older cells for a one-time migration).
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
    // What this server allows (e.g. custom code is off on a public deployment)
    const [config, setConfig] = useState({ custom_code: false });
    const [notebooks, setNotebooks] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState(null);

    // Remember which notebook is open. Skipped while none is open, so the first render can't
    // wipe "last open notebook" before it's restored.
    useEffect(() => {
        if (!sessionId) return;
        saveStore({ ...loadStore(), current: sessionId });
    }, [sessionId]);

    // Save cells to the server, waiting for a short pause so bursts of changes send once
    useEffect(() => {
        if (!sessionId) return undefined;
        const timer = setTimeout(() => {
            api.saveCells(sessionId, cells).catch(() => {
                // Saving cells is best-effort; the recipe itself is already saved on the server
            });
        }, 600);
        return () => clearTimeout(timer);
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
            const [data, saved] = await Promise.all([api.getSession(id), api.getCells(id)]);
            // One-time migration: notebooks from before cells lived on the server
            const legacy = loadStore().cells?.[id] || [];
            setSessionId(id);
            setSummary(data.summary);
            setCells(saved.cells.length ? saved.cells : legacy);
        } catch (err) {
            setError(`Couldn't open that notebook: ${err.message}`);
        }
    }, []);

    // First load: tool schemas, the notebook list, and the notebook that was open last time
    useEffect(() => {
        (async () => {
            try {
                const [{ tools: list }, serverConfig] = await Promise.all([api.getTools(), api.getConfig()]);
                setTools(Object.fromEntries(list.map(t => [t.name, t])));
                setConfig(serverConfig);
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

    const deleteNotebook = async (id) => {
        try {
            await api.deleteNotebook(id);
            const store = loadStore();
            if (store.cells) delete store.cells[id];
            saveStore(store);
            if (id === sessionId) newNotebook();
            refreshNotebooks();
        } catch (err) {
            setError(`Couldn't delete that notebook: ${err.message}`);
        }
    };

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
            patchCell(id, {
                status: data.failed ? 'error' : 'done',
                reply: data.response,
                suggestions: data.suggested_actions,
                provider: data.provider,
                model: data.model,
                fallbackNote: data.fallback_note,
            });
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
        patchCell(cell.id, { message: data.message, evidence: data.evidence, version: data.new_state_summary?.data_version, cleared: false });
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

    // Actions and views: run your own code in this cell's place (remembering what it was, for Revert)
    const customizeCell = (cell, customTool, code) => busy(async () => {
        const basedOn = cell.tool.startsWith('custom_') ? cell.args?.based_on : { tool: cell.tool, args: cell.args };
        const args = { code, ...(basedOn ? { based_on: basedOn } : {}) };
        const data = await api.executeAction(sessionId, customTool, args);
        if (!data.success) {
            setError(data.message || 'Your code failed.');
            return false;
        }
        setSummary(data.new_state_summary);
        patchCell(cell.id, {
            tool: customTool, args, message: data.message, evidence: data.evidence,
            version: data.new_state_summary?.data_version,
        });
        return true;
    });

    const revertAction = (cell) => busy(async () => {
        const original = cell.args?.based_on;
        if (!original) return;
        const data = await api.executeAction(sessionId, original.tool, original.args || {});
        if (!data.success) {
            setError(data.message || 'Revert failed.');
            return;
        }
        setSummary(data.new_state_summary);
        patchCell(cell.id, {
            tool: original.tool, args: original.args || {}, message: data.message, evidence: data.evidence,
            version: data.new_state_summary?.data_version,
        });
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

    // Delete several cells; the recipe steps among them are undone together, as one change
    const deleteCells = (ids) => busy(async () => {
        const chosen = cells.filter(c => ids.includes(c.id));
        const stepIds = chosen.map(c => c.stepId).filter(id => id && summary?.recipe?.some(s => s.id === id));
        if (stepIds.length) {
            const data = await api.deleteSteps(sessionId, stepIds);
            setSummary(data.new_state_summary);
            addCell({ type: 'note', text: data.message });
        }
        setCells(prev => prev.filter(c => !ids.includes(c.id)));
        return true;
    });

    const copyCell = (cell) => run({ tool: cell.tool, arguments: cell.args }, cell.id);

    // Change an action cell's settings and run it again in place (e.g. re-lock at 30%)
    const editAction = (cell, args) => busy(async () => {
        const data = await api.executeAction(sessionId, cell.tool, args);
        if (!data.success) {
            setError(data.message || 'That change failed.');
            return false;
        }
        setSummary(data.new_state_summary);
        patchCell(cell.id, { args, message: data.message, evidence: data.evidence, version: data.new_state_summary?.data_version });
        return true;
    });

    // Swap a step with its neighbour in the recipe, then show the cells in the new order
    const moveCell = (cell, direction) => busy(async () => {
        const data = await api.moveStep(sessionId, cell.stepId, direction);
        setSummary(data.new_state_summary);
        const order = data.new_state_summary.recipe.map(s => s.id);
        setCells(prev => {
            const stepCells = prev.filter(c => c.stepId && order.includes(c.stepId))
                .sort((a, b) => order.indexOf(a.stepId) - order.indexOf(b.stepId));
            let i = 0;
            return prev.map(c => (c.stepId && order.includes(c.stepId) ? stepCells[i++] : c));
        });
    });

    // ── Kernel controls (like Jupyter's) ─────────────────────────────────────
    // Never re-run automatically: choosing the target (re-lock), the final exam (another look at the
    // test rows), and one-off actions
    const NO_AUTO_RERUN = new Set(['select_target', 'clean_only', 'final_exam', 'undo', 'save_model', 'transition_stage']);

    const restart = () => busy(async () => {
        const data = await api.restartSession(sessionId);
        setSummary(data.new_state_summary);
        addCell({ type: 'note', text: data.message });
    });

    const runAll = (restartFirst) => busy(async () => {
        if (restartFirst) {
            const data = await api.restartSession(sessionId);
            setSummary(data.new_state_summary);
        }
        const toRun = cells.filter(c => c.type === 'run' && !c.stepId && !NO_AUTO_RERUN.has(c.tool));
        let ok = 0;
        for (const cell of toRun) {
            const data = await api.executeAction(sessionId, cell.tool, cell.args || {});
            if (data.success) {
                ok += 1;
                setSummary(data.new_state_summary);
                patchCell(cell.id, { message: data.message, evidence: data.evidence, version: data.new_state_summary?.data_version, cleared: false });
            } else {
                patchCell(cell.id, { message: `⚠ ${data.message}`, evidence: null });
            }
        }
        const skipped = cells.filter(c => c.type === 'run' && NO_AUTO_RERUN.has(c.tool)).length;
        addCell({
            type: 'note',
            text: `${restartFirst ? 'Restarted, rebuilt the data from scratch, and re-ran' : 'Re-ran'} ${ok} of ${toRun.length} cells. ` +
                (skipped ? `Skipped ${skipped} (target, final exam and one-off actions are never re-run automatically).` : ''),
        });
    });

    const clearOutputs = () => setCells(prev => prev.map(c => (c.type === 'run' ? { ...c, evidence: null, cleared: true } : c)));

    return {
        sessionId,
        summary,
        cells,
        tools,
        config,
        notebooks,
        isProcessing,
        error,
        clearError: () => setError(null),
        showError: setError,
        upload,
        ask,
        run,
        rerun,
        editStep,
        editCode,
        revertCell,
        customizeCell,
        revertAction,
        editAction,
        moveCell,
        restart,
        runAll,
        clearOutputs,
        deleteCell,
        deleteCells,
        copyCell,
        openNotebook,
        newNotebook,
        deleteNotebook,
        refreshNotebooks,
    };
}
