/**
 * API Client — fetch wrapper with configurable base URL and error handling.
 */

import { accessToken } from './supabase';

/** The backend's address as typed in a dashboard, made usable: "api.example.com/" -> "https://api.example.com".
 *  (Without https:// the browser would read it as a page on this site, which answers with the website.) */
export function backendAddress(value) {
    let url = (value || 'http://localhost:8000').trim().replace(/\/+$/, '');
    if (!/^https?:\/\//i.test(url)) url = `${/^(localhost|127\.)/.test(url) ? 'http' : 'https'}://${url}`;
    return url;
}

const API_URL = backendAddress(import.meta.env.VITE_API_URL);

/**
 * This browser's visitor id: random and unguessable, made once and kept in localStorage.
 * The server only shows a notebook to the visitor who created it.
 */
function visitorId() {
    const KEY = 'automl.visitor';
    try {
        let id = localStorage.getItem(KEY);
        if (!id) {
            id = (crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`).replace(/[^A-Za-z0-9_-]/g, '');
            localStorage.setItem(KEY, id);
        }
        return id;
    } catch {
        // Storage blocked (private mode): a per-tab id still works until the tab closes
        visitorId.fallback ||= crypto.randomUUID?.().replace(/-/g, '') || `${Date.now()}${Math.random().toString(36).slice(2)}`;
        return visitorId.fallback;
    }
}

/**
 * Who's asking: the login token when signed in (notebooks follow the account), plus this
 * browser's visitor id (used when signed out, and to move its notebooks in on sign-in).
 */
async function identity() {
    const token = await accessToken();
    return { 'X-Visitor-Id': visitorId(), ...(token ? { Authorization: `Bearer ${token}` } : {}) };
}

/** A readable message from an error reply. Validation errors (422) carry a list, not a sentence. */
function errorText(data, status) {
    const { detail } = data || {};
    if (Array.isArray(detail)) {
        return detail.map(d => d.msg?.replace(/^Value error, /, '') || String(d)).join(' ');
    }
    if (status === 413) return detail || 'That request is too large.';
    if (status === 429) return detail || "You're going a bit fast. Please wait a moment.";
    return detail || data?.message || `HTTP ${status}`;
}

/**
 * Make an API request with standard error handling.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_URL}${endpoint}`;
    const headers = {
        ...options.headers,
        ...(await identity()),
    };

    // Don't set Content-Type for FormData (browser sets it with boundary)
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }

    try {
        const response = await fetch(url, { ...options, headers });
        const isJson = (response.headers.get('content-type') || '').includes('application/json');
        if (response.ok && !isJson) {
            // A web page where data was expected: the address isn't the Rhombus backend
            throw new Error(`The backend address (${API_URL}) answered with a web page, not the Rhombus API. ` +
                'Set VITE_API_URL to the backend service\'s domain and redeploy the frontend.');
        }
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(errorText(data, response.status));
        }

        return data;
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error(`Cannot reach the backend at ${API_URL}. It may be down or still starting, or it ` +
                "doesn't allow this site yet (its CORS_ORIGINS must be this page's address).", { cause: error });
        }
        throw error;
    }
}

/** One file, or several (a FileList / array) to stack into one table with a source_file column. */
export async function uploadFile(fileOrFiles) {
    const formData = new FormData();
    const list = fileOrFiles instanceof File ? [fileOrFiles] : Array.from(fileOrFiles);
    if (list.length === 1) formData.append('file', list[0]);
    else list.forEach(f => formData.append('files', f));
    return apiRequest('/api/upload', { method: 'POST', body: formData });
}

/** The notebook assistant: an answer, or one code cell ({reply, code?, kind?}) for the user to run. */
export async function assist(sessionId, message) {
    return apiRequest('/api/assist', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, message }),
    });
}

export async function sendChat(sessionId, message) {
    return apiRequest('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, message }),
    });
}

export async function executeAction(sessionId, tool, args) {
    return apiRequest('/api/action', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, tool, arguments: args }),
    });
}

export async function getSession(sessionId) {
    return apiRequest(`/api/session/${sessionId}`, { method: 'GET' });
}

export async function getCells(sessionId) {
    return apiRequest(`/api/session/${sessionId}/cells`, { method: 'GET' });
}

export async function saveCells(sessionId, cells) {
    return apiRequest(`/api/session/${sessionId}/cells`, {
        method: 'PUT',
        body: JSON.stringify({ cells }),
    });
}

export async function deleteNotebook(sessionId) {
    return apiRequest(`/api/session/${sessionId}`, { method: 'DELETE' });
}

export async function listSessions() {
    return apiRequest('/api/sessions', { method: 'GET' });
}

/** Download an export (cleaned data or the pipeline script) and save it as a file. */
export async function downloadExport(sessionId, what, format = 'csv') {
    const params = new URLSearchParams({ what, format });
    const response = await fetch(`${API_URL}/api/session/${sessionId}/export?${params}`, {
        headers: await identity(),
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(errorText(data, response.status));
    }
    const name = /filename="([^"]+)"/.exec(response.headers.get('content-disposition') || '')?.[1] || 'export';
    const url = URL.createObjectURL(await response.blob());
    const link = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** After signing in: move the notebooks this browser made anonymously into the account. */
export async function claimNotebooks() {
    return apiRequest('/api/account/claim', { method: 'POST' });
}

export async function getConfig() {
    return apiRequest('/api/config', { method: 'GET' });
}

export async function getTools() {
    return apiRequest('/api/tools', { method: 'GET' });
}

export async function updateStep(sessionId, stepId, args) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}`, {
        method: 'PUT',
        body: JSON.stringify({ args }),
    });
}

export async function getActionCode(sessionId, tool, args) {
    return apiRequest(`/api/session/${sessionId}/code`, {
        method: 'POST',
        body: JSON.stringify({ tool, args }),
    });
}

export async function updateStepCode(sessionId, stepId, code) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}/code`, {
        method: 'PUT',
        body: JSON.stringify({ code }),
    });
}

export async function revertStep(sessionId, stepId) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}/revert`, { method: 'POST' });
}

export async function moveStep(sessionId, stepId, direction) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}/move`, {
        method: 'POST',
        body: JSON.stringify({ direction }),
    });
}

export async function restartSession(sessionId) {
    return apiRequest(`/api/session/${sessionId}/restart`, { method: 'POST' });
}

/** Undo several recipe steps as one change (one replay; refused as a whole if the rest can't run). */
export async function deleteSteps(sessionId, stepIds) {
    return apiRequest(`/api/session/${sessionId}/recipe/delete`, {
        method: 'POST',
        body: JSON.stringify({ step_ids: stepIds }),
    });
}

export async function deleteStep(sessionId, stepId) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}`, { method: 'DELETE' });
}

export async function healthCheck() {
    return apiRequest('/api/health', { method: 'GET' });
}
