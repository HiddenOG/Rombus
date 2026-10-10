/**
 * API Client — fetch wrapper with configurable base URL and error handling.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
 * Make an API request with standard error handling.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_URL}${endpoint}`;
    const headers = {
        ...options.headers,
        'X-Visitor-Id': visitorId(),
    };

    // Don't set Content-Type for FormData (browser sets it with boundary)
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }

    try {
        const response = await fetch(url, { ...options, headers });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.message || `HTTP ${response.status}`);
        }

        return data;
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Cannot connect to server. Is the backend running?', { cause: error });
        }
        throw error;
    }
}

export async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    return apiRequest('/api/upload', { method: 'POST', body: formData });
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
        headers: { 'X-Visitor-Id': visitorId() },
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || `Export failed (HTTP ${response.status})`);
    }
    const name = /filename="([^"]+)"/.exec(response.headers.get('content-disposition') || '')?.[1] || 'export';
    const url = URL.createObjectURL(await response.blob());
    const link = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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

export async function deleteStep(sessionId, stepId) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}`, { method: 'DELETE' });
}

export async function healthCheck() {
    return apiRequest('/api/health', { method: 'GET' });
}
