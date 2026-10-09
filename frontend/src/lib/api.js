/**
 * API Client — fetch wrapper with configurable base URL and error handling.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';

/**
 * Make an API request with standard error handling.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_URL}${endpoint}`;
    const headers = {
        ...options.headers,
    };

    if (API_KEY) {
        headers['X-API-Key'] = API_KEY;
    }

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

export async function getTools() {
    return apiRequest('/api/tools', { method: 'GET' });
}

export async function updateStep(sessionId, stepId, args) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}`, {
        method: 'PUT',
        body: JSON.stringify({ args }),
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

export async function deleteStep(sessionId, stepId) {
    return apiRequest(`/api/session/${sessionId}/recipe/${stepId}`, { method: 'DELETE' });
}

export async function healthCheck() {
    return apiRequest('/api/health', { method: 'GET' });
}
