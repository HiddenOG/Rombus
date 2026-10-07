/**
 * useSession — Custom hook for session state management.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { uploadFile, sendChat, executeAction as execAction } from '../lib/api';

export function useSession() {
    const [sessionId, setSessionId] = useState(null);
    const [currentStage, setCurrentStage] = useState('ACQUIRE');
    const [summary, setSummary] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState(null);

    const chatEndRef = useRef(null);

    const scrollToBottom = useCallback(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, scrollToBottom]);

    const addMessage = useCallback((role, content, suggestions = [], evidence = null) => {
        setChatHistory(prev => [
            ...prev,
            { role, content, suggestions, evidence, timestamp: new Date() },
        ]);
    }, []);

    const clearError = useCallback(() => setError(null), []);

    const handleUpload = useCallback(async (file) => {
        setIsProcessing(true);
        setError(null);
        try {
            const data = await uploadFile(file);
            setSessionId(data.session_id);
            setCurrentStage(data.current_stage);
            setSummary(data.summary);
            addMessage('system', `Dataset "${file.name}" uploaded successfully! Stage: ${data.current_stage}`);
            addMessage('ai', "I've analyzed your dataset. We're following the ASEMIC workflow. What would you like to do next?");

            // Auto-analyze
            const chatData = await sendChat(
                data.session_id,
                'Provide a brief initial summary of the dataset and suggest the first step in the ASEMIC workflow.'
            );
            if (chatData) {
                addMessage('ai', chatData.response, chatData.suggested_actions);
                setCurrentStage(chatData.current_stage);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    }, [addMessage]);

    const handleChat = useCallback(async (message) => {
        if (!sessionId || !message.trim()) return;
        setIsProcessing(true);
        setError(null);
        addMessage('user', message);

        try {
            const data = await sendChat(sessionId, message);
            addMessage('ai', data.response, data.suggested_actions);
            setCurrentStage(data.current_stage);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    }, [sessionId, addMessage]);

    const handleAction = useCallback(async (action) => {
        if (!sessionId) return;
        setIsProcessing(true);
        setError(null);

        try {
            const data = await execAction(sessionId, action.tool, action.arguments);

            if (data.success) {
                setSummary(data.new_state_summary);
                if (data.stage_transition) setCurrentStage(data.stage_transition);
                addMessage('system', data.message);

                if (data.evidence) {
                    addMessage('ai', `Analysis results for ${action.tool}:`, [], data.evidence);
                }

                // Auto-ask for next step
                const chatData = await sendChat(
                    sessionId,
                    `I've executed the ${action.tool} action. What should we do now?`
                );
                if (chatData) {
                    addMessage('ai', chatData.response, chatData.suggested_actions);
                    setCurrentStage(chatData.current_stage);
                }
            } else {
                setError(data.message || 'Action failed.');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    }, [sessionId, addMessage]);

    return {
        sessionId,
        currentStage,
        summary,
        chatHistory,
        isProcessing,
        error,
        chatEndRef,
        handleUpload,
        handleChat,
        handleAction,
        clearError,
    };
}
